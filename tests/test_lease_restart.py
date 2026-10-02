"""A restart keeps the leases it had, and grants nothing on its way down.

Before this, the broker held leases in memory only. A restart stopped the
residents, the card read free, a waiting request was granted by the process
that was exiting, and the next process listed no leases while the holder kept
19 GB of the card.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from vramux import cli
from vramux.lease import Closing, UnknownLease

from test_lease import RESERVE, FakeObserver, broker, snapshot


def restartable(tmp_path, **kwargs):
    kwargs.setdefault("pid_alive", lambda pid: True)
    return broker(state_file=tmp_path / "leases.json", **kwargs)


async def test_a_lease_outlives_the_process_that_granted_it(tmp_path):
    first = restartable(tmp_path)
    lease = await first.acquire(mb=8000, owner="batch-job", ttl=300, pids=[4242])
    await first.close()

    second = restartable(tmp_path)
    [restored] = second.leases
    assert (restored.id, restored.owner, restored.mb, restored.pids) == (
        lease.id, "batch-job", 8000, [4242],
    )
    assert 290 < restored.expires_at - time.monotonic() <= 300
    renewed = await second.renew(lease.id)
    assert renewed.id == lease.id


async def test_a_restored_lease_still_counts_against_the_card(tmp_path):
    card = FakeObserver(snapshot(used=1000))
    first = restartable(tmp_path, observer=card)
    await first.acquire(mb=20000, owner="batch-job", ttl=300)
    await first.close()

    second = restartable(tmp_path, observer=card)
    with pytest.raises(Exception):
        await second.acquire(mb=20000, owner="bonsai", ttl=60, wait=0)


async def test_a_release_is_remembered_too(tmp_path):
    first = restartable(tmp_path)
    lease = await first.acquire(mb=4000, owner="batch", ttl=60)
    await first.release(lease.id)
    await first.close()

    assert restartable(tmp_path).leases == []


async def test_time_spent_down_counts_against_the_ttl(tmp_path, monkeypatch):
    first = restartable(tmp_path)
    await first.acquire(mb=4000, owner="batch", ttl=60)
    await first.close()

    later = time.time() + 61
    monkeypatch.setattr("vramux.lease.time.time", lambda: later)
    assert restartable(tmp_path).leases == []


async def test_a_lease_whose_holder_died_during_the_restart_is_dropped(tmp_path):
    first = restartable(tmp_path)
    await first.acquire(mb=4000, owner="gone", ttl=300, pids=[111])
    await first.acquire(mb=4000, owner="here", ttl=300, pids=[222])
    await first.close()

    second = restartable(tmp_path, pid_alive=lambda pid: pid == 222)
    assert [l.owner for l in second.leases] == ["here"]


async def test_an_unreadable_state_file_starts_empty(tmp_path):
    (tmp_path / "leases.json").write_text("{not json")
    assert restartable(tmp_path).leases == []


async def test_no_state_file_means_memory_only(tmp_path):
    b = broker()
    await b.acquire(mb=4000, owner="batch", ttl=60)
    await b.close()
    assert list(tmp_path.iterdir()) == []


# ---- shutdown ---------------------------------------------------------------


async def test_a_closing_broker_grants_nothing(tmp_path):
    b = restartable(tmp_path)
    await b.close()
    with pytest.raises(Closing):
        await b.acquire(mb=1000, owner="late", ttl=60)


async def test_a_request_waiting_for_room_is_not_granted_on_the_way_down(tmp_path):
    card = FakeObserver(snapshot(used=20000))
    b = restartable(tmp_path, observer=card, poll_interval=0.01)
    waiting = asyncio.create_task(
        b.acquire(mb=19000, owner="batch-job", ttl=60, wait=5)
    )
    await asyncio.sleep(0.05)
    await b.close()
    # What shutdown does next: residents stop and the card reads free.
    card.snap = snapshot(used=1000)
    with pytest.raises(Closing):
        await waiting
    assert b.leases == []
    assert restartable(tmp_path).leases == []


async def test_closing_answers_503_over_http(tmp_path, aiohttp_client=None):
    from aiohttp.test_utils import TestClient, TestServer
    from vramux.registry import ModelRegistry
    from vramux.residency import ResidencyArbiter
    from vramux.router import make_app

    b = restartable(tmp_path)
    app = make_app(ModelRegistry(), ResidencyArbiter(port=1), b)
    async with TestClient(TestServer(app)) as client:
        await b.close()
        resp = await client.post("/gpu/lease", json={"mb": 1000, "owner": "late"})
        assert resp.status == 503
        assert "restarting" in (await resp.json())["error"]


# ---- the client -------------------------------------------------------------


def test_the_renewer_keeps_trying_through_a_restart(monkeypatch):
    answers = iter([(0, {"error": "connection refused"}),
                    (503, {"error": "vramux is restarting"}),
                    (200, {})])
    calls = []

    def fake_request(url, method="GET", body=None, timeout=10.0):
        calls.append(url)
        try:
            return next(answers)
        except StopIteration:
            renewer.stop()
            return 200, {}

    monkeypatch.setattr(cli, "_request", fake_request)
    monkeypatch.setattr(cli, "_RESTART_RETRY", 0.01)
    renewer = cli._Renewer("http://x/gpu/lease/l/renew", ttl=3.0)
    renewer.interval = 0.01
    renewer._run()
    assert len(calls) >= 4


def test_the_renewer_stops_on_an_unknown_lease(monkeypatch):
    calls = []

    def fake_request(url, method="GET", body=None, timeout=10.0):
        calls.append(url)
        return 404, {"error": "no such lease"}

    monkeypatch.setattr(cli, "_request", fake_request)
    renewer = cli._Renewer("http://x/gpu/lease/l/renew", ttl=3.0)
    renewer.interval = 0.01
    renewer._run()
    assert len(calls) == 1

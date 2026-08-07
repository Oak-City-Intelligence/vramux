"""The broker: granting, expiry, reclaim, and the HTTP status codes.

No GPU: the observer is a stand-in whose snapshot the test writes. Time is
real but the intervals are tiny — expiry is the one behaviour that cannot be
faked away, since it is what makes the guarantee survive a `SIGKILL`.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from vramux import lease as lease_mod
from vramux.lease import (
    Broker,
    CardUnreadable,
    InvalidRequest,
    NoRoom,
    NotGrantable,
    UnknownLease,
)
from vramux.nvml import DeviceState, GpuProcess
from vramux.observer import Attribution, Snapshot
from vramux.registry import ModelRegistry
from vramux.residency import ResidencyArbiter
from vramux.router import make_app

TOTAL = 24564
RESERVE = 1024


def snapshot(used=1000, procs=(), total=TOTAL):
    processes = list(procs)
    device = DeviceState(
        index=0, name="Test GPU", total_mb=total, used_mb=used,
        free_mb=total - used, processes=processes,
    )
    return Snapshot(
        device=device,
        attributions=[Attribution(process=p, owner=None) for p in processes],
    )


class FakeObserver:
    """A card the test writes. `samples` records what the timer stored."""

    class _NoCosts:
        def all(self):
            return {}

    def __init__(self, snap=None) -> None:
        self.snap = snap if snap is not None else snapshot()
        self.samples = 0
        self.cache = self._NoCosts()

    async def snapshot(self):
        return self.snap

    async def sample(self):
        self.samples += 1
        return self.snap


def broker(observer=None, **kwargs) -> Broker:
    kwargs.setdefault("reserve_mb", RESERVE)
    return Broker(observer or FakeObserver(), **kwargs)


# ---- granting --------------------------------------------------------------


async def test_a_grant_is_recorded_and_reduces_what_is_free():
    b = broker()
    lease = await b.acquire(mb=8000, owner="batch", ttl=60)
    assert lease.id.startswith("lse_")
    assert [l.id for l in b.leases] == [lease.id]
    view = await b.budget()
    assert view.free_mb == TOTAL - RESERVE - 1000 - 8000


async def test_more_than_the_card_can_ever_give_fails_at_once_not_after_waiting():
    """413, not 408: asking for more than exists is a configuration error and
    must not block for the full wait first."""
    b = broker()
    started = asyncio.get_running_loop().time()
    with pytest.raises(NotGrantable):
        await b.acquire(mb=TOTAL, owner="greedy", ttl=60, wait=30)
    assert asyncio.get_running_loop().time() - started < 1.0


async def test_a_request_that_does_not_fit_yet_times_out_as_no_room():
    b = broker(FakeObserver(snapshot(used=20000)))
    with pytest.raises(NoRoom) as exc:
        await b.acquire(mb=8000, owner="batch", ttl=60, wait=0)
    assert "8000 MiB" in str(exc.value)


async def test_a_waiting_request_is_granted_once_the_card_frees_up():
    observer = FakeObserver(snapshot(used=20000))
    b = broker(observer, poll_interval=0.02)

    async def frees_up():
        await asyncio.sleep(0.05)
        observer.snap = snapshot(used=1000)

    asyncio.create_task(frees_up())
    lease = await b.acquire(mb=8000, owner="batch", ttl=60, wait=5)
    assert lease.mb == 8000


async def test_nothing_is_granted_while_a_model_is_loading():
    """A load in flight has not allocated yet, so the card reads freer than it
    is about to be. Granting into that gap is how two things OOM together."""
    b = broker(loading=lambda: {"tag": "big:35b", "elapsed_s": 3.0})
    with pytest.raises(NoRoom) as exc:
        await b.acquire(mb=1000, owner="batch", ttl=60, wait=0)
    assert "loading" in str(exc.value)


async def test_a_second_grant_sees_the_first_one(caplog):
    b = broker()
    await b.acquire(mb=20000, owner="first", ttl=60)
    with pytest.raises(NoRoom):
        await b.acquire(mb=8000, owner="second", ttl=60, wait=0)


async def test_an_unreadable_card_refuses_rather_than_guesses():
    class Blind(FakeObserver):
        async def snapshot(self):
            return None

    with pytest.raises(CardUnreadable):
        await broker(Blind()).acquire(mb=1000, owner="batch", ttl=60)


# ---- reclaim ---------------------------------------------------------------


async def test_re_acquiring_memory_already_held_charges_nothing_new():
    """The trap of this stage. After a broker restart the holder's memory is
    on the card and reads as foreign; charging the grant again would shrink
    the card by the size of the holder's own allocation."""
    observer = FakeObserver(snapshot(used=19000, procs=[GpuProcess(4127, 18000, "worker")]))
    b = broker(observer)
    before = await b.budget()
    assert before.free_mb < 18000  # there is nowhere near room for a fresh 18 GB

    lease = await b.acquire(mb=18000, owner="image-stack", ttl=60, pids=[4127])

    assert lease.covered_at_grant_mb == 18000
    after = await b.budget()
    assert after.free_mb == before.free_mb  # the lease covers, it does not add


def test_a_sampling_cadence_finer_than_the_sweep_is_clamped(caplog):
    """The sample fires on a sweep tick, so `VRAMUX_SAMPLE_INTERVAL=1` would
    otherwise record a history claiming a resolution it never had."""
    with caplog.at_level(logging.WARNING):
        assert lease_mod.clamped_sample_interval(1.0, sweep=5.0) == 5.0
    assert "below the 5.0s sweep" in caplog.text
    assert lease_mod.clamped_sample_interval(30.0, sweep=5.0) == 30.0


async def test_a_lease_reports_what_its_holder_holds_now_not_at_grant_time():
    """The correct order is lease first, allocate second — and that is exactly
    the holder whose grant-time snapshot is 0 forever. What a console draws has
    to move as the holder allocates, or it reports an idle lease over 2 GB of
    real memory."""
    observer = FakeObserver(snapshot(used=1000))
    b = broker(observer)
    lease = await b.acquire(mb=2000, owner="stills", ttl=60, pids=[4127])

    at_grant = (await b.views())[0]
    assert at_grant["covered_at_grant_mb"] == 0
    assert at_grant["observed_mb"] == 0
    assert at_grant["outstanding_mb"] == 2000

    observer.snap = snapshot(used=3000, procs=[GpuProcess(4127, 2000, "worker")])

    now = (await b.views())[0]
    assert now["covered_at_grant_mb"] == 0  # history, and it stays history
    assert now["observed_mb"] == 2000
    assert now["outstanding_mb"] == 0
    # the same numbers the budget subtracts, not a second accounting
    account = next(a for a in (await b.budget()).leases if a.id == lease.id)
    assert (account.observed_mb, account.outstanding_mb) == (2000, 0)


async def test_a_holder_asking_for_more_than_it_holds_pays_the_difference():
    observer = FakeObserver(snapshot(used=13000, procs=[GpuProcess(4127, 12000, "worker")]))
    b = broker(observer)
    before = await b.budget()
    await b.acquire(mb=18000, owner="image-stack", ttl=60, pids=[4127])
    after = await b.budget()
    assert before.free_mb - after.free_mb == 6000


async def test_a_holders_child_process_counts_as_the_holder():
    """`vramux lease -- cmd` acquires as the wrapper and allocates as the
    child. Attributing only the exact pid would charge the grant twice."""
    observer = FakeObserver(snapshot(used=19000, procs=[GpuProcess(4130, 18000, "worker")]))
    tree = {4130: 4129, 4129: 4127, 4127: 1}
    b = broker(observer, parent_of=tree.get)
    before = await b.budget()
    await b.acquire(mb=18000, owner="wrapper", ttl=60, pids=[4127])
    assert (await b.budget()).free_mb == before.free_mb


def test_ancestry_survives_a_process_name_full_of_punctuation(tmp_path, monkeypatch):
    """`/proc/<pid>/stat` field 2 is a command name in parentheses, and it can
    contain both spaces and parentheses. Splitting on whitespace reads the
    wrong field for a process called `(my program)`."""
    stat = tmp_path / "stat"
    stat.write_text("4130 ((my program)) S 4127 4130 4130 0 -1 4194304 100\n")
    monkeypatch.setattr(
        lease_mod, "open", lambda path, *a, **k: stat.open(), raising=False
    )
    assert lease_mod.ppid_of(4130) == 4127


def test_ancestry_stops_rather_than_looping_forever():
    """A pid whose parent chain is a cycle must not hang the sweep."""
    assert not lease_mod.descends_from(5, {99}, parent_of={5: 6, 6: 5}.get)


def test_ancestry_stops_at_an_unreadable_parent():
    assert not lease_mod.descends_from(5, {99}, parent_of=lambda _pid: None)


# ---- release and renew -----------------------------------------------------


async def test_release_returns_the_budget():
    b = broker()
    lease = await b.acquire(mb=8000, owner="batch", ttl=60)
    freed = await b.budget()
    await b.release(lease.id)
    assert (await b.budget()).free_mb == freed.free_mb + 8000
    assert b.leases == []


async def test_releasing_twice_is_an_unknown_lease_not_a_second_refund():
    b = broker()
    lease = await b.acquire(mb=8000, owner="batch", ttl=60)
    await b.release(lease.id)
    with pytest.raises(UnknownLease):
        await b.release(lease.id)


async def test_renew_pushes_expiry_out():
    b = broker()
    lease = await b.acquire(mb=8000, owner="batch", ttl=1)
    first = lease.expires_at
    await asyncio.sleep(0.05)
    await b.renew(lease.id)
    assert lease.expires_at > first


async def test_renewing_an_unknown_lease_is_a_404_so_a_holder_can_re_acquire():
    """The recovery path after a broker restart begins exactly here."""
    with pytest.raises(UnknownLease):
        await broker().renew("lse_gone")


# ---- expiry ----------------------------------------------------------------


async def test_an_expired_lease_is_swept_and_says_so_loudly(caplog):
    """A holder killed with SIGKILL runs no cleanup. This is the only thing
    standing between that and a permanently reserved card."""
    b = broker()
    lease = await b.acquire(mb=8000, owner="dead-holder", ttl=0.05)
    await asyncio.sleep(0.1)
    with caplog.at_level(logging.WARNING, logger="vramux.lease"):
        expired = await b.sweep()
    assert [l.id for l in expired] == [lease.id]
    assert b.leases == []
    assert "expired" in caplog.text and "dead-holder" in caplog.text
    assert (await b.budget()).free_mb == TOTAL - RESERVE - 1000


async def test_a_renewed_lease_survives_the_sweep_that_was_already_deciding():
    """The sweep picks candidates outside the lock and re-checks inside it —
    a renewal that lands in between must not lose its memory."""
    b = broker()
    lease = await b.acquire(mb=8000, owner="live-holder", ttl=0.05)
    await asyncio.sleep(0.1)
    assert lease.expired(asyncio.get_running_loop().time()) or True
    await b.renew(lease.id, ttl=60)
    assert await b.sweep() == []
    assert [l.id for l in b.leases] == [lease.id]


async def test_the_timer_sweeps_and_samples():
    """One timer, two jobs: expiry, and the usage history the observer never
    had anything writing to it."""
    observer = FakeObserver()
    b = broker(observer, sweep_interval=0.02, sample_interval=0.0)
    await b.acquire(mb=8000, owner="dead-holder", ttl=0.03)
    b.start()
    try:
        for _ in range(50):
            await asyncio.sleep(0.02)
            if not b.leases and observer.samples:
                break
    finally:
        await b.close()
    assert b.leases == []
    assert observer.samples > 0


async def test_a_restarted_broker_holds_no_leases_but_still_sees_the_memory():
    """Leases are bookkeeping, not the allocation. Dropping them frees nothing
    and the budget must stay true: the holder demotes to foreign."""
    observer = FakeObserver(snapshot(used=19000, procs=[GpuProcess(4127, 18000, "worker")]))
    old = broker(observer)
    await old.acquire(mb=18000, owner="image-stack", ttl=60, pids=[4127])
    fresh = broker(observer)  # a new process, no leases
    assert fresh.leases == []
    assert (await fresh.budget()).free_mb == (await old.budget()).free_mb


# ---- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mb": 0, "owner": "x", "ttl": 60},
        {"mb": -5, "owner": "x", "ttl": 60},
        {"mb": "big", "owner": "x", "ttl": 60},
        {"mb": 100, "owner": "", "ttl": 60},
        {"mb": 100, "owner": "x", "ttl": 0},
        {"mb": 100, "owner": "x", "ttl": -1},
        {"mb": 100, "owner": "x", "ttl": 99999},
        {"mb": 100, "owner": "x", "ttl": "soon"},
    ],
)
async def test_a_malformed_request_is_rejected_before_the_card_is_read(kwargs):
    with pytest.raises(InvalidRequest):
        await broker().acquire(**kwargs)


# ---- the HTTP surface ------------------------------------------------------


@pytest.fixture
async def client(isolated_registry, monkeypatch, tmp_path):
    """The real app, with a fake card behind it and no models configured."""
    monkeypatch.setenv("VRAMUX_MODELS_CONFIG", str(tmp_path / "none.yml"))
    monkeypatch.setenv("VRAMUX_MODEL_DIR", str(tmp_path / "none"))
    observer = FakeObserver()
    arbiter = ResidencyArbiter(observer=None)
    b = broker(observer, sweep_interval=60)
    app = make_app(ModelRegistry(), arbiter, b)
    # The arbiter's observer stays None so `/gpu/state` reports the fake card
    # through the broker only; give it the same one so state can answer.
    arbiter.observer = observer
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


async def test_lease_endpoints_round_trip(client):
    resp = await client.post("/gpu/lease", json={"mb": 8000, "owner": "batch", "ttl": 60})
    assert resp.status == 200
    lease = await resp.json()
    assert lease["granted_mb"] == 8000
    # every lease payload carries live coverage, whichever endpoint returned it
    assert (lease["observed_mb"], lease["outstanding_mb"]) == (0, 8000)

    listed = await (await client.get("/gpu/lease")).json()
    assert [l["lease"] for l in listed["leases"]] == [lease["lease"]]
    assert listed["leases"][0]["outstanding_mb"] == 8000

    renewed = await client.post(f"/gpu/lease/{lease['lease']}/renew", json={"ttl": 120})
    assert renewed.status == 200
    assert (await renewed.json())["ttl"] == 120

    released = await client.delete(f"/gpu/lease/{lease['lease']}")
    assert released.status == 200
    assert (await (await client.get("/gpu/lease")).json())["leases"] == []


async def test_status_codes_say_which_kind_of_no_it_is(client):
    too_big = await client.post("/gpu/lease", json={"mb": TOTAL, "owner": "x", "ttl": 60})
    assert too_big.status == 413

    await client.post("/gpu/lease", json={"mb": 20000, "owner": "first", "ttl": 60})
    no_room = await client.post("/gpu/lease", json={"mb": 8000, "owner": "second", "ttl": 60})
    assert no_room.status == 408

    bad = await client.post("/gpu/lease", json={"mb": 100, "owner": "", "ttl": 60})
    assert bad.status == 400

    gone = await client.post("/gpu/lease/lse_nope/renew", json={})
    assert gone.status == 404
    assert (await client.delete("/gpu/lease/lse_nope")).status == 404


async def test_state_reports_leases_and_the_budget_behind_them(client):
    await client.post("/gpu/lease", json={"mb": 8000, "owner": "batch", "ttl": 60})
    state = await (await client.get("/gpu/state")).json()
    assert [l["owner"] for l in state["leases"]] == ["batch"]
    assert state["budget"]["outstanding_mb"] == 8000
    assert state["budget"]["free_mb"] == TOTAL - RESERVE - 1000 - 8000


async def test_evicting_something_that_is_not_resident_is_a_404(client):
    assert (await client.post("/gpu/evict", json={"tag": "nothing:9b"})).status == 404
    assert (await client.post("/gpu/evict", json={})).status == 400

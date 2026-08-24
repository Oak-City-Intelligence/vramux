"""The console: what it draws, and what the router pushes at it.

Two halves tested apart, the way they are built apart. `render()` never sees a
terminal and `StateFeed` never sees a card — between them that is the whole of
`vramux top` except the twenty lines of curses that draw strings somebody else
produced.
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from vramux.console import Feed, _bar, _duration, _iter_sse, render
from vramux.observer import UsageLog
from vramux.registry import ModelRegistry
from vramux.residency import ResidencyArbiter
from vramux.router import ROUTER, StateFeed, make_app

from test_lease import snapshot
from test_residency import packing_arbiter, spec


# ---- a state payload to draw ----------------------------------------------


def state(**overrides) -> dict:
    payload = {
        "device": {"index": 0, "name": "NVIDIA GeForce RTX 4090", "total_mb": 24564,
                   "used_mb": 14009, "free_mb": 10555, "unattributed_mb": 800},
        "recognised_mb": 13599,
        "foreign_mb": 410,
        "processes": [
            {"pid": 2185, "used_mb": 240, "name": "/usr/bin/compositor", "owner": None},
            {"pid": 999, "used_mb": 9000, "name": "llama-server", "owner": "a:9b"},
        ],
        "unlocated_owners": [],
        "residents": ["a:9b", "b:12b"],
        "resident_detail": [
            {"tag": "a:9b", "port": 18080, "inflight": 1, "cost_mb": 6591,
             "last_use": "2026-08-07T12:00:00+00:00"},
            {"tag": "b:12b", "port": 18081, "inflight": 0, "cost_mb": None,
             "last_use": None},
        ],
        "loading": None,
        "costs": {},
        "leases": [],
        "budget": {"total_mb": 24564, "reserve_mb": 1024, "used_mb": 14009,
                   "recognised_mb": 13599, "foreign_mb": 410, "unattributed_mb": 800,
                   "granted_mb": 0, "outstanding_mb": 0, "ceiling_mb": 23540,
                   "free_mb": 9531},
    }
    payload.update(overrides)
    return payload


# noon on the day the sample timestamps above are from
NOON = 1786104000.0  # 2026-08-07T12:00:00Z


def text(lines) -> str:
    return "\n".join(line.text for line in lines)


def styles(lines, needle: str) -> str:
    return next(line.style for line in lines if needle in line.text)


# ---- rendering -------------------------------------------------------------


def test_every_resident_gets_a_row_including_the_one_with_no_known_cost():
    """A model with no measured cost is exactly the one being served alone.
    Drawing a blank there and a number elsewhere is the point."""
    drawn = text(render(state(), width=100, now=NOON + 30))
    assert "a:9b" in drawn and "b:12b" in drawn
    assert "6 591" in drawn, "a measured cost is shown"
    assert "—" in drawn, "an unknown cost is a dash, not a zero"


def test_idle_is_computed_from_the_absolute_timestamp_the_router_sent():
    """The router deliberately sends `last_use` rather than an age, so the age
    has to be worked out here — and it has to keep counting between frames."""
    early = text(render(state(), width=100, now=NOON + 30))
    later = text(render(state(), width=100, now=NOON + 3600))
    assert "30s" in early
    assert "1h" in later


def test_a_lease_nobody_has_allocated_against_is_marked():
    lease = {"lease": "abc", "owner": "batch", "granted_mb": 12000,
             "observed_mb": 980, "outstanding_mb": 11020, "priority": 5,
             "expires_at": "2026-08-07T12:01:00+00:00", "ttl": 60}
    lines = render(state(leases=[lease]), width=100, now=NOON)
    assert styles(lines, "batch") == "warn"
    assert "11 020" in text(lines)
    assert "1m" in text(lines), "TTL counts down from the absolute expiry"


def test_a_lease_that_has_allocated_is_not_marked():
    lease = {"lease": "abc", "owner": "batch", "granted_mb": 12000,
             "observed_mb": 12100, "outstanding_mb": 0, "priority": 5,
             "expires_at": "2026-08-07T12:01:00+00:00", "ttl": 60}
    assert styles(render(state(leases=[lease]), now=NOON), "batch") == ""


def test_the_headline_goes_hot_when_nothing_more_can_be_granted():
    """`free_mb` at or below the reserve means the next request is refused.
    That is the number worth colouring, not raw usage."""
    tight = state()
    tight["budget"]["free_mb"] = 512
    assert styles(render(tight, width=100), "used") == "hot"
    assert styles(render(state(), width=100), "used") == ""


def test_only_foreign_processes_are_listed_as_foreign():
    """A resident's own llama-server is on the card and is not foreign. Listing
    it under FOREIGN would double it in the reader's head, having already been
    counted as a resident above."""
    drawn = text(render(state(), width=100))
    assert "compositor" in drawn
    assert "llama-server" not in drawn


def test_a_leaseholder_is_not_listed_again_as_foreign():
    """Its memory is already on screen as HELD. Drawing the same process under
    FOREIGN as well reads as two allocations of the same 2 386 MiB — which is
    the exact confusion the broker's one-accounting rule exists to avoid."""
    lease = {"lease": "abc", "owner": "probe", "granted_mb": 2000,
             "observed_mb": 2386, "outstanding_mb": 0, "priority": 5,
             "pids": [1939837], "expires_at": "2026-08-07T12:01:00+00:00", "ttl": 60}
    holder = {"pid": 1939837, "used_mb": 2386, "name": "python3", "owner": None}
    drawn = text(render(state(leases=[lease], processes=[holder]), width=100, now=NOON))
    assert "probe" in drawn
    assert "1939837" not in drawn


def test_the_process_list_is_bounded_by_the_terminal_and_says_what_it_hid():
    crowd = [{"pid": 100 + i, "used_mb": 100 + i, "name": f"p{i}", "owner": None}
             for i in range(40)]
    drawn = text(render(state(processes=crowd), width=100, height=24))
    assert "more" in drawn
    assert len(drawn.splitlines()) <= 24


def test_a_load_in_progress_is_drawn_rather_than_looking_like_a_stall():
    drawn = text(render(state(loading={"tag": "big:27b", "elapsed_s": 12.4,
                                       "budget_s": 180.0})))
    assert "loading big:27b" in drawn and "12.4s" in drawn


def test_an_older_router_without_resident_detail_still_names_its_residents():
    """The console must survive being newer than the router it points at: the
    fallback is the tag list `/gpu/state` has always had."""
    old = state()
    del old["resident_detail"]
    assert "a:9b" in text(render(old, width=100))


def test_an_error_payload_is_shown_instead_of_a_blank_screen():
    assert "no GPU visible" in text(render({"error": "no GPU visible"}))
    assert "waiting" in text(render(None))
    # Transport status never replaces the message: "polling" alone on a blank
    # screen reads as a card with nothing on it.
    blank = text(render(None, note="polling"))
    assert "waiting" in blank and "polling" in blank


def test_a_long_tag_is_clipped_to_the_terminal_rather_than_wrapping():
    long_tag = state()
    long_tag["resident_detail"][0]["tag"] = "x" * 200
    for line in render(long_tag, width=60):
        assert len(line.text) <= 60


def test_the_bar_draws_the_reserve_as_neither_used_nor_free():
    bar = _bar(used=12000, total=24000, reserve=2400, width=20)
    assert bar.count("#") == 10 and bar.count(":") == 2
    assert len(bar) == 22, "brackets plus exactly the width asked for"
    assert _bar(0, 0, 0, 10).strip("[] ") == "", "an unreadable card draws empty"


@pytest.mark.parametrize("seconds,expected", [
    (-5, "0s"), (0, "0s"), (59, "59s"), (60, "1m"), (3599, "59m"), (7200, "2h"),
])
def test_durations_stay_short_enough_for_a_column(seconds, expected):
    assert _duration(seconds) == expected


# ---- the transport ---------------------------------------------------------


def test_sse_framing_skips_keepalives_and_survives_a_broken_frame():
    raw = [b"event: state\n", b'data: {"a": 1}\n', b"\n", b": keepalive\n", b"\n",
           b"data: not json\n", b"\n", b'data: {"a": 2}\n', b"\n"]
    assert list(_iter_sse(iter(raw))) == [{"a": 1}, {"a": 2}]


def test_the_reader_stops_mid_stream_when_asked():
    """A console quitting must not wait for the next frame from a card that is
    doing nothing."""
    stop = threading.Event()
    stop.set()
    assert list(_iter_sse(iter([b'data: {"a": 1}\n']), stop)) == []


async def test_the_feed_publishes_only_what_changed():
    payloads = [{"used": 1}, {"used": 1}, {"used": 2}]
    feed = StateFeed(lambda: _immediately(payloads.pop(0)))
    queue = feed.subscribe()
    try:
        assert await feed.sample_once() == {"used": 1}
        assert await feed.sample_once() is None, "an unchanged card is not news"
        assert await feed.sample_once() == {"used": 2}
        assert queue.get_nowait() == {"used": 2}
    finally:
        feed.unsubscribe(queue)


async def test_a_slow_watcher_gets_the_newest_state_not_a_backlog():
    """One slot per subscriber: a console that stalled wants the card as it is
    now, not four frames of what it missed."""
    values = [{"n": 1}, {"n": 2}, {"n": 3}]
    feed = StateFeed(lambda: _immediately(values.pop(0)))
    queue = feed.subscribe()
    try:
        for _ in range(3):
            await feed.sample_once()
        assert queue.qsize() == 1
        assert queue.get_nowait() == {"n": 3}
    finally:
        feed.unsubscribe(queue)


async def test_the_sampler_runs_only_while_somebody_is_watching():
    """Nothing reads the card when no console is attached: an idle router has
    to stay exactly as quiet as it was before this endpoint existed."""
    feed = StateFeed(lambda: _immediately({"n": 1}), interval=0.01)
    assert feed._task is None
    first = feed.subscribe()
    second = feed.subscribe()
    assert feed.watchers == 2
    task = feed._task
    assert task is not None
    feed.unsubscribe(first)
    assert feed._task is task, "one console leaving does not stop the other's feed"
    feed.unsubscribe(second)
    assert feed._task is None
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()


async def test_a_failing_build_becomes_an_error_frame_not_a_dead_feed():
    """A card that cannot be read is something the console should say. A feed
    that dies of it takes every attached console down with it."""
    async def boom():
        raise RuntimeError("nvidia-smi went away")

    feed = StateFeed(boom)
    published = await feed.sample_once()
    assert published == {"error": "nvidia-smi went away"}


async def _immediately(value):
    return value


# ---- the endpoint, over a real socket --------------------------------------


@pytest.fixture
async def client(isolated_registry, monkeypatch, tmp_path):
    monkeypatch.setenv("VRAMUX_MODELS_CONFIG", str(tmp_path / "none.yml"))
    monkeypatch.setenv("VRAMUX_MODEL_DIR", str(tmp_path / "none"))
    arbiter = packing_arbiter({"a:9b": 6591}, free_mb=20000)
    await arbiter.acquire(spec("a:9b"))
    arbiter.release("a:9b")
    # The residency fakes do not read a card; the state endpoints do. One
    # written snapshot is all either of them needs.
    snap = snapshot(used=7000)
    arbiter.observer.snapshot = lambda: _immediately(snap)
    app = make_app(ModelRegistry(), arbiter, None)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


async def test_the_stream_opens_with_the_current_state_not_the_next_change(client):
    """A console attaching to an idle card must draw something immediately —
    otherwise "no changes for ten minutes" and "the router is wedged" look
    identical."""
    resp = await client.get("/gpu/events")
    assert resp.status == 200
    assert resp.headers["Content-Type"].startswith("text/event-stream")
    payload = await _first_event(resp)
    assert payload["residents"] == ["a:9b"]
    assert payload["resident_detail"][0]["port"] == 18080
    resp.close()


async def test_the_stream_and_the_poll_return_the_same_body(client):
    resp = await client.get("/gpu/events")
    streamed = await _first_event(resp)
    resp.close()
    polled = await (await client.get("/gpu/state")).json()
    # `last_use` is the only field that can differ, and only in its seconds.
    for body in (streamed, polled):
        for row in body["resident_detail"]:
            row.pop("last_use", None)
    assert streamed == polled


async def test_watchers_are_forgotten_when_their_console_goes_away(client):
    resp = await client.get("/gpu/events")
    await _first_event(resp)
    router = client.app[ROUTER]
    assert router.feed.watchers == 1
    resp.close()
    # The handler drops its subscription in a `finally`, so the feed must be
    # back to nobody watching — and stop sampling — once the socket is gone.
    for _ in range(100):
        await asyncio.sleep(0.02)
        if router.feed.watchers == 0:
            break
    assert router.feed.watchers == 0
    assert router.feed._task is None, "the last console leaving stops the sampler"


async def _first_event(resp) -> dict:
    async for raw in resp.content:
        line = raw.decode().strip()
        if line.startswith("data:"):
            return json.loads(line[5:])
    raise AssertionError("the stream ended without an event")


# ---- history, for the sparkline --------------------------------------------


async def test_history_returns_the_recorded_window_and_its_spacing(client, tmp_path):
    """Rows without `interval_s` are a shape at an unknown resolution, so the
    two travel together: twelve points is an hour or a minute depending on it.
    """
    now = datetime.now(timezone.utc)
    log = UsageLog(tmp_path / "usage.jsonl")
    log.path.write_text("".join(
        json.dumps({"t": (now - timedelta(minutes=m)).isoformat(),
                    "used_mb": 7000 + m, "foreign_mb": 400}) + "\n"
        for m in (300, 40, 10)
    ))
    client.app[ROUTER].arbiter.observer.history = log

    body = await (await client.get("/gpu/history?minutes=60")).json()
    assert [r["used_mb"] for r in body["rows"]] == [7040, 7010]
    assert body["minutes"] == 60
    assert "interval_s" in body, "the caller cannot label the axis without it"


async def test_history_reads_the_file_and_never_the_card(client, tmp_path):
    """The whole argument for a separate endpoint: a page redrawing its
    sparkline must not cost an `nvidia-smi` call."""
    def explode():
        raise AssertionError("the history endpoint probed the device")

    client.app[ROUTER].arbiter.observer.snapshot = explode
    assert (await client.get("/gpu/history")).status == 200


async def test_history_refuses_a_window_that_is_not_a_number(client):
    body = await (await client.get("/gpu/history?minutes=lastweek")).json()
    assert "must be numbers" in body["error"]


async def test_history_refuses_a_window_of_nothing(client):
    """`minutes=0` would return an empty chart that reads as "the card was
    idle", which is a different claim from "you asked for no time at all"."""
    assert (await client.get("/gpu/history?minutes=0")).status == 400
    assert (await client.get("/gpu/history?limit=-1")).status == 400


async def test_history_reports_the_samplers_interval_when_there_is_a_broker(client):
    """`interval_s` is the broker's, because the broker's timer is what writes
    the rows — nothing else in the process knows the cadence."""
    router = client.app[ROUTER]
    router.broker = SimpleNamespace(sample_interval=42.0)
    try:
        body = await (await client.get("/gpu/history")).json()
    finally:
        router.broker = None
    assert body["interval_s"] == 42.0


async def test_history_is_refused_when_nothing_is_observing(isolated_registry,
                                                            monkeypatch, tmp_path):
    monkeypatch.setenv("VRAMUX_MODELS_CONFIG", str(tmp_path / "none.yml"))
    monkeypatch.setenv("VRAMUX_MODEL_DIR", str(tmp_path / "none"))
    app = make_app(ModelRegistry(), ResidencyArbiter(observer=None), None)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        assert (await client.get("/gpu/history")).status == 503
    finally:
        await client.close()


async def test_events_are_refused_when_nothing_is_observing(isolated_registry,
                                                            monkeypatch, tmp_path):
    monkeypatch.setenv("VRAMUX_MODELS_CONFIG", str(tmp_path / "none.yml"))
    monkeypatch.setenv("VRAMUX_MODEL_DIR", str(tmp_path / "none"))
    app = make_app(ModelRegistry(), ResidencyArbiter(observer=None), None)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        assert (await client.get("/gpu/events")).status == 503
    finally:
        await client.close()


def test_the_feed_reports_a_router_that_is_not_there(monkeypatch):
    """`vramux top` against a stopped router says so on the screen rather than
    drawing an empty card, which reads as "nothing is loaded"."""
    feed = Feed("127.0.0.1", 1)  # nothing listens on port 1
    feed.poll_once()
    assert feed.state is None
    assert "cannot reach vramux" in (feed.error or "")


async def test_the_browser_console_is_served_and_needs_nothing_off_the_network(client):
    """One file, no assets: the box that wants this page is usually the box
    with no network, and a page that fetches a font from a CDN is a blank
    screen exactly then."""
    resp = await client.get("/gpu/console")
    assert resp.status == 200
    assert resp.headers["Content-Type"].startswith("text/html")
    page = await resp.text()
    assert "/gpu/events" in page, "it draws from the same stream the TUI does"
    for remote in ("http://", "https://", "//cdn", "<img", "@import"):
        assert remote not in page, f"the page must not reach for {remote}"


def test_a_lease_being_asked_to_yield_outranks_the_outstanding_mark():
    """Two different facts, and only one of them means somebody is waiting: an
    unallocated grant is normal, a yield request is contention happening now."""
    lease = {"lease": "abc", "owner": "batch", "granted_mb": 12000,
             "observed_mb": 12100, "outstanding_mb": 0, "priority": 1,
             "expires_at": "2026-08-07T12:01:00+00:00", "ttl": 60,
             "yield": {"wanted_mb": 6000, "by": "serving:big:27b", "priority": 7,
                       "requested_at": "2026-08-07T12:00:00+00:00",
                       "deadline": "2026-08-07T12:00:30+00:00"}}
    lines = render(state(leases=[lease]), width=100, now=NOON)
    assert styles(lines, "batch") == "hot"
    drawn = text(lines)
    assert "asked for 6 000 MiB by serving:big:27b" in drawn
    assert "30s left" in drawn, "the holder's deadline counts down"

"""The arbiter's side of the render scheduler: a hold that parks every other
acquire, and a scheduled resident that waives lend_after and is not restored.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from vramux import residency

from test_polite_exclusive import QueueArbiter, stack
from test_residency import spec


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(residency, "_BUSY_POLL", 0.01)


async def test_a_hold_parks_other_models_until_it_is_released():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0, busy_wait=0.05)
    img, llm = stack(), spec("llm:27b")
    sup.hold_render("stack:1")
    await sup.acquire(img, scheduled=True)

    load = asyncio.create_task(sup.acquire(llm))
    await asyncio.sleep(0.2)
    assert not load.done(), "parked, and not capped by busy_wait"
    assert sup.loading is None, "a park is not a load: leases are not refused for it"
    view = sup.parked_view
    assert view["tag"] == "llm:27b" and view["behind"] == "render jobs on stack:1"

    sup.release("stack:1")
    sup.release_render()
    await asyncio.wait_for(load, 2)
    assert sup.events == ["start:stack:1", "stop:stack:1", "start:llm:27b"]
    assert sup.parked_view is None


async def test_the_held_model_itself_is_not_parked():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0)
    img = stack()
    sup.hold_render("stack:1")
    await sup.acquire(img, scheduled=True)
    await asyncio.wait_for(sup.acquire(img), 0.5)   # a human at the UI, via /m/
    assert sup._residents["stack:1"].scheduled, "a hand request mid-render keeps the waiver"


async def test_a_scheduled_resident_waives_lend_after_and_is_not_restored():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0, busy_wait=5.0)
    img, llm = stack(lend_after=60, restore_after=60), spec("llm:27b")
    sup.hold_render("stack:1")
    await sup.acquire(img, scheduled=True)
    sup.release("stack:1")
    sup.release_render()
    await asyncio.wait_for(sup.acquire(llm), 0.5)
    assert sup.current_tag == "llm:27b"
    assert not sup._displaced, "a render nobody is watching is not given the card back"


async def test_a_plain_load_after_a_render_gets_its_protections_back():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0, busy_wait=5.0)
    img, llm = stack(lend_after=60), spec("llm:27b")
    sup.hold_render("stack:1")
    await sup.acquire(img, scheduled=True)
    sup.release("stack:1")
    sup.release_render()
    await sup.acquire(img)                       # somebody loads it by hand
    sup.release("stack:1")
    sup._residents["stack:1"].last_busy = time.monotonic()
    load = asyncio.create_task(sup.acquire(llm))
    await asyncio.sleep(0.1)
    assert not load.done(), "lend_after is back for a load nobody scheduled"
    load.cancel()


async def test_what_a_render_displaces_is_not_remembered():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0)
    chat, img = spec("chat:9b", restore_after=30), stack()
    await sup.acquire(chat)
    sup.release("chat:9b")
    timings = {}
    sup.hold_render("stack:1")
    await sup.acquire(img, scheduled=True, timings=timings)
    assert not sup._displaced
    assert set(timings) == {"evict_s", "load_s"}


async def test_after_the_scheduler_lets_go_the_stack_is_an_ordinary_resident():
    """The waiver covers the flip-back, not the stack's whole life: a human's
    next job on it is protected by lend_after again."""
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0, busy_wait=5.0)
    img, llm = stack(lend_after=60), spec("llm:27b")
    sup.hold_render("stack:1")
    await sup.acquire(img, scheduled=True)
    sup.jobs["stack:1"] = 1                       # our last workflow, still on it
    assert await sup._busy_jobs(sup._residents["stack:1"]) == 1
    sup.jobs["stack:1"] = 0
    sup.release("stack:1", scheduled=True)
    sup.release_render()
    resident = sup._residents["stack:1"]
    assert not resident.scheduled
    assert sup._settling(resident) == 0, "the flip-back itself is not held"

    sup.jobs["stack:1"] = 1                       # somebody at the UI queues
    await sup._busy_jobs(resident)
    sup.jobs["stack:1"] = 0
    load = asyncio.create_task(sup.acquire(llm))
    await asyncio.sleep(0.1)
    assert not load.done(), "lend_after protects the human's next use"
    load.cancel()


async def test_a_parked_request_whose_client_left_is_dropped(monkeypatch):
    monkeypatch.setattr(residency, "_PARK_POLL", 0.01)
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0)
    sup.hold_render("stack:1")
    await sup.acquire(stack(), scheduled=True)
    gone = {"yes": False}
    load = asyncio.create_task(sup.acquire(spec("llm:27b"), gone=lambda: gone["yes"]))
    await asyncio.sleep(0.05)
    assert len(sup.parked) == 1
    gone["yes"] = True
    with pytest.raises(residency.ClientGone):
        await asyncio.wait_for(load, 1)
    assert sup.parked == [], "it no longer counts toward llm_latency_bound"
    assert sup.arriving_except("stack:1") == 0

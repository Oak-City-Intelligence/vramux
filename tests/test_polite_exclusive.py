"""Two exclusive residents getting out of each other's way.

A workflow server whose jobs bypass the router is displaced only between jobs,
and comes back once the borrower is done with the card.
"""

from __future__ import annotations

import asyncio

import pytest

from vramux import residency
from vramux.registry import KIND_DOCKER
from vramux.residency import InsufficientVRAM

from test_residency import FakeArbiter, FakeBackend, spec


class QueueBackend(FakeBackend):
    """A fake whose unseen job queue the test controls through the arbiter."""

    async def busy(self):
        return self.arbiter.jobs.get(self.spec.tag, 0)


class QueueArbiter(FakeArbiter):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.jobs = {}

    def _make_backend(self, spec, port=None):
        b = QueueBackend(spec, self, self.start_delay, self.fail_next_start)
        self.backends.append(b)
        return b


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(residency, "_BUSY_POLL", 0.01)


def stack(**kw):
    return spec(
        "stack:1", kind=KIND_DOCKER, exclusive=True, idle_timeout=0,
        busy_path="/queue", **kw,
    )


async def test_a_load_waits_for_the_running_job_instead_of_stopping_it():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0, busy_wait=5.0)
    img, llm = stack(), spec("llm:27b")
    await sup.acquire(img)
    sup.release("stack:1")
    sup.jobs["stack:1"] = 1

    load = asyncio.create_task(sup.acquire(llm))
    await asyncio.sleep(0.1)
    assert not load.done(), "the load must wait while the stack has a job"
    assert "stop:stack:1" not in sup.events
    assert sup.loading and "stack:1" in str(sup.loading)

    sup.jobs["stack:1"] = 0
    await asyncio.wait_for(load, 2)
    assert sup.events == ["start:stack:1", "stop:stack:1", "start:llm:27b"]


async def test_a_queue_that_never_empties_is_refused_not_killed():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0, busy_wait=0.1)
    img, llm = stack(), spec("llm:27b")
    await sup.acquire(img)
    sup.release("stack:1")
    sup.jobs["stack:1"] = 2
    with pytest.raises(InsufficientVRAM, match="not stopping it"):
        await sup.acquire(llm)
    assert sup.current_tag == "stack:1"
    assert "stop:stack:1" not in sup.events


async def test_the_borrower_is_held_to_the_lenders_wait_and_the_card_goes_back():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0)
    img, llm = stack(restore_after=60), spec("llm:27b")
    await sup.acquire(img)
    sup.release("stack:1")
    await sup.acquire(llm)
    sup.release("llm:27b")
    assert sup._effective_idle_timeout(sup._residents["llm:27b"]) == 60

    assert await sup.evict("llm:27b")
    assert sup.current_tag == "stack:1", "the card goes back once the borrower leaves"
    assert sup.events[-2:] == ["stop:llm:27b", "start:stack:1"]
    assert not sup._displaced


async def test_an_idle_borrower_gives_the_card_back(monkeypatch):
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0)
    img, llm = stack(restore_after=60), spec("llm:27b")
    await sup.acquire(img)
    sup.release("stack:1")
    await sup.acquire(llm)
    sup.release("llm:27b")
    sup._residents["llm:27b"].last_use -= 61

    real_sleep = asyncio.sleep
    monkeypatch.setattr(residency.asyncio, "sleep", lambda s: real_sleep(0.01))
    sup._idle_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sup._idle_task
    sup._ensure_idle_watcher()
    for _ in range(100):
        if sup.current_tag == "stack:1":
            break
        await real_sleep(0.01)
    assert sup.current_tag == "stack:1"
    assert "stop:llm:27b" in sup.events


async def test_evicting_the_lender_by_hand_forgets_it():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0)
    img, llm = stack(restore_after=60), spec("llm:27b")
    await sup.acquire(img)
    sup.release("stack:1")
    await sup.acquire(llm)
    sup.release("llm:27b")
    await sup.evict("stack:1")
    assert not sup._displaced
    assert await sup.evict("llm:27b")
    assert sup.current_tag is None


async def test_without_restore_after_a_displaced_stack_stays_down():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0)
    img, llm = stack(), spec("llm:27b")
    await sup.acquire(img)
    sup.release("stack:1")
    await sup.acquire(llm)
    sup.release("llm:27b")
    await sup.evict("llm:27b")
    assert sup.current_tag is None


async def test_a_lease_cannot_evict_a_busy_resident():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0)
    img = stack()
    await sup.acquire(img)
    sup.release("stack:1")
    sup.jobs["stack:1"] = 1
    assert await sup.make_room_for_lease(20000, by="batch", priority=9) == 0
    assert sup.current_tag == "stack:1"


async def test_a_parked_request_cannot_take_the_card_the_instant_the_stack_comes_up():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0, busy_wait=5.0)
    img, llm = stack(lend_after=0.3), spec("llm:27b")
    await sup.acquire(img)
    sup.release("stack:1")

    load = asyncio.create_task(sup.acquire(llm))
    await asyncio.sleep(0.1)
    assert not load.done(), "a freshly loaded stack is not lent before its client can queue"
    sup.jobs["stack:1"] = 1          # the client queues its job inside the window
    await asyncio.sleep(0.5)
    assert not load.done(), "and then the job holds it"
    assert "stop:stack:1" not in sup.events

    sup.jobs["stack:1"] = 0
    await asyncio.sleep(0.1)
    assert not load.done(), "the window restarts when the queue empties"
    await asyncio.wait_for(load, 2)
    assert sup.events == ["start:stack:1", "stop:stack:1", "start:llm:27b"]


async def test_without_lend_after_an_idle_stack_is_lent_at_once():
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0, busy_wait=5.0)
    img, llm = stack(), spec("llm:27b")
    await sup.acquire(img)
    sup.release("stack:1")
    await asyncio.wait_for(sup.acquire(llm), 0.5)
    assert sup.current_tag == "llm:27b"


async def test_the_wait_is_labelled_by_what_it_is_actually_waiting_on():
    """Every request queued on the lock reads the holder's label. It used to
    say "running job(s)" through a lend_after window while the holder's own
    line said "0 job(s)", so the same wait was logged two contradictory ways."""
    sup = QueueArbiter(drain_timeout=0.2, startup_timeout=5.0, busy_wait=5.0)
    img, llm = stack(lend_after=0.3), spec("llm:27b")
    await sup.acquire(img)
    sup.release("stack:1")

    load = asyncio.create_task(sup.acquire(llm))
    await asyncio.sleep(0.05)
    assert "lend_after window" in sup.loading["behind"]
    sup.jobs["stack:1"] = 2
    await asyncio.sleep(0.05)
    assert sup.loading["behind"] == "stack:1's 2 running job(s)"
    sup.jobs["stack:1"] = 0
    await asyncio.wait_for(load, 2)

"""Serving queues behind a lease it does not outrank, instead of erroring.

The property under test throughout is the narrowness of the branch: a load
parks only when the blockers are leases at or above its priority AND
releasing them would actually make it fit. Every other shortfall — a peer
that was asked to yield and declined, foreign memory, a cost that never
fits — raises `InsufficientVRAM` exactly as it did before the queue existed.
And the park is bounded twice over: by `queue_wait`, and by the blocking
lease's own TTL, because TTL being mandatory is what makes waiting a promise
rather than a hope.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest

from vramux.lease import PRIORITY_BATCH
from vramux.residency import InsufficientVRAM

from test_lease import FakeObserver, broker
from test_residency import packing_arbiter, spec

PRIORITY_URGENT = 8  # outranks default serving (7)


def queued_arbiter(costs, free_mb, b, **kw):
    """An arbiter wired to a broker both ways: yield out, outrankers in."""
    sup = packing_arbiter(costs, free_mb=free_mb, **kw)
    sup.use_yield(b.request_yield)
    sup.use_outrankers(b.outrankers)
    return sup


# ---- the happy path: parked, then served -----------------------------------


async def test_an_outranked_load_parks_and_proceeds_when_the_lease_releases():
    """One connection, no 503: the caller sees a slow first token."""
    b = broker(FakeObserver(), sweep_interval=60)
    lease = await b.acquire(mb=18000, owner="image-stack", ttl=60,
                            priority=PRIORITY_URGENT)
    sup = queued_arbiter({"big:27b": 19000}, 10000, b,
                         yield_wait=0.1, queue_wait=10.0)

    async def release_soon():
        await asyncio.sleep(0.3)
        await b.release(lease.id)
        sup.free_mb = 20000  # the card the fake budget reports

    asyncio.get_running_loop().create_task(release_soon())
    await asyncio.wait_for(sup.acquire(spec("big:27b")), timeout=5)
    sup.release("big:27b")

    assert [r.tag for r in sup.residents] == ["big:27b"]


async def test_the_park_ends_when_the_blocking_lease_expires():
    """A holder killed with SIGKILL runs no cleanup; expiry is what keeps the
    promise bounded, and the park rides the same sweep expiry rides."""
    b = broker(FakeObserver(), sweep_interval=60)
    await b.acquire(mb=18000, owner="image-stack", ttl=0.5,
                    priority=PRIORITY_URGENT)
    sup = queued_arbiter({"big:27b": 19000}, 10000, b,
                         yield_wait=0.1, queue_wait=10.0)

    async def expire_soon():
        await asyncio.sleep(0.7)
        await b.sweep()          # expiry is enforced by the sweep
        sup.free_mb = 20000

    asyncio.get_running_loop().create_task(expire_soon())
    await asyncio.wait_for(sup.acquire(spec("big:27b")), timeout=5)
    sup.release("big:27b")

    assert [r.tag for r in sup.residents] == ["big:27b"]


# ---- the refusals that must stay refusals ----------------------------------


async def test_the_cap_is_a_cap_and_the_refusal_names_the_blocker():
    b = broker(FakeObserver(), sweep_interval=60)
    await b.acquire(mb=18000, owner="image-stack", ttl=60,
                    priority=PRIORITY_URGENT)
    sup = queued_arbiter({"big:27b": 19000}, 10000, b,
                         yield_wait=0.1, queue_wait=0.4)

    started = time.monotonic()
    with pytest.raises(InsufficientVRAM) as exc:
        await sup.acquire(spec("big:27b"))

    assert time.monotonic() - started < 3.0
    assert "image-stack" in str(exc.value)
    assert "outranks serving" in str(exc.value)


async def test_a_blocker_serving_outranks_gets_the_old_behavior():
    """Below serving's priority there is nothing to park behind: the holder
    was asked to yield, declined, and the refusal is exactly the old one."""
    b = broker(FakeObserver(), sweep_interval=60)
    holder = await b.acquire(mb=18000, owner="batch", ttl=60,
                             priority=PRIORITY_BATCH)
    sup = queued_arbiter({"big:27b": 19000}, 10000, b,
                         yield_wait=0.2, queue_wait=10.0)

    started = time.monotonic()
    with pytest.raises(InsufficientVRAM):
        await sup.acquire(spec("big:27b"))

    assert time.monotonic() - started < 3.0, "no ten-second park happened"
    assert holder.yield_request is not None, "the holder was asked first"


async def test_no_park_when_releasing_the_outrankers_would_not_fit_the_load():
    """Foreign grew, or the memory was never the lease's: waiting cannot
    help, so this fails now rather than parking on false hope."""
    b = broker(FakeObserver(), sweep_interval=60)
    await b.acquire(mb=1000, owner="probe", ttl=60, priority=PRIORITY_URGENT)
    sup = queued_arbiter({"big:27b": 19000}, 10000, b,
                         yield_wait=0.1, queue_wait=10.0)

    started = time.monotonic()
    with pytest.raises(InsufficientVRAM):
        await sup.acquire(spec("big:27b"))
    assert time.monotonic() - started < 2.0


async def test_queue_wait_zero_restores_the_old_behavior():
    b = broker(FakeObserver(), sweep_interval=60)
    await b.acquire(mb=18000, owner="image-stack", ttl=60,
                    priority=PRIORITY_URGENT)
    sup = queued_arbiter({"big:27b": 19000}, 10000, b,
                         yield_wait=0.1, queue_wait=0.0)

    started = time.monotonic()
    with pytest.raises(InsufficientVRAM):
        await sup.acquire(spec("big:27b"))
    assert time.monotonic() - started < 2.0


async def test_an_unwired_arbiter_is_the_arbiter_there_was_before():
    """`use_outrankers` never called — every test written before the queue —
    and the refusal path is untouched."""
    b = broker(FakeObserver(), sweep_interval=60)
    await b.acquire(mb=18000, owner="image-stack", ttl=60,
                    priority=PRIORITY_URGENT)
    sup = packing_arbiter({"big:27b": 19000}, free_mb=10000,
                          yield_wait=0.1, queue_wait=10.0)
    sup.use_yield(b.request_yield)

    started = time.monotonic()
    with pytest.raises(InsufficientVRAM):
        await sup.acquire(spec("big:27b"))
    assert time.monotonic() - started < 2.0


# ---- honesty while parked --------------------------------------------------


async def test_a_parked_load_is_visible_and_says_what_it_is_behind():
    """A silent socket is indistinguishable from a wedge. While parked, the
    load registers as in flight with the lease it is behind, so a second
    caller and the console both get the truthful message."""
    b = broker(FakeObserver(), sweep_interval=60)
    lease = await b.acquire(mb=18000, owner="image-stack", ttl=60,
                            priority=PRIORITY_URGENT)
    sup = queued_arbiter({"big:27b": 19000}, 10000, b,
                         yield_wait=0.1, queue_wait=10.0)

    task = asyncio.create_task(sup.acquire(spec("big:27b")))
    await asyncio.sleep(0.3)
    try:
        load = sup.loading
        assert load is not None
        assert lease.id in load["behind"]
        assert "queued behind" in sup._wait_message(0)
    finally:
        await b.release(lease.id)
        sup.free_mb = 20000
        await asyncio.wait_for(task, timeout=5)
        sup.release("big:27b")


async def test_a_client_that_disconnects_abandons_the_park():
    """aiohttp cancels the handler when the socket closes; the park must
    surrender the lock and leave no phantom load behind."""
    b = broker(FakeObserver(), sweep_interval=60)
    await b.acquire(mb=18000, owner="image-stack", ttl=60,
                    priority=PRIORITY_URGENT)
    sup = queued_arbiter({"big:27b": 19000}, 10000, b,
                         yield_wait=0.1, queue_wait=10.0)

    task = asyncio.create_task(sup.acquire(spec("big:27b")))
    await asyncio.sleep(0.3)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert sup.loading is None, "no phantom load left behind"
    assert not sup._lock.locked(), "the lock came back"


async def test_renewal_extends_the_horizon_but_never_past_the_cap():
    """A holder that heartbeats forever must not turn the park into a hang:
    `queue_wait` is a hard ceiling, renewals or not."""
    b = broker(FakeObserver(), sweep_interval=60)
    lease = await b.acquire(mb=18000, owner="image-stack", ttl=0.6,
                            priority=PRIORITY_URGENT)
    sup = queued_arbiter({"big:27b": 19000}, 10000, b,
                         yield_wait=0.1, queue_wait=1.2)

    async def keep_renewing():
        while True:
            await asyncio.sleep(0.25)
            with contextlib.suppress(Exception):
                await b.renew(lease.id)

    renewer = asyncio.create_task(keep_renewing())
    started = time.monotonic()
    try:
        with pytest.raises(InsufficientVRAM):
            await sup.acquire(spec("big:27b"))
    finally:
        renewer.cancel()

    elapsed = time.monotonic() - started
    assert 0.9 < elapsed < 4.0, f"capped at queue_wait, took {elapsed:.1f}s"

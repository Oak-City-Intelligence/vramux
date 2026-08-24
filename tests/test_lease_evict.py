"""Tier 1, read from the lease's side: a lease may evict managed residents.

The property under test throughout is the priority gate. A default lease (5)
against default serving (7) evicts nothing, so every client written before
this existed behaves byte-for-byte as it did — the feature is asked for
explicitly, per request, by taking a priority above the resident's. And what
eviction means is exactly what it means when serving does it: drain first,
then stop, and the victim pays a reload later rather than losing anything.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from vramux.lease import (
    PRIORITY_BATCH,
    PRIORITY_DEFAULT,
    NoRoom,
)
from vramux.registry import ModelSpec

from test_lease import TOTAL, FakeObserver, broker, snapshot
from test_residency import packing_arbiter, spec

# One above default serving (7): the generation-lease case the feature is for.
PRIORITY_URGENT = 8

FULL_CARD = TOTAL - 3000  # a big model resident: little free, much evictable


def short_broker(observer=None, **kw):
    """A broker whose card is nearly full, so any real lease is short."""
    observer = observer or FakeObserver(snapshot(used=FULL_CARD))
    return broker(observer, sweep_interval=60, **kw), observer


async def resident_arbiter(costs, tag="big:27b", **kw):
    """An arbiter with `tag` resident and idle, measured per `costs`."""
    sup = packing_arbiter(costs, free_mb=20000, **kw)
    await sup.acquire(spec(tag))
    sup.release(tag)
    return sup


def freeing(sup, observer):
    """Residency's make-room, with the card played by the test: when an
    eviction really happens the device would read freer, so the fake does."""
    async def make_room(mb, by, priority):
        evicted = await sup.make_room_for_lease(mb, by, priority)
        if evicted:
            observer.snap = snapshot(used=1000)
        return evicted
    return make_room


# ---- the priority gate -----------------------------------------------------


async def test_a_default_lease_evicts_nothing():
    """The compatibility story in one test: 5 does not outrank 7, so a lease
    that predates this feature waits and fails exactly as it always did."""
    b, observer = short_broker()
    sup = await resident_arbiter({"big:27b": 19000})
    b.use_make_room(freeing(sup, observer))

    with pytest.raises(NoRoom):
        await b.acquire(mb=8000, owner="batch", ttl=60,
                        priority=PRIORITY_DEFAULT, wait=0.2)

    assert [r.tag for r in sup.residents] == ["big:27b"], "still resident"
    assert "stop:big:27b" not in sup.events


async def test_an_outranking_lease_evicts_the_resident_and_is_granted():
    b, observer = short_broker()
    sup = await resident_arbiter({"big:27b": 19000})
    b.use_make_room(freeing(sup, observer))

    lease = await b.acquire(mb=8000, owner="image-stack", ttl=60,
                            priority=PRIORITY_URGENT, wait=2.0)

    assert lease.mb == 8000
    assert sup.events == ["start:big:27b", "stop:big:27b"]
    assert sup.residents == []


async def test_a_pinned_resident_outranks_the_lease():
    """`priority:` in the model config is the pin: a resident at or above the
    asker is never touched, which is how an operator protects the brain."""
    b, observer = short_broker()
    sup = packing_arbiter({"pinned:27b": 19000}, free_mb=20000)
    await sup.acquire(ModelSpec(tag="pinned:27b", priority=9))
    sup.release("pinned:27b")
    b.use_make_room(freeing(sup, observer))

    with pytest.raises(NoRoom):
        await b.acquire(mb=8000, owner="image-stack", ttl=60,
                        priority=PRIORITY_URGENT, wait=0.2)

    assert [r.tag for r in sup.residents] == ["pinned:27b"]


async def test_wait_zero_never_evicts():
    """Eviction only arrives by waiting — drain, stop, the card catching up —
    so a caller that declared it will not wait cannot be given it. Same
    contract as yield, which also only runs under a deadline."""
    b, observer = short_broker()
    sup = await resident_arbiter({"big:27b": 19000})
    b.use_make_room(freeing(sup, observer))

    with pytest.raises(NoRoom):
        await b.acquire(mb=8000, owner="image-stack", ttl=60,
                        priority=PRIORITY_URGENT, wait=0.0)

    assert "stop:big:27b" not in sup.events


# ---- who goes first --------------------------------------------------------


async def test_the_cheapest_eligible_resident_goes_first():
    """Same reasoning as yield's cheapest-first: a 1.5 GB shortfall must not
    cost a 19 GB reload when a 2 GB embedder covers it."""
    sup = packing_arbiter({"small:1b": 2000, "big:27b": 19000}, free_mb=24000)
    await sup.acquire(spec("small:1b"))
    sup.release("small:1b")
    await sup.acquire(spec("big:27b"))
    sup.release("big:27b")

    evicted = await sup.make_room_for_lease(1500, "lease:probe", PRIORITY_URGENT)

    assert evicted == 1
    assert "stop:small:1b" in sup.events
    assert "stop:big:27b" not in sup.events


async def test_eviction_stops_once_the_shortfall_is_covered():
    sup = packing_arbiter({"a:9b": 6000, "b:9b": 6000}, free_mb=24000)
    await sup.acquire(spec("a:9b"))
    sup.release("a:9b")
    await sup.acquire(spec("b:9b"))
    sup.release("b:9b")

    evicted = await sup.make_room_for_lease(5000, "lease:x", PRIORITY_URGENT)

    assert evicted == 1, "the first eviction covered it"
    assert len(sup.residents) == 1


async def test_an_unmeasured_resident_is_evicted_but_never_counted():
    """What an unmeasured model frees is the card's to say. Evict it, stop
    counting, and let the caller's re-read of the budget decide."""
    sup = packing_arbiter({}, free_mb=4000)
    await sup.acquire(spec("mystery:12b"))
    sup.release("mystery:12b")

    evicted = await sup.make_room_for_lease(18000, "lease:x", PRIORITY_URGENT)

    assert evicted == 1
    assert sup.residents == []


# ---- eviction is a drain, not a cut ----------------------------------------


async def test_eviction_waits_for_the_victims_stream_to_drain():
    sup = packing_arbiter({"big:27b": 19000}, free_mb=20000, drain_timeout=5.0)
    await sup.acquire(spec("big:27b"))  # in flight: not released

    task = asyncio.create_task(
        sup.make_room_for_lease(8000, "lease:image-stack", PRIORITY_URGENT))
    await asyncio.sleep(0.05)
    assert "stop:big:27b" not in sup.events, "a stream in progress is never cut"

    sup.release("big:27b")
    assert await asyncio.wait_for(task, timeout=2) == 1
    assert "stop:big:27b" in sup.events


async def test_a_drain_that_never_ends_is_forced_like_any_swap():
    sup = packing_arbiter({"big:27b": 19000}, free_mb=20000, drain_timeout=0.1)
    await sup.acquire(spec("big:27b"))  # held forever

    evicted = await asyncio.wait_for(
        sup.make_room_for_lease(8000, "lease:x", PRIORITY_URGENT), timeout=2)

    assert evicted == 1
    assert "stop:big:27b" in sup.events


# ---- the broker's side -----------------------------------------------------


async def test_an_unwired_broker_is_the_broker_there_was_before():
    """`use_make_room` never called — a bare embedding, every existing test —
    and the acquire path is untouched."""
    b, _ = short_broker()
    with pytest.raises(NoRoom):
        await b.acquire(mb=8000, owner="image-stack", ttl=60,
                        priority=PRIORITY_URGENT, wait=0.2)


async def test_a_broken_residency_degrades_to_waiting(caplog):
    """A residency layer that raises must cost the lease nothing but the
    eviction it was not going to get anyway."""
    b, _ = short_broker()

    async def broken(mb, by, priority):
        raise RuntimeError("residency fell over")

    b.use_make_room(broken)
    with caplog.at_level(logging.WARNING, logger="vramux.lease"):
        with pytest.raises(NoRoom):
            await b.acquire(mb=8000, owner="image-stack", ttl=60,
                            priority=PRIORITY_URGENT, wait=0.2)
    assert any("could not evict" in r.message for r in caplog.records)


async def test_eviction_that_covers_the_shortfall_asks_nobody_to_yield():
    """Evict-before-yield, and the reason for the order: the resident pays a
    reload later, while a yielding holder stops running work now. When the
    cheap disruption covers it, the expensive ask never happens."""
    b, observer = short_broker()
    holder = await b.acquire(mb=1000, owner="batch", ttl=60,
                             priority=PRIORITY_BATCH)
    sup = await resident_arbiter({"big:27b": 19000})
    b.use_make_room(freeing(sup, observer))

    await b.acquire(mb=8000, owner="image-stack", ttl=60,
                    priority=PRIORITY_URGENT, wait=2.0)

    assert holder.yield_request is None, "the eviction covered it"
    assert "stop:big:27b" in sup.events

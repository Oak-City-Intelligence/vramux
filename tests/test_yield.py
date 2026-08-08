"""Tier 3: asking a leaseholder for its memory back.

The property under test throughout is that **asking is not taking**. Every
test here that looks like it is about cooperation is really about the failure
mode: a holder that ignores the request keeps every byte, and the requester
ends up exactly where it would have been if yield had never been built.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from vramux.lease import (
    PRIORITY_BATCH,
    PRIORITY_DEFAULT,
    PRIORITY_INTERACTIVE,
    NoRoom,
)
from vramux.registry import ModelSpec

from test_lease import RESERVE, TOTAL, FakeObserver, broker, snapshot
from test_residency import packing_arbiter, spec


def held(b, mb: int, owner: str, priority: int = PRIORITY_DEFAULT):
    return b.acquire(mb=mb, owner=owner, ttl=60, priority=priority)


# ---- who gets asked --------------------------------------------------------


async def test_only_holders_below_the_asker_are_asked():
    """Priority is higher-wins, and equal never yields: with everything at the
    default, "at or below" would have every ordinary holder asking every other
    ordinary holder to get off the card."""
    b = broker(sweep_interval=60)
    low = await held(b, 2000, "batch", PRIORITY_BATCH)
    same = await held(b, 2000, "peer", PRIORITY_DEFAULT)
    high = await held(b, 2000, "interactive", PRIORITY_INTERACTIVE)

    asked = await b.request_yield(4000, by="serving:big", priority=PRIORITY_DEFAULT)

    assert [l.id for l in asked] == [low.id]
    assert same.yield_request is None and high.yield_request is None


async def test_the_cheapest_holders_that_cover_the_shortfall_are_asked():
    """A 27B arriving should not stop a whole batch pipeline when a 2 GB probe
    covers the gap. Asking is not free to whoever is on the other end."""
    b = broker(sweep_interval=60)
    small = await held(b, 1000, "probe", PRIORITY_BATCH)
    medium = await held(b, 3000, "helper", PRIORITY_BATCH)
    big = await held(b, 12000, "pipeline", PRIORITY_BATCH)

    asked = await b.request_yield(3500, by="serving:big", priority=PRIORITY_DEFAULT)

    assert sorted(l.owner for l in asked) == ["helper", "probe"]
    assert big.yield_request is None, "the expensive holder was not needed"


async def test_a_holder_is_asked_once_not_once_per_poll():
    b = broker(sweep_interval=60)
    lease = await held(b, 2000, "batch", PRIORITY_BATCH)
    first = await b.request_yield(2000, by="a", priority=PRIORITY_DEFAULT)
    again = await b.request_yield(2000, by="b", priority=PRIORITY_DEFAULT)

    assert [l.id for l in first] == [lease.id]
    assert again == [], "an outstanding request is not replaced by a second one"
    assert lease.yield_request.by == "a"


# ---- asking is not taking --------------------------------------------------


async def test_a_yield_request_takes_no_memory_and_shortens_no_lease():
    """The whole safety argument in one test: a holder that has never heard of
    yield behaves identically with the feature turned on."""
    observer = FakeObserver()
    b = broker(observer, sweep_interval=60)
    lease = await held(b, 8000, "batch", PRIORITY_BATCH)
    before = (await b.budget()).free_mb
    expires_before = lease.expires_at

    await b.request_yield(8000, by="serving:big", priority=PRIORITY_INTERACTIVE)

    assert lease.id in {l.id for l in b.leases}, "still held"
    assert lease.mb == 8000, "still the whole grant"
    assert lease.expires_at == expires_before, "not expired early"
    assert (await b.budget()).free_mb == before, "no memory came back"


async def test_releasing_after_being_asked_is_what_actually_frees_it():
    b = broker(sweep_interval=60)
    lease = await held(b, 8000, "batch", PRIORITY_BATCH)
    short = (await b.budget()).free_mb
    await b.request_yield(8000, by="serving:big", priority=PRIORITY_INTERACTIVE)
    await b.release(lease.id)
    assert (await b.budget()).free_mb == short + 8000


async def test_a_holder_that_ignores_the_deadline_is_logged_exactly_once(caplog):
    """Same rule as expiry: quiet would hide the one number that says whether
    cooperative eviction is cooperating on this machine."""
    b = broker(sweep_interval=60)
    await held(b, 2000, "stubborn", PRIORITY_BATCH)
    await b.request_yield(2000, by="serving:big", priority=PRIORITY_DEFAULT,
                          deadline=0.2)
    with caplog.at_level(logging.WARNING, logger="vramux.lease"):
        await asyncio.sleep(0.25)
        await b.sweep()
        await b.sweep()
    ignored = [r for r in caplog.records if "asked to yield" in r.message]
    assert len(ignored) == 1, "said once, not every sweep for the rest of the TTL"


# ---- the holder's side of the wire -----------------------------------------


async def test_the_request_reaches_the_holder_on_its_next_heartbeat():
    """The heartbeat is the transport. No callback URL, no second connection,
    nothing for a holder behind anything to open."""
    b = broker(sweep_interval=60)
    lease = await held(b, 2000, "batch", PRIORITY_BATCH)
    assert (await b.view(lease))["yield"] is None

    await b.request_yield(2000, by="serving:big", priority=PRIORITY_DEFAULT)

    renewed = await b.renew(lease.id)
    payload = await b.view(renewed)
    assert payload["yield"]["by"] == "serving:big"
    assert payload["yield"]["wanted_mb"] == 2000
    assert payload["yield"]["deadline"]


async def test_renewing_does_not_clear_the_request():
    """Renewal says "I am still here", not "I dealt with it". Clearing on renew
    would mean a holder that heartbeats through the deadline looks compliant."""
    b = broker(sweep_interval=60)
    lease = await held(b, 2000, "batch", PRIORITY_BATCH)
    await b.request_yield(2000, by="serving:big", priority=PRIORITY_DEFAULT)
    await b.renew(lease.id)
    assert lease.yield_request is not None


# ---- a waiting acquire asks ------------------------------------------------


async def test_a_waiting_request_asks_the_holder_it_is_waiting_on():
    """`wait` already means "I am willing to wait". Asking during that wait is
    strictly better than waiting silently."""
    b = broker(FakeObserver(), sweep_interval=60)
    batch = await held(b, TOTAL - RESERVE - 2000, "batch", PRIORITY_BATCH)

    with pytest.raises(NoRoom):
        await b.acquire(mb=8000, owner="interactive", ttl=60,
                        priority=PRIORITY_INTERACTIVE, wait=0.3)

    assert batch.yield_request is not None
    assert batch.yield_request.by == "lease:interactive"
    # Asked for the shortfall, not for the whole request: the holder should
    # give up as little as gets the waiter moving.
    assert batch.yield_request.wanted_mb < 8000


async def test_a_request_that_fits_asks_nobody():
    b = broker(FakeObserver(), sweep_interval=60)
    batch = await held(b, 2000, "batch", PRIORITY_BATCH)
    await b.acquire(mb=2000, owner="interactive", ttl=60,
                    priority=PRIORITY_INTERACTIVE, wait=1.0)
    assert batch.yield_request is None


async def test_a_load_in_flight_is_not_a_shortfall_a_holder_can_fix():
    """The memory is about to be taken by something that already won admission.
    Asking a leaseholder to stop work for that would be asking for nothing."""
    loading = {"tag": None}
    b = broker(FakeObserver(), sweep_interval=60,
               loading=lambda: loading["tag"])
    batch = await held(b, 2000, "batch", PRIORITY_BATCH)
    loading["tag"] = {"tag": "big:27b"}
    with pytest.raises(NoRoom):
        await b.acquire(mb=2000, owner="interactive", ttl=60,
                        priority=PRIORITY_INTERACTIVE, wait=0.2)
    assert batch.yield_request is None


# ---- admission asks, and loads anyway --------------------------------------


def arbiter_with_broker(costs, free_mb, b, **kw):
    sup = packing_arbiter(costs, free_mb=free_mb, **kw)
    sup.use_yield(b.request_yield)
    return sup


async def test_admission_asks_when_it_is_short_and_loads_anyway():
    """The load is not blocked on an answer. Waiting forever for a holder that
    may never reply turns a cooperative gesture into a hang."""
    b = broker(FakeObserver(), sweep_interval=60)
    batch = await held(b, 8000, "batch", PRIORITY_BATCH)
    sup = arbiter_with_broker({"big:27b": 19000}, 10000, b, yield_wait=0.3)

    await sup.acquire(spec("big:27b"))
    sup.release("big:27b")

    assert batch.yield_request is not None, "the holder was asked"
    assert batch.yield_request.by == "serving:big:27b"
    assert [r.tag for r in sup.residents] == ["big:27b"], "and the load went ahead"


async def test_admission_stops_waiting_the_moment_the_memory_comes_back():
    b = broker(FakeObserver(), sweep_interval=60)
    lease = await held(b, 8000, "batch", PRIORITY_BATCH)
    sup = arbiter_with_broker({"big:27b": 19000}, 10000, b, yield_wait=30.0)

    async def release_soon():
        await asyncio.sleep(0.2)
        await b.release(lease.id)
        sup.free_mb = 20000  # the card the fake budget reports

    asyncio.get_running_loop().create_task(release_soon())
    await asyncio.wait_for(sup.acquire(spec("big:27b")), timeout=5)
    sup.release("big:27b")
    assert [r.tag for r in sup.residents] == ["big:27b"]


async def test_a_model_that_fits_asks_nobody():
    b = broker(FakeObserver(), sweep_interval=60)
    batch = await held(b, 2000, "batch", PRIORITY_BATCH)
    sup = arbiter_with_broker({"small:9b": 6000}, 20000, b, yield_wait=5.0)
    await sup.acquire(spec("small:9b"))
    sup.release("small:9b")
    assert batch.yield_request is None


async def test_yield_wait_zero_turns_asking_off_entirely():
    """The knob that makes this revertible: nothing is asked, nothing waits,
    and admission behaves exactly as it did before tier 3 existed."""
    b = broker(FakeObserver(), sweep_interval=60)
    batch = await held(b, 8000, "batch", PRIORITY_BATCH)
    sup = arbiter_with_broker({"big:27b": 19000}, 10000, b, yield_wait=0.0)
    await sup.acquire(spec("big:27b"))
    sup.release("big:27b")
    assert batch.yield_request is None


async def test_a_model_can_outrank_serving_or_duck_under_it():
    """`priority:` in the model config, so one model can be the interactive one
    without every model becoming it."""
    b = broker(FakeObserver(), sweep_interval=60)
    holder = await held(b, 8000, "holder", PRIORITY_DEFAULT)
    sup = arbiter_with_broker({"quiet:9b": 19000}, 10000, b, yield_wait=0.2)

    quiet = ModelSpec(tag="quiet:9b", priority=PRIORITY_BATCH)
    await sup.acquire(quiet)
    sup.release("quiet:9b")
    assert holder.yield_request is None, "a batch-priority model asks nobody"


async def test_a_holder_at_serving_priority_is_never_asked():
    """How a holder says "do not ask me": take a priority at or above the one
    serving admits with. A single-operator honour system, and stated as one."""
    b = broker(FakeObserver(), sweep_interval=60)
    holder = await held(b, 8000, "protected", PRIORITY_INTERACTIVE)
    sup = arbiter_with_broker({"big:27b": 19000}, 10000, b, yield_wait=0.2)
    await sup.acquire(spec("big:27b"))
    sup.release("big:27b")
    assert holder.yield_request is None

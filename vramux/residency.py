"""Who gets to be on the card, and for how long.

A *resident* is one model held in VRAM: its backend, its own in-flight request
count, its own idle clock. The arbiter admits residents, evicts them to make
room, and answers "where do I send this request".

The budget is open: more than one model may be resident, and what decides is
measured cost against the same budget leases are granted from. Three rules
carry the whole of it, and each exists because of a specific way this goes
wrong:

* **A model with no known cost is served alone.** Measured or declared, or it
  gets the card to itself — there is no estimate anywhere in this path. An
  underestimate is an OOM, and an OOM on a shared card takes the innocent
  resident down with the greedy one. A model becomes packable the first time
  it loads, because that load measures it.
* **Room is decided from `budget.free_mb`, never from a sum of resident
  costs.** The device already reports what residents, leaseholders and foreign
  processes are using; adding declared costs on top of that would count the
  same memory twice. This is the same arithmetic a lease is granted from, on
  purpose — two accountings of one card drift, and only one of them is tested.
* **Admitting is a guess until the allocation exists.** Free memory is not
  always allocatable memory, so a load can still fail after admission says
  yes. That path evicts the peers and retries once, alone, rather than
  reporting a failure that a second attempt would have survived.

The distinction that made it possible: in-flight counting is *per resident*.
Evicting model A waits on requests in flight against A, not against B. It was
built in Stage 3, when it was invisible.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .backends import Backend, DockerComposeBackend, ProcessBackend
from .observer import Observer, known_cost_mb
from .registry import KIND_DOCKER, ModelSpec

log = logging.getLogger("vramux.residency")


# How long admission waits for a leaseholder to act on a yield request before
# returning a retryable refusal. Bounded and short: the holder is being asked,
# not told, and a request that waits minutes for an answer that may never come
# has turned a cooperative gesture into a hang.
DEFAULT_YIELD_WAIT = 30.0

# What serving outranks by default — one step above the lease default, because
# a human is waiting on a chat request and nothing is waiting on a batch stage.
# A holder that does not want to be asked takes this priority or higher.
DEFAULT_SERVING_PRIORITY = 7

# How often the budget is re-read while waiting for a yield.
_YIELD_POLL = 1.0

# How long an outranked load request parks, waiting for the lease above it to
# release or expire, before refusing. The park is safe to promise because TTL
# is mandatory: the blocker *will* end, and the broker knows the latest moment
# it can. 0 restores the old behavior — refuse the moment yield concludes.
DEFAULT_QUEUE_WAIT = 600.0

# How often the budget is re-read while parked behind a lease.
_PARK_POLL = 1.0

# Slack past a blocking lease's expiry before a parked request gives up on it:
# expiry is enforced by a sweep, not at the instant, and giving up in the gap
# between the two would fail a request the very moment its memory came back.
_PARK_GRACE = 10.0

# Ceiling on residents, whatever the budget says. The budget is the real
# constraint; this is a bound on how many upstream ports and backend processes
# one card is allowed to sprout, and a place to stand if packing ever turns
# out to cost more than it saves.
DEFAULT_MAX_RESIDENTS = 2

# A cold container load legitimately takes minutes. Rather than let the card
# look wedged, a caller waiting behind it says so on this cadence.
_WAIT_LOG_INTERVAL = 15.0

# How long a waiter tolerates a load overrunning its own startup budget before
# giving up. The load is bounded; this is the slack on top of that bound.
_WAIT_SLACK = 30.0


class BackendLoading(RuntimeError):
    """A load is in progress and did not finish within its own budget.

    Raised instead of waiting forever, so the caller gets a "still loading,
    try again" rather than a socket that never answers.
    """


class InsufficientVRAM(BackendLoading):
    """A known model footprint does not fit in the currently free budget."""


class PartialOffload(BackendLoading):
    """A backend started but used far less GPU memory than its declared floor."""


MIN_DECLARED_FOOTPRINT_RATIO = 0.90


@dataclass
class Resident:
    """One model on the card.

    `drained` is set exactly when `inflight == 0`, so an evictor can wait on it
    without polling. It is per-resident on purpose: waiting on a global drain
    would make one model's slow stream block another model's eviction.
    """

    spec: ModelSpec
    backend: Backend
    inflight: int = 0
    last_use: float = 0.0
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    # The upstream port this resident borrowed, returned to the pool when it
    # stops. None for a container, which publishes its own.
    port: Optional[int] = None

    def __post_init__(self) -> None:
        self.drained.set()

    @property
    def tag(self) -> str:
        return self.spec.tag


@dataclass
class Loading:
    """A load in progress, so a waiter can report a wait instead of a silence.

    `behind` is set when this is not a load at all but a park: the model is
    queued behind a lease that outranks it, and every message about the wait
    should say so rather than describing a load that is not happening.
    """

    tag: str
    started: float
    budget: float
    behind: Optional[str] = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.elapsed)


class ResidencyArbiter:
    """Decides which models are resident and routes requests to them."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18080,
        idle_timeout: float = 900.0,  # 15min, matches OLLAMA_KEEP_ALIVE=15m
        startup_timeout: float = 180.0,
        drain_timeout: float = 120.0,
        admission_timeout: float = 660.0,
        observer: Optional[Observer] = None,
        max_residents: int = DEFAULT_MAX_RESIDENTS,
        budget=None,
        request_yield=None,
        yield_wait: float = DEFAULT_YIELD_WAIT,
        serving_priority: int = DEFAULT_SERVING_PRIORITY,
        outrankers=None,
        queue_wait: float = DEFAULT_QUEUE_WAIT,
    ) -> None:
        self.host = host
        self.port = port
        self.max_residents = max(1, int(max_residents))
        # `budget()` is the broker's, injected the same way `loading` goes the
        # other way, so residency does not import the broker and the broker
        # does not import residency. Without it — no observation, no leasing —
        # admission has no honest way to decide a second resident fits, and
        # falls back to serving one at a time.
        self._budget = budget
        # The broker's `request_yield()`, injected the same way. Tier 3 of the
        # eviction order is the only one residency cannot perform itself: it
        # can stop what it started, and a leaseholder is somebody else's
        # process. None disables asking entirely, which is what every test that
        # does not care about leases gets.
        self._request_yield = request_yield
        self.yield_wait = max(0.0, float(yield_wait))
        # What serving outranks when a model does not name its own priority.
        self.serving_priority = serving_priority
        # The broker's `outrankers()`, injected like the two above. It answers
        # the one question a refused load has left: is the memory held by
        # leases that outrank me — which will end, and are worth waiting for —
        # or by something waiting can never fix. None disables parking, which
        # is what every test written before the queue existed gets.
        self._outrankers = outrankers
        self.queue_wait = max(0.0, float(queue_wait))
        # Consecutive upstream ports, one per possible llama-server. Containers
        # publish their own and never take one.
        self._free_ports: List[int] = [port + i for i in range(self.max_residents)]
        # None disables observation entirely. It only ever watches, so nothing
        # below is allowed to change whether a load succeeds.
        self.observer = observer
        self.idle_timeout = idle_timeout
        self.startup_timeout = startup_timeout
        self.drain_timeout = drain_timeout
        # Longest a request waits for admission before being told the card is
        # busy loading. Sits above the largest startup budget so a normal cold
        # container never trips it — only a load that has stopped progressing.
        self.admission_timeout = admission_timeout

        self._residents: Dict[str, Resident] = {}
        # Why the last admission attempt did not fit, so the eviction it causes
        # says something more useful than "making room".
        self._room_reason = "the card is full"
        self._loading: Optional[Loading] = None
        self._lock = asyncio.Lock()
        self._idle_task: Optional[asyncio.Task] = None

    def use_yield(self, request_yield) -> None:
        """Hand residency the broker's `request_yield()`.

        Same shape as `use_budget`, for the same reason: one callable each way
        rather than two modules importing one another.
        """
        self._request_yield = request_yield

    def use_outrankers(self, outrankers) -> None:
        """Hand residency the broker's `outrankers()`. Same shape, same reason."""
        self._outrankers = outrankers

    def use_budget(self, budget) -> None:
        """Hand residency the broker's `budget()`.

        Called after both objects exist, because the broker needs `loading`
        from here and this needs the budget from there. One callable each way
        rather than two modules importing one another.
        """
        self._budget = budget

    # ---- what is resident -----------------------------------------------------

    @property
    def residents(self) -> List[Resident]:
        return list(self._residents.values())

    def _hot_resident(self) -> Optional[Resident]:
        """The resident to answer single-model questions with.

        With admission open there is no "the" resident, so the honest answer to
        "what is loaded" — for the `auto` sentinel, which wants to ride
        whatever is already warm rather than force a swap — is the most
        recently used one. Most-recently-*admitted* would send a side task to a
        model that was admitted once and has been cold ever since.
        """
        residents = self.residents
        if not residents:
            return None
        return max(residents, key=lambda r: r.last_use)

    @property
    def current_tag(self) -> Optional[str]:
        resident = self._hot_resident()
        return resident.tag if resident else None

    @property
    def current_spec(self) -> Optional[ModelSpec]:
        """The most recently used running model, or None when idle."""
        resident = self._hot_resident()
        if resident is not None and resident.backend.alive():
            return resident.spec
        return None

    @property
    def current_specs(self) -> List[ModelSpec]:
        """Every running model, hottest first. What `/api/ps` should report."""
        return [
            r.spec for r in sorted(self.residents, key=lambda r: -r.last_use)
            if r.backend.alive()
        ]

    def upstream_for(self, tag: str) -> str:
        """Base URL of the backend serving `tag`.

        Per-tag and not a single `upstream` property, because with two
        residents there is no single upstream and a global one would quietly
        proxy every request to whichever model was admitted last — a wrong
        answer that returns 200.
        """
        resident = self._residents.get(tag)
        if resident is not None:
            return resident.backend.upstream
        return f"http://{self.host}:{self.port}"

    @property
    def loading(self) -> Optional[Dict[str, object]]:
        """The load in progress, if any — a wait made legible."""
        load = self._loading
        if load is None:
            return None
        payload = {
            "tag": load.tag,
            "elapsed_s": round(load.elapsed, 1),
            "budget_s": round(load.budget, 1),
        }
        if load.behind is not None:
            # Not loading at all: queued behind a lease that outranks it. The
            # console needs the difference — a card that looks idle while
            # requests stack up behind a lease is exactly the "where did my
            # memory go" moment `/gpu/state` exists for.
            payload["behind"] = load.behind
        return payload

    # ---- startup --------------------------------------------------------------

    async def reconcile(self, specs) -> None:
        """Stop any router-managed container left running by a previous process.

        Without this, a router restart while a docker model is loaded leaves the
        container holding ~20 GB: the fresh arbiter believes the card is free
        and happily starts a llama-server alongside it, and both OOM.
        """
        for spec in specs:
            if spec.kind != KIND_DOCKER:
                continue
            try:
                backend = DockerComposeBackend(spec)
                backend.adopt()  # claim the orphan so stop() will act on it
                await backend.stop()
                log.info("reconcile: ensured %s container is stopped", spec.tag)
            except Exception as exc:  # never block startup on cleanup
                log.warning("reconcile: could not stop %s: %s", spec.tag, exc)

    # ---- admission ------------------------------------------------------------

    async def acquire(self, spec: ModelSpec) -> None:
        """Ensure `spec` is resident and register one in-flight request on it.

        If room has to be made, the eviction is deferred until the *victim's*
        own in-flight requests have drained — so a streaming response is never
        killed underneath its caller. Every successful acquire() MUST be paired
        with exactly one release().
        """
        await self._admit()
        try:
            resident = self._residents.get(spec.tag)
            if resident is not None and not resident.backend.alive():
                await self._stop_resident(resident)
                resident = None
            # "Running" is not "working". A backend can wedge with its process
            # or container still up — one container backend does exactly this
            # if its detokenizer stalls, answering /health with 503 while
            # accepting requests that never return. Reusing it would feed every
            # later request into a black hole, so verify health and recycle if
            # red. Only when nothing is in flight: a busy backend is allowed to
            # be slow to answer a probe.
            if (
                resident is not None
                and resident.inflight == 0
                and not await self._resident_healthy(resident)
            ):
                log.warning("loaded backend %s is unhealthy — recycling", resident.tag)
                await self._stop_resident(resident)
                resident = None
            if resident is None:
                await self._make_room_for(spec)
                resident = await self._start_with_retry(spec)
            resident.inflight += 1
            resident.drained.clear()
            resident.last_use = time.monotonic()
        finally:
            self._lock.release()

    def release(self, tag: Optional[str] = None) -> None:
        """Mark one in-flight request complete; wake a waiting eviction at zero.

        `tag` names the resident the request was acquired on, and now that more
        than one model can be resident it is how the right counter gets
        decremented. Omitting it is tolerated only when there is exactly one
        resident to mean; with two it would decrement the wrong model's
        counter, which reads as a leaked request and blocks that model's next
        eviction until the drain times out.
        """
        if tag is not None:
            resident = self._residents.get(tag)
        elif len(self._residents) == 1:
            resident = next(iter(self._residents.values()))
        else:
            if self._residents:
                log.warning(
                    "release() without a tag while %d models are resident — "
                    "nothing released", len(self._residents),
                )
            return
        if resident is None:
            # The resident was force-evicted out from under a request that
            # overran the drain timeout. Nothing left to decrement.
            return
        resident.inflight = max(0, resident.inflight - 1)
        if resident.inflight == 0:
            resident.drained.set()

    async def _admit(self) -> None:
        """Take the arbiter lock, reporting a slow load as a wait, not a hang.

        A cold container can hold this for ten minutes, which is correct — the
        GPU really is busy — but a caller sitting on a silent socket cannot
        tell that from a wedge. So: log the wait on a cadence, and bound it.
        The bound sits above the load's own startup budget, so it fires only
        when the load itself has stopped making progress.
        """
        load = self._loading
        bound = (load.remaining + _WAIT_SLACK) if load else self.admission_timeout
        reporter = (
            asyncio.create_task(self._report_wait()) if self._lock.locked() else None
        )
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=bound)
        except asyncio.TimeoutError:
            raise BackendLoading(self._wait_message(bound)) from None
        finally:
            if reporter is not None:
                reporter.cancel()

    def _wait_message(self, bound: float) -> str:
        load = self._loading
        if load is None:
            return f"the GPU is busy and did not free up within {bound:.0f}s"
        if load.behind is not None:
            return (
                f"{load.tag} is queued behind {load.behind} "
                f"({load.elapsed:.0f}s of a {load.budget:.0f}s cap) — try again shortly"
            )
        return (
            f"{load.tag} is still loading after {load.elapsed:.0f}s "
            f"(budget {load.budget:.0f}s) — try again shortly"
        )

    async def _report_wait(self) -> None:
        while True:
            await asyncio.sleep(_WAIT_LOG_INTERVAL)
            load = self._loading
            if load is None:
                log.info("waiting for the GPU to free up")
            elif load.behind is not None:
                log.info(
                    "waiting: %s is queued behind %s (%.0fs of a %.0fs cap)",
                    load.tag, load.behind, load.elapsed, load.budget,
                )
            else:
                log.info(
                    "waiting: %s is still loading (%.0fs of a %.0fs budget)",
                    load.tag, load.elapsed, load.budget,
                )

    # ---- residency changes ----------------------------------------------------

    async def _make_room_for(self, spec: ModelSpec) -> None:
        """Evict until `spec` can be admitted. Least recently used goes first.

        The loop is on *cost*, not on a count: the count ceiling is checked
        first because it is cheap, and then every remaining pass asks the
        budget whether the incoming model actually fits beside what is there.
        """
        reason = self._must_be_alone(spec)
        while self._residents:
            if reason is not None:
                await self._evict_lru(spec, reason)
                continue
            if len(self._residents) >= self.max_residents:
                await self._evict_lru(
                    spec, f"the {self.max_residents}-resident ceiling is reached"
                )
                continue
            fits = await self._fits_beside_residents(spec)
            if fits is None:  # the card cannot be read — do not guess
                await self._evict_lru(spec, "the card cannot be read")
                continue
            if fits:
                return
            await self._evict_lru(spec, self._room_reason)
        # Everything vramux can stop is stopped. What is left on the card
        # belongs to somebody else, and tier 3 of §6 is the only move: ask.
        await self._ask_leases_to_yield(spec)

    async def _ask_leases_to_yield(self, spec: ModelSpec) -> None:
        """Ask leaseholders below this model's priority for the shortfall.

        Bounded and voluntary: nothing here revokes a lease. If the known
        footprint still does not fit, refuse this request rather than letting
        llama-server auto-fit a CPU-heavy backend that stays degraded after
        the card becomes free.
        """
        cost = self._cost_mb(spec)
        if cost is None or self._budget is None:
            return
        try:
            budget = await self._budget()
        except Exception as exc:
            log.warning("could not read the budget to ask for a yield: %s", exc)
            return
        short = cost - budget.free_mb
        if short <= 0:
            return
        reason = f"{spec.tag} needs {cost} MiB but only {budget.free_mb} MiB is free"
        if self._request_yield is None or self.yield_wait <= 0:
            await self._park_behind_leases(spec, cost, reason)
            return
        priority = spec.priority if spec.priority is not None else self.serving_priority
        asked = await self._request_yield(
            short, f"serving:{spec.tag}", priority, self.yield_wait,
        )
        if not asked:
            # Nobody below this model's priority holds anything — which is
            # exactly the state a park exists for: the memory is above it.
            await self._park_behind_leases(spec, cost, reason)
            return
        log.info(
            "%s needs %d MiB more than is free; asked %d lease(s) to yield, "
            "waiting up to %.0fs", spec.tag, short, len(asked), self.yield_wait,
        )
        deadline = time.monotonic() + self.yield_wait
        while time.monotonic() < deadline:
            await asyncio.sleep(min(_YIELD_POLL, max(0.05, deadline - time.monotonic())))
            try:
                budget = await self._budget()
            except Exception:
                return
            if cost <= budget.free_mb:
                log.info("%s: the memory came back, %d MiB free", spec.tag, budget.free_mb)
                return
        await self._park_behind_leases(
            spec, cost,
            f"{spec.tag} needs {cost} MiB but only {budget.free_mb} MiB is free "
            f"after waiting {self.yield_wait:.0f}s for a lease to yield",
        )

    async def _park_behind_leases(self, spec: ModelSpec, cost: int, reason: str) -> None:
        """Wait for the leases outranking this model to release or expire.

        The one shortfall that is not an error: a lease above serving's
        priority holds the card, and the operator arranged that ranking on
        purpose. A chat request arriving mid-generation is not failing — it is
        outranked, and the correct behavior is to wait its turn. Two facts
        make the wait safe to promise: TTL is mandatory, so the blocker *will*
        end and `outrankers()` reports the latest moment it can; and this
        parks only when releasing the outrankers would actually make the load
        fit — every other cause raises `InsufficientVRAM` exactly as it did
        before the queue existed.

        Holds the arbiter lock deliberately. No other model could be served
        anyway — the card is spoken for — and a second request stacking on
        `_admit` gets the honest queue message and a bounded timeout for free,
        instead of independently discovering the lease.
        """
        if self.queue_wait <= 0 or self._outrankers is None or self._budget is None:
            raise InsufficientVRAM(reason)
        priority = spec.priority if spec.priority is not None else self.serving_priority
        started = time.monotonic()
        hard_cap = started + self.queue_wait
        previous = self._loading
        behind = None
        last_log = started
        try:
            while True:
                try:
                    budget = await self._budget()
                except Exception as exc:
                    raise InsufficientVRAM(
                        f"{reason} (and the budget became unreadable while queued: {exc})"
                    ) from None
                if cost <= budget.free_mb:
                    if behind is not None:
                        log.info(
                            "%s: the card came back after %.0fs queued",
                            spec.tag, time.monotonic() - started,
                        )
                    return
                try:
                    blockers = await self._outrankers(priority)
                except Exception as exc:
                    log.warning("could not read the leases while queued: %s", exc)
                    blockers = []
                short = cost - budget.free_mb
                if sum(b["mb"] for b in blockers) < short:
                    # Releasing every outranker would still not fit the load:
                    # foreign grew, or the memory was never theirs. Waiting
                    # cannot help, so this is the old refusal, not a queue.
                    raise InsufficientVRAM(reason)
                now = time.monotonic()
                biggest = max(blockers, key=lambda b: b["mb"])
                # Never past the moment the last blocker must renew or die. A
                # renewal pushes the horizon out; nothing pushes it past the cap.
                deadline = min(
                    hard_cap,
                    now + max(b["remaining_s"] for b in blockers) + _PARK_GRACE,
                )
                if now >= deadline:
                    raise InsufficientVRAM(
                        f"{spec.tag} needs {cost} MiB; lease {biggest['id']} "
                        f"({biggest['owner']}, priority {biggest['priority']}, "
                        f"ttl {biggest['remaining_s']:.0f}s) holds the card and "
                        f"outranks serving — gave up after {now - started:.0f}s queued"
                    )
                if behind is None:
                    behind = f"lease {biggest['id']} ({biggest['owner']})"
                    self._loading = Loading(
                        tag=spec.tag, started=started,
                        budget=self.queue_wait, behind=behind,
                    )
                    log.info(
                        "%s is queued behind %s — outranked, waiting up to %.0fs",
                        spec.tag, behind, self.queue_wait,
                    )
                if now - last_log >= _WAIT_LOG_INTERVAL:
                    last_log = now
                    log.info(
                        "queued: %s behind %s (%.0fs of ttl left)",
                        spec.tag, behind, biggest["remaining_s"],
                    )
                await asyncio.sleep(min(_PARK_POLL, max(0.05, deadline - now)))
        finally:
            # A parked request that returns, refuses, or is cancelled by its
            # client disconnecting must not leave a phantom load behind.
            self._loading = previous

    def _must_be_alone(self, spec: ModelSpec) -> Optional[str]:
        """Why `spec` cannot share the card, or None if it can.

        Three ways to end up alone, and they are all the same decision made
        from different evidence: it says so, a resident says so, or nobody
        knows what it costs.
        """
        if spec.exclusive:
            return f"{spec.tag} is exclusive"
        for resident in self.residents:
            if resident.spec.exclusive:
                return f"{resident.tag} is exclusive"
        if self._budget is None:
            return "there is no budget to decide from"
        if self._cost_mb(spec) is None:
            return f"nothing has measured {spec.tag} yet"
        return None

    def _cost_mb(self, spec: ModelSpec) -> Optional[int]:
        cache = getattr(self.observer, "cache", None)
        return known_cost_mb(cache, spec)

    async def _fits_beside_residents(self, spec: ModelSpec) -> Optional[bool]:
        """Whether `spec` fits in what is currently free. None if unreadable.

        `free_mb` is the broker's, so residents already on the card are
        accounted for by what the device reports them using — this must never
        become a sum of what vramux believes each resident costs, or the two
        accountings drift and the untested one is the one that OOMs.
        """
        cost = self._cost_mb(spec)
        if cost is None or self._budget is None:
            return None
        try:
            budget = await self._budget()
        except Exception as exc:
            log.warning("could not read the budget for admission: %s", exc)
            return None
        self._room_reason = (
            f"{spec.tag} needs {cost} MiB and {budget.free_mb} MiB is free"
        )
        return cost <= budget.free_mb

    async def _evict_lru(self, incoming: ModelSpec, reason: str) -> None:
        victim = min(self._residents.values(), key=lambda r: r.last_use)
        log.info("making room for %s: %s", incoming.tag, reason)
        await self._evict(victim, incoming=incoming)

    async def _evict(self, resident: Resident, *, incoming: Optional[ModelSpec] = None) -> None:
        """Drain this resident's own in-flight requests, then stop it."""
        if resident.inflight > 0:
            log.info(
                "deferring swap %s -> %s until %d in-flight request(s) drain",
                resident.tag, incoming.tag if incoming else "-", resident.inflight,
            )
            try:
                await asyncio.wait_for(resident.drained.wait(), timeout=self.drain_timeout)
            except asyncio.TimeoutError:
                log.warning(
                    "drain timed out after %.0fs with %d in-flight; forcing swap",
                    self.drain_timeout, resident.inflight,
                )
        if incoming is None:
            log.info("evicting %s", resident.tag)
        else:
            log.info("swapping model: %s -> %s", resident.tag, incoming.tag)
        await self._stop_resident(resident)

    async def evict(self, tag: str) -> bool:
        """Drop one resident by name. False when it was not resident.

        Goes through the same drain as an eviction made to fit something else,
        so a hand-evicted model does not kill a stream in progress either.
        """
        async with self._lock:
            resident = self._residents.get(tag)
            if resident is None:
                return False
            await self._evict(resident)
            return True

    def _effective_priority(self, resident: Resident) -> int:
        """What this resident outranks: its own say-so, else serving's."""
        priority = resident.spec.priority
        return priority if priority is not None else self.serving_priority

    async def make_room_for_lease(self, mb: int, by: str, priority: int) -> int:
        """Evict managed residents below `priority` until `mb` MiB is freed.

        The broker calls this — injected, never imported — when a lease is
        short on budget. It is the §3 tier table finally wired to leases:
        managed consumers are the tier vramux may evict silently, and until
        now only serving ever exercised that. The rule is the mirror image of
        `_ask_leases_to_yield`: a lease may evict a resident whose effective
        priority is *strictly below* its own, so a default lease (5) against
        default serving (7) evicts nothing and behaves exactly as before.

        Victims go cheapest-first among the eligible — the same reasoning
        `request_yield` gives: freeing a 2 GB embedder must be tried before a
        20 GB chat model when 2 GB covers the shortfall. Each one drains
        before it stops, exactly as a serving-side eviction would.

        Returns the number of residents evicted, not megabytes: a resident
        nobody has measured frees an amount only the card can report, and the
        caller re-reads the budget anyway. The count is the honest answer to
        the one question the caller has — did anything change.
        """
        mb = max(0, int(mb))
        if mb <= 0:
            return 0
        evicted = 0
        freed = 0
        async with self._lock:
            while freed < mb:
                eligible = [
                    r for r in self._residents.values()
                    if self._effective_priority(r) < priority
                ]
                if not eligible:
                    break
                costs = {r.tag: self._cost_mb(r.spec) for r in eligible}
                victim = min(
                    eligible,
                    key=lambda r: (costs[r.tag] is None, costs[r.tag] or 0),
                )
                cost = costs[victim.tag]
                log.info(
                    "evicting %s (%s MiB) for lease %s (priority %d over %d)",
                    victim.tag,
                    cost if cost is not None else "unmeasured",
                    by, priority, self._effective_priority(victim),
                )
                await self._evict(victim)
                evicted += 1
                if cost is None:
                    # What came back is the card's to say, not ours to count.
                    # Stop here and let the caller's re-read decide.
                    break
                freed += cost
        return evicted

    def _make_backend(self, spec: ModelSpec, port: Optional[int]) -> Backend:
        if spec.kind == KIND_DOCKER:
            return DockerComposeBackend(spec)
        return ProcessBackend(self.host, port)

    def _take_port(self) -> int:
        """Borrow an upstream port for a llama-server.

        The pool is exhausted only if the resident count ceiling has already
        been passed, which admission does not allow — but a pool that returns
        a port already in use would bind-fail in a way that reads as a broken
        model, so it raises instead.
        """
        if not self._free_ports:
            raise RuntimeError("no free upstream port — every one is in use")
        return self._free_ports.pop(0)

    def _return_port(self, port: Optional[int]) -> None:
        if port is None or port in self._free_ports:
            return
        # Ordered, so a restarted resident tends to land back on the same port
        # and a stale stderr file is about the model it says it is about.
        self._free_ports.append(port)
        self._free_ports.sort()

    def _startup_budget(self, spec: ModelSpec) -> float:
        # A docker model can legitimately take minutes on a cold container;
        # give it its own budget rather than the llama-server one.
        if spec.kind == KIND_DOCKER:
            return max(self.startup_timeout, 600.0)
        return self.startup_timeout

    async def _start_with_retry(self, spec: ModelSpec) -> Resident:
        """Start `spec`, and if it fails beside peers, try once more alone.

        Free memory is not always allocatable memory (`DESIGN.md` §11): the
        allocator fragments, and admission can honestly say yes to a load that
        then cannot place its weights. Alone, the same load usually succeeds,
        so a request that would have failed gets served instead — at the cost
        of the peers, which is the trade a swap always made anyway.

        Exactly one retry. A load that fails on an empty card is failing for
        its own reasons, and retrying that in a loop turns a broken model into
        a hung request.
        """
        try:
            return await self._start_resident(spec)
        except Exception as exc:
            peers = self.residents
            if not peers:
                raise
            log.warning(
                "%s failed to load beside %s (%s) — evicting and retrying alone",
                spec.tag, ", ".join(r.tag for r in peers), exc,
            )
            for peer in peers:
                await self._evict(peer, incoming=spec)
            return await self._start_resident(spec)

    async def _start_resident(self, spec: ModelSpec) -> Resident:
        # The port is taken here rather than inside `_make_backend` so that it
        # is returned to the pool by the one place that knows the resident
        # died — a backend that raises on the way up would otherwise keep it.
        port = None if spec.kind == KIND_DOCKER else self._take_port()
        try:
            backend = self._make_backend(spec, port)
        except Exception:
            self._return_port(port)
            raise
        resident = Resident(spec=spec, backend=backend, port=port)
        self._residents[spec.tag] = resident
        budget = self._startup_budget(spec)
        self._loading = Loading(tag=spec.tag, started=time.monotonic(), budget=budget)
        try:
            if self.observer is None:
                await backend.start(spec, budget)
            else:
                async with self.observer.measuring(spec) as measurement:
                    await backend.start(spec, budget)
                    # Claim before the window closes, or the process we just
                    # started reads as foreign drift and voids its own
                    # measurement.
                    self.observer.claim(spec.tag, await self._backend_pids(backend))
                declared = spec.vram_mb
                # Third-party/test observers written before measurement
                # results were exposed may yield no result object. Observation
                # remains optional; absence must not break an otherwise valid
                # load.
                measured = getattr(measurement, "measured_mb", None)
                if (
                    declared
                    and measured is not None
                    and measured < int(declared * MIN_DECLARED_FOOTPRINT_RATIO)
                ):
                    raise PartialOffload(
                        f"{spec.tag} started with only {measured} MiB on the GPU; "
                        f"its declared floor is {declared} MiB"
                    )
        except Exception:
            await self._stop_resident(resident)
            raise
        finally:
            self._loading = None
        self._ensure_idle_watcher()
        return resident

    async def _backend_pids(self, backend: Backend) -> List[int]:
        try:
            return await backend.pids()
        except Exception as exc:  # attribution is a nicety, never a blocker
            log.debug("could not determine pids for the running backend: %s", exc)
            return []

    async def _resident_healthy(self, resident: Resident) -> bool:
        try:
            return await resident.backend.healthy()
        except Exception as exc:  # an unanswerable probe is a red one
            log.debug("health probe for %s failed: %s", resident.tag, exc)
            return False

    async def _stop_resident(self, resident: Resident) -> None:
        tag = resident.tag
        before = None
        if self.observer is not None:
            before = await self.observer._safe_snapshot()
        try:
            await resident.backend.stop()
        finally:
            # Both in a `finally`: a backend whose stop() raises still has to
            # leave the bookkeeping clean, or a corpse holds a port and a slot
            # forever and the next load fails for a reason nobody can see.
            self._residents.pop(tag, None)
            self._return_port(resident.port)
        if self.observer is not None:
            self.observer.release(tag)
            await self.observer.observe_unload(tag, before)

    async def stop(self) -> None:
        """Evict everything. Used on shutdown and by the unload endpoint.

        Takes the lock directly rather than going through `_admit`: an unload
        that arrives mid-load should queue behind it, and shutdown must not be
        able to fail with `BackendLoading`.
        """
        async with self._lock:
            for resident in self.residents:
                await self._stop_resident(resident)

    # ---- idle -----------------------------------------------------------------

    def touch(self) -> None:
        resident = self._hot_resident()
        if resident is not None:
            resident.last_use = time.monotonic()

    def _effective_idle_timeout(self, resident: Optional[Resident] = None) -> float:
        if resident is None:
            resident = self._hot_resident()
        if resident is not None and resident.spec.idle_timeout is not None:
            return resident.spec.idle_timeout
        return self.idle_timeout

    def _ensure_idle_watcher(self) -> None:
        if self._idle_task and not self._idle_task.done():
            return
        self._idle_task = asyncio.create_task(self._idle_watch())

    async def _idle_watch(self) -> None:
        """One watcher for every resident: each has its own clock and timeout."""
        while True:
            await asyncio.sleep(30)
            if not self._residents:
                return
            now = time.monotonic()
            expired = [
                r for r in self.residents
                if r.inflight == 0 and now - r.last_use > self._effective_idle_timeout(r)
            ]
            dead = [r for r in self.residents if not r.backend.alive()]
            if not expired and not dead:
                continue
            async with self._lock:
                for resident in expired:
                    if resident.inflight:  # re-check under lock
                        continue
                    if self._residents.get(resident.tag) is not resident:
                        continue
                    log.info(
                        "idle timeout (%.0fs), unloading %s",
                        self._effective_idle_timeout(resident), resident.tag,
                    )
                    await self._stop_resident(resident)
                for resident in dead:
                    # Died on its own — reap the bookkeeping so a later acquire
                    # does not reuse a corpse.
                    if self._residents.get(resident.tag) is resident:
                        log.warning("backend for %s is gone — reaping", resident.tag)
                        await self._stop_resident(resident)
            if not self._residents:
                return

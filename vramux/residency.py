"""Who gets to be on the card, and for how long.

A *resident* is one model held in VRAM: its backend, its own in-flight request
count, its own idle clock. The arbiter admits residents, evicts them to make
room, and answers "where do I send this request".

Everything here is written for more than one resident. Admission is the single
exception: `_ADMITTED_RESIDENTS` is 1, so today's behaviour is exactly the old
one-model-at-a-time swap. Opening it needs a memory budget built from measured
costs, which is Stage 6's job — the structure is ready, the budget is shut.

The distinction that matters: in-flight counting is *per resident*. Evicting
model A waits on requests in flight against A, not against B. With one
resident that is a distinction without a difference; with two it is the whole
game, and retrofitting it later would mean re-reasoning about every drain.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .backends import Backend, DockerComposeBackend, ProcessBackend
from .observer import Observer
from .registry import KIND_DOCKER, ModelSpec

log = logging.getLogger("vramux.residency")


# How many models may be resident at once. One, until measured costs can prove
# a second one fits. Every other part of this module is already plural.
_ADMITTED_RESIDENTS = 1

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

    def __post_init__(self) -> None:
        self.drained.set()

    @property
    def tag(self) -> str:
        return self.spec.tag


@dataclass
class Loading:
    """A load in progress, so a waiter can report a wait instead of a silence."""

    tag: str
    started: float
    budget: float

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
    ) -> None:
        self.host = host
        self.port = port
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
        self._loading: Optional[Loading] = None
        self._lock = asyncio.Lock()
        self._idle_task: Optional[asyncio.Task] = None

    # ---- what is resident -----------------------------------------------------

    @property
    def residents(self) -> List[Resident]:
        return list(self._residents.values())

    def _sole_resident(self) -> Optional[Resident]:
        """The resident to answer single-model questions with.

        Admission is one, so "the" resident is well defined. When it opens, the
        callers of this — `/api/ps`, the `auto` model sentinel — need a real
        answer about *which* model, and this is where that lands. Until then,
        most-recently-admitted is both correct and the whole set.
        """
        residents = self.residents
        return residents[-1] if residents else None

    @property
    def current_tag(self) -> Optional[str]:
        resident = self._sole_resident()
        return resident.tag if resident else None

    @property
    def current_spec(self) -> Optional[ModelSpec]:
        """The ModelSpec currently loaded (running), or None when idle."""
        resident = self._sole_resident()
        if resident is not None and resident.backend.alive():
            return resident.spec
        return None

    @property
    def upstream(self) -> str:
        """Base URL of the loaded backend.

        Backend-dependent: llama-server binds our fixed upstream port, a docker
        model publishes its own. Falls back to the llama-server port when
        nothing is loaded.
        """
        resident = self._sole_resident()
        if resident is not None:
            return resident.backend.upstream
        return f"http://{self.host}:{self.port}"

    @property
    def loading(self) -> Optional[Dict[str, object]]:
        """The load in progress, if any — a wait made legible."""
        load = self._loading
        if load is None:
            return None
        return {
            "tag": load.tag,
            "elapsed_s": round(load.elapsed, 1),
            "budget_s": round(load.budget, 1),
        }

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
                resident = await self._start_resident(spec)
            resident.inflight += 1
            resident.drained.clear()
            resident.last_use = time.monotonic()
        finally:
            self._lock.release()

    def release(self, tag: Optional[str] = None) -> None:
        """Mark one in-flight request complete; wake a waiting eviction at zero.

        `tag` names the resident the request was acquired on. It is optional
        only while admission is one — omitting it then is unambiguous. A caller
        that knows its tag should pass it, and from Stage 6 must.
        """
        resident = self._residents.get(tag) if tag is not None else self._sole_resident()
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
            else:
                log.info(
                    "waiting: %s is still loading (%.0fs of a %.0fs budget)",
                    load.tag, load.elapsed, load.budget,
                )

    # ---- residency changes ----------------------------------------------------

    async def _make_room_for(self, spec: ModelSpec) -> None:
        """Evict until `spec` can be admitted. Least recently used goes first."""
        while len(self._residents) >= _ADMITTED_RESIDENTS:
            victim = min(self._residents.values(), key=lambda r: r.last_use)
            await self._evict(victim, incoming=spec)

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
        log.info("swapping model: %s -> %s", resident.tag, incoming.tag if incoming else "-")
        await self._stop_resident(resident)

    def _make_backend(self, spec: ModelSpec) -> Backend:
        if spec.kind == KIND_DOCKER:
            return DockerComposeBackend(spec)
        return ProcessBackend(self.host, self.port)

    def _startup_budget(self, spec: ModelSpec) -> float:
        # A docker model can legitimately take minutes on a cold container;
        # give it its own budget rather than the llama-server one.
        if spec.kind == KIND_DOCKER:
            return max(self.startup_timeout, 600.0)
        return self.startup_timeout

    async def _start_resident(self, spec: ModelSpec) -> Resident:
        backend = self._make_backend(spec)
        resident = Resident(spec=spec, backend=backend)
        self._residents[spec.tag] = resident
        budget = self._startup_budget(spec)
        self._loading = Loading(tag=spec.tag, started=time.monotonic(), budget=budget)
        try:
            if self.observer is None:
                await backend.start(spec, budget)
            else:
                async with self.observer.measuring(spec):
                    await backend.start(spec, budget)
                    # Claim before the window closes, or the process we just
                    # started reads as foreign drift and voids its own
                    # measurement.
                    self.observer.claim(spec.tag, await self._backend_pids(backend))
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
            self._residents.pop(tag, None)
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
        resident = self._sole_resident()
        if resident is not None:
            resident.last_use = time.monotonic()

    def _effective_idle_timeout(self, resident: Optional[Resident] = None) -> float:
        if resident is None:
            resident = self._sole_resident()
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

"""Swap, drain and recycle logic against a fake backend.

No GPU, no subprocess, no docker. The arbiter's real backends are replaced
wholesale, so what is under test is the arbitration policy itself.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

import pytest

from vramux.budget import Budget
from vramux.observer import cost_key
from vramux.registry import KIND_DOCKER, ModelSpec
from vramux.residency import ResidencyArbiter


class FakeBackend:
    """Records its lifecycle into a shared event log.

    Implements the `Backend` protocol; health is delegated back to the fake
    arbiter so a test can turn a live backend red mid-run.
    """

    def __init__(self, spec: ModelSpec, arbiter: "FakeArbiter", start_delay: float = 0.0,
                 fail: bool = False) -> None:
        self.spec = spec
        self.upstream = "http://fake"
        self.arbiter = arbiter
        self.events = arbiter.events
        self.start_delay = start_delay
        self.fail = fail
        self.started = False
        self.startup_timeout: Optional[float] = None

    def alive(self) -> bool:
        return self.started

    async def healthy(self) -> bool:
        self.arbiter.health_checks += 1
        return self.arbiter.healthy

    def adopt(self) -> None:
        self.started = True

    async def start(self, spec: ModelSpec, startup_timeout: float) -> None:
        self.startup_timeout = startup_timeout
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        if self.fail:
            self.events.append(f"start-failed:{spec.tag}")
            raise RuntimeError("boom")
        self.started = True
        self.events.append(f"start:{spec.tag}")

    async def pids(self):
        return [4242] if self.started else []

    async def stop(self) -> None:
        if self.started:
            self.events.append(f"stop:{self.spec.tag}")
        self.started = False


class FakeArbiter(ResidencyArbiter):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.events: List[str] = []
        self.backends: List[FakeBackend] = []
        self.healthy = True
        self.health_checks = 0
        self.start_delay = 0.0
        self.fail_next_start = False

    def _make_backend(self, spec: ModelSpec, port=None) -> FakeBackend:
        b = FakeBackend(spec, self, self.start_delay, self.fail_next_start)
        self.fail_next_start = False
        self.backends.append(b)
        return b


def spec(tag: str, **kw) -> ModelSpec:
    return ModelSpec(tag=tag, **kw)


@pytest.fixture
def sup():
    return FakeArbiter(drain_timeout=0.2, startup_timeout=5.0)


# ---- basic slot behaviour -------------------------------------------------


async def test_acquire_starts_and_release_leaves_it_loaded(sup):
    a = spec("a:1b")
    await sup.acquire(a)
    assert sup.events == ["start:a:1b"]
    assert sup.current_tag == "a:1b" and sup.current_spec is a
    sup.release()
    assert sup.current_spec is a, "release must not unload; only a swap or idle does"


async def test_repeated_acquire_of_same_model_does_not_restart(sup):
    a = spec("a:1b")
    await sup.acquire(a)
    sup.release()
    await sup.acquire(a)
    sup.release()
    assert sup.events == ["start:a:1b"]


async def test_swap_stops_before_starting(sup):
    a, b = spec("a:1b"), spec("b:2b")
    await sup.acquire(a)
    sup.release()
    await sup.acquire(b)
    sup.release()
    assert sup.events == ["start:a:1b", "stop:a:1b", "start:b:2b"]


async def test_upstream_tracks_the_loaded_backend(sup):
    assert sup.upstream_for("a:1b") == "http://127.0.0.1:18080"
    await sup.acquire(spec("a:1b"))
    assert sup.upstream_for("a:1b") == "http://fake"
    # an unknown tag falls back rather than borrowing another model's backend
    assert sup.upstream_for("b:2b") == "http://127.0.0.1:18080"


async def test_current_spec_is_none_when_nothing_loaded(sup):
    assert sup.current_spec is None and sup.current_tag is None


async def test_failed_start_leaves_the_slot_empty_and_raises(sup):
    sup.fail_next_start = True
    with pytest.raises(RuntimeError):
        await sup.acquire(spec("a:1b"))
    assert sup.current_spec is None
    # The lock must have been released, or the router deadlocks on the next call.
    await asyncio.wait_for(sup.acquire(spec("b:2b")), timeout=1)


async def test_docker_gets_a_longer_startup_budget(sup):
    await sup.acquire(spec("d:35b", kind=KIND_DOCKER))
    assert sup.backends[-1].startup_timeout == 600.0


async def test_stop_unloads(sup):
    await sup.acquire(spec("a:1b"))
    sup.release()
    await sup.stop()
    assert sup.current_spec is None
    assert sup.events == ["start:a:1b", "stop:a:1b"]


async def test_evict_by_name_unloads_that_resident(sup):
    await sup.acquire(spec("a:1b"))
    sup.release()
    assert await sup.evict("a:1b") is True
    assert sup.current_spec is None
    assert sup.events == ["start:a:1b", "stop:a:1b"]


async def test_evicting_something_not_resident_reports_it_rather_than_raising(sup):
    assert await sup.evict("nothing:9b") is False


async def test_evict_by_name_drains_first(sup):
    """A hand-evicted model must not kill a stream in progress either."""
    a = spec("a:1b")
    await sup.acquire(a)  # one request in flight, never released
    task = asyncio.create_task(sup.evict("a:1b"))
    await asyncio.sleep(0.05)
    assert sup.current_spec is a, "eviction must wait for the drain"
    sup.release()
    assert await asyncio.wait_for(task, timeout=1) is True
    assert sup.current_spec is None


# ---- drain ----------------------------------------------------------------


async def test_swap_waits_for_inflight_request_to_drain(sup):
    """A streaming response must never have its backend torn down underneath
    it. The swap blocks until the in-flight count reaches zero."""
    a, b = spec("a:1b"), spec("b:2b")
    await sup.acquire(a)  # one request in flight on `a`

    swap = asyncio.create_task(sup.acquire(b))
    await asyncio.sleep(0.05)
    assert not swap.done(), "swap must not proceed while a request is in flight"
    assert sup.events == ["start:a:1b"]

    sup.release()
    await asyncio.wait_for(swap, timeout=1)
    assert sup.events == ["start:a:1b", "stop:a:1b", "start:b:2b"]
    sup.release()


async def test_drain_timeout_forces_the_swap(sup):
    """A client that never finishes must not wedge the slot forever."""
    a, b = spec("a:1b"), spec("b:2b")
    await sup.acquire(a)  # never released
    await asyncio.wait_for(sup.acquire(b), timeout=2)
    assert sup.events == ["start:a:1b", "stop:a:1b", "start:b:2b"]


async def test_concurrent_requests_for_the_same_model_share_one_backend(sup):
    a = spec("a:1b")
    await asyncio.gather(*(sup.acquire(a) for _ in range(5)))
    assert sup.events == ["start:a:1b"]
    for _ in range(5):
        sup.release()


async def test_release_never_goes_negative(sup):
    sup.release()
    sup.release()
    await sup.acquire(spec("a:1b"))
    sup.release()
    # A stray release must not leave the drain gate stuck closed.
    a = asyncio.create_task(sup.acquire(spec("b:2b")))
    await asyncio.wait_for(a, timeout=1)


# ---- health recycle -------------------------------------------------------


async def test_wedged_backend_is_recycled_on_next_acquire(sup):
    """The observed failure mode: process/container still up, /health 503, and
    every request fed to it hangs forever. Reusing it is worse than a restart."""
    a = spec("a:1b")
    await sup.acquire(a)
    sup.release()

    sup.healthy = False
    await sup.acquire(a)
    assert sup.events == ["start:a:1b", "stop:a:1b", "start:a:1b"]
    assert len(sup.backends) == 2


async def test_health_is_not_checked_while_requests_are_in_flight(sup):
    """A busy backend can be slow to answer a probe. Tearing it down for that
    would kill a working stream."""
    a = spec("a:1b")
    await sup.acquire(a)  # still in flight
    sup.healthy = False
    await sup.acquire(a)
    assert sup.events == ["start:a:1b"], "must not recycle a backend that is serving"
    assert sup.health_checks == 0


async def test_health_check_skipped_when_swapping_anyway(sup):
    await sup.acquire(spec("a:1b"))
    sup.release()
    checks = sup.health_checks
    await sup.acquire(spec("b:2b"))
    assert sup.health_checks == checks, "a different model is being loaded regardless"


# ---- idle -----------------------------------------------------------------


async def test_per_model_idle_timeout_overrides_the_default(sup):
    await sup.acquire(spec("d:35b", kind=KIND_DOCKER, idle_timeout=3600.0))
    assert sup._effective_idle_timeout() == 3600.0
    sup.release()
    await sup.acquire(spec("a:1b"))
    assert sup._effective_idle_timeout() == sup.idle_timeout


# ---- reconcile ------------------------------------------------------------


async def test_reconcile_stops_orphaned_containers_only(monkeypatch):
    """A router restart while a container is loaded leaves it holding ~20 GB
    while the fresh arbiter believes the card is free."""
    from vramux import residency as sup_mod

    stopped: List[str] = []

    class RecordingDocker:
        def __init__(self, spec):
            self.spec = spec
            self._running = False

        def adopt(self):
            self._running = True

        async def stop(self):
            if self._running:
                stopped.append(self.spec.tag)

    monkeypatch.setattr(sup_mod, "DockerComposeBackend", RecordingDocker)
    s = FakeArbiter()
    await s.reconcile([
        spec("a:1b"),
        spec("d:35b", kind=KIND_DOCKER),
        spec("e:9b", kind=KIND_DOCKER),
    ])
    assert stopped == ["d:35b", "e:9b"]


async def test_reconcile_failure_does_not_block_startup(monkeypatch):
    from vramux import residency as sup_mod

    class ExplodingDocker:
        def __init__(self, spec):
            raise RuntimeError("docker not installed")

    monkeypatch.setattr(sup_mod, "DockerComposeBackend", ExplodingDocker)
    await FakeArbiter().reconcile([spec("d:35b", kind=KIND_DOCKER)])


# ---- observation ----------------------------------------------------------


class RecordingObserver:
    """Stands in for the real observer: records the calls, decides nothing."""

    def __init__(self) -> None:
        self.claims = []
        self.releases = []
        self.measured = []
        self.unloads = []

    def claim(self, owner, pids=()):
        self.claims.append((owner, list(pids)))

    def release(self, owner):
        self.releases.append(owner)

    async def _safe_snapshot(self):
        return "snapshot"

    async def observe_unload(self, tag, before):
        self.unloads.append((tag, before))

    def measuring(self, spec):
        observer = self

        class _Ctx:
            async def __aenter__(self):
                observer.measured.append(spec.tag)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


async def test_a_load_is_measured_and_its_pids_claimed():
    obs = RecordingObserver()
    sup = FakeArbiter(observer=obs)
    await sup.acquire(spec("a:1b"))
    assert obs.measured == ["a:1b"]
    # Claimed inside the measurement window, or the process just started reads
    # as foreign drift and voids its own measurement.
    assert obs.claims == [("a:1b", [4242])]


async def test_an_unload_releases_the_claim_and_is_observed():
    obs = RecordingObserver()
    sup = FakeArbiter(observer=obs)
    await sup.acquire(spec("a:1b"))
    sup.release()
    await sup.stop()
    assert obs.releases == ["a:1b"]
    assert obs.unloads == [("a:1b", "snapshot")]


async def test_a_backend_that_cannot_report_pids_still_loads():
    class NoPids(FakeBackend):
        async def pids(self):
            raise RuntimeError("docker compose top failed")

    obs = RecordingObserver()
    sup = FakeArbiter(observer=obs)
    sup._make_backend = lambda spec_, port=None: NoPids(spec_, sup)
    await sup.acquire(spec("d:35b", kind=KIND_DOCKER))
    assert sup.current_tag == "d:35b"
    assert obs.claims == [("d:35b", [])]


async def test_observation_is_optional():
    """The supervisor must run identically with no observer attached."""
    sup = FakeArbiter(observer=None)
    await sup.acquire(spec("a:1b"))
    sup.release()
    await sup.stop()
    assert sup.events == ["start:a:1b", "stop:a:1b"]


# ---- multi-residency ------------------------------------------------------
#
# The card is faked in one direction only: costs come from a stand-in cache and
# free memory from a stand-in budget, so what is under test is the packing
# decision and nothing about NVML.


class FakeCache:
    """Measured costs, by cost key. `record` is what a real load would do."""

    def __init__(self, by_tag=None) -> None:
        self.by_tag = dict(by_tag or {})

    def get(self, key):
        for tag, mb in self.by_tag.items():
            if cost_key(ModelSpec(tag=tag)) == key:
                return {"tag": tag, "measured_mb": mb}
        return None

    def all(self):
        return {}


class CostedObserver(RecordingObserver):
    def __init__(self, costs=None) -> None:
        super().__init__()
        self.cache = FakeCache(costs)


def packing_arbiter(costs=None, free_mb=20000, **kw) -> FakeArbiter:
    """An arbiter that can pack: measured costs, and a card with room."""
    sup = FakeArbiter(observer=CostedObserver(costs), **kw)
    sup.free_mb = free_mb
    sup.use_budget(lambda: _fake_budget(sup))
    return sup


async def _fake_budget(sup):
    return Budget(
        total_mb=24564, reserve_mb=1024,
        used_mb=24564 - 1024 - sup.free_mb,
        recognised_mb=0, foreign_mb=0, unattributed_mb=0,
    )


async def test_two_measured_models_are_resident_at_once():
    """The stage, in one test: a second model joins instead of replacing."""
    sup = packing_arbiter({"a:9b": 6591, "b:9b": 6195}, free_mb=20000)
    await sup.acquire(spec("a:9b"))
    sup.release("a:9b")
    await sup.acquire(spec("b:9b"))
    sup.release("b:9b")

    assert sup.events == ["start:a:9b", "start:b:9b"], "nothing should have stopped"
    assert sorted(r.tag for r in sup.residents) == ["a:9b", "b:9b"]


async def test_a_model_nobody_has_measured_is_served_alone():
    """No estimate anywhere in this path: an underestimate is an OOM that
    takes the innocent resident with it."""
    sup = packing_arbiter({"a:9b": 6591}, free_mb=20000)
    await sup.acquire(spec("a:9b"))
    sup.release("a:9b")
    await sup.acquire(spec("unmeasured:12b"))
    sup.release("unmeasured:12b")

    assert sup.events == ["start:a:9b", "stop:a:9b", "start:unmeasured:12b"]


async def test_a_declared_cost_is_enough_to_pack_a_container():
    """A container's internals are not introspectable, so `vram_mb:` is the
    operator's word for it — and it is enough."""
    sup = packing_arbiter({"a:9b": 6591}, free_mb=20000)
    await sup.acquire(spec("a:9b"))
    sup.release("a:9b")
    await sup.acquire(spec("c:7b", kind=KIND_DOCKER, vram_mb=5000))
    sup.release("c:7b")

    assert sup.events == ["start:a:9b", "start:c:7b"]


async def test_a_second_model_that_does_not_fit_evicts_the_first():
    sup = packing_arbiter({"a:9b": 6591, "big:27b": 18970}, free_mb=10000)
    await sup.acquire(spec("a:9b"))
    sup.release("a:9b")
    await sup.acquire(spec("big:27b"))
    sup.release("big:27b")

    assert sup.events == ["start:a:9b", "stop:a:9b", "start:big:27b"]


async def test_an_exclusive_model_takes_the_card_in_both_directions():
    sup = packing_arbiter({"a:9b": 6591, "solo:35b": 6000}, free_mb=20000)
    await sup.acquire(spec("a:9b"))
    sup.release("a:9b")
    # cheap enough to fit twice over, and still alone because it says so
    await sup.acquire(spec("solo:35b", exclusive=True))
    sup.release("solo:35b")
    assert sup.events == ["start:a:9b", "stop:a:9b", "start:solo:35b"]
    # and nothing joins it either
    await sup.acquire(spec("a:9b"))
    sup.release("a:9b")
    assert sup.events[-2:] == ["stop:solo:35b", "start:a:9b"]


async def test_the_resident_ceiling_holds_even_when_everything_fits():
    sup = packing_arbiter(
        {"a:9b": 100, "b:9b": 100, "c:9b": 100}, free_mb=20000, max_residents=2,
    )
    for tag in ("a:9b", "b:9b", "c:9b"):
        await sup.acquire(spec(tag))
        sup.release(tag)
    assert len(sup.residents) == 2
    assert "stop:a:9b" in sup.events, "the least recently used goes first"


async def test_a_load_that_fails_beside_a_peer_retries_alone():
    """Free memory is not always allocatable memory. Admission can honestly
    say yes to a load that then cannot place its weights."""
    sup = packing_arbiter({"a:9b": 6591, "b:9b": 6195}, free_mb=20000)
    await sup.acquire(spec("a:9b"))
    sup.release("a:9b")

    started = {"n": 0}
    real_make = sup._make_backend

    def flaky(spec_, port=None):
        started["n"] += 1
        backend = real_make(spec_, port)
        backend.fail = started["n"] == 1 and spec_.tag == "b:9b"
        return backend

    sup._make_backend = flaky
    await sup.acquire(spec("b:9b"))
    sup.release("b:9b")

    assert sup.events == [
        "start:a:9b", "start-failed:b:9b", "stop:a:9b", "start:b:9b",
    ]
    assert [r.tag for r in sup.residents] == ["b:9b"]


async def test_a_load_that_fails_alone_is_not_retried():
    """A load failing on an empty card is failing for its own reasons, and
    retrying that in a loop turns a broken model into a hung request."""
    sup = packing_arbiter({"a:9b": 6591})
    sup.fail_next_start = True
    with pytest.raises(RuntimeError):
        await sup.acquire(spec("a:9b"))
    assert sup.events == ["start-failed:a:9b"]


async def test_each_resident_gets_its_own_upstream_port():
    """Two llama-servers cannot share one port, and a global upstream would
    proxy every request to whichever model was admitted last."""
    sup = packing_arbiter({"a:9b": 100, "b:9b": 100}, free_mb=20000)
    await sup.acquire(spec("a:9b"))
    sup.release("a:9b")
    await sup.acquire(spec("b:9b"))
    sup.release("b:9b")

    ports = sorted(r.port for r in sup.residents)
    assert ports == [18080, 18081]

    await sup.evict("a:9b")
    # the freed port goes back to the pool rather than being burned
    await sup.acquire(spec("a:9b"))
    sup.release("a:9b")
    assert sorted(r.port for r in sup.residents) == [18080, 18081]


async def test_a_container_takes_no_port_from_the_pool():
    sup = packing_arbiter({"c:7b": 100, "a:9b": 100}, free_mb=20000)
    sup._make_backend = lambda spec_, port=None: FakeBackend(spec_, sup)
    await sup.acquire(spec("c:7b", kind=KIND_DOCKER, vram_mb=100))
    sup.release("c:7b")
    assert sup.residents[0].port is None


async def test_auto_follows_the_hottest_resident_not_the_newest():
    """`auto` exists so a side task rides whatever is already warm. With two
    residents, most-recently-*admitted* would send it to the cold one."""
    sup = packing_arbiter({"a:9b": 100, "b:9b": 100}, free_mb=20000)
    await sup.acquire(spec("a:9b"))
    sup.release("a:9b")
    await sup.acquire(spec("b:9b"))
    sup.release("b:9b")
    assert sup.current_tag == "b:9b"

    await sup.acquire(spec("a:9b"))  # a is used again; b goes cold
    sup.release("a:9b")
    assert sup.current_tag == "a:9b"
    assert len(sup.residents) == 2, "using a warm resident must not swap anything"


async def test_release_without_a_tag_is_refused_when_it_is_ambiguous():
    sup = packing_arbiter({"a:9b": 100, "b:9b": 100}, free_mb=20000)
    await sup.acquire(spec("a:9b"))
    sup.release()  # unambiguous: one resident
    await sup.acquire(spec("b:9b"))
    sup.release()  # ambiguous: decrementing the wrong counter blocks a drain
    assert [r.inflight for r in sup.residents] == [0, 1]
    sup.release("b:9b")
    assert [r.inflight for r in sup.residents] == [0, 0]

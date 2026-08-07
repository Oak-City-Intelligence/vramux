"""Swap, drain and recycle logic against a fake backend.

No GPU, no subprocess, no docker. The supervisor's real backends are replaced
wholesale, so what is under test is the arbitration policy itself.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

import pytest

from vramux.registry import KIND_DOCKER, ModelSpec
from vramux.supervisor import LlamaServerSupervisor


class FakeBackend:
    """Records its lifecycle into a shared event log."""

    def __init__(self, spec: ModelSpec, events: List[str], start_delay: float = 0.0,
                 fail: bool = False) -> None:
        self.spec = spec
        self.upstream = "http://fake"
        self.events = events
        self.start_delay = start_delay
        self.fail = fail
        self.started = False
        self.startup_timeout: Optional[float] = None

    def alive(self) -> bool:
        return self.started

    async def start(self, spec: ModelSpec, startup_timeout: float) -> None:
        self.startup_timeout = startup_timeout
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        if self.fail:
            self.events.append(f"start-failed:{spec.tag}")
            raise RuntimeError("boom")
        self.started = True
        self.events.append(f"start:{spec.tag}")

    async def stop(self) -> None:
        if self.started:
            self.events.append(f"stop:{self.spec.tag}")
        self.started = False


class FakeSupervisor(LlamaServerSupervisor):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.events: List[str] = []
        self.backends: List[FakeBackend] = []
        self.healthy = True
        self.health_checks = 0
        self.start_delay = 0.0
        self.fail_next_start = False

    def _make_backend(self, spec: ModelSpec) -> FakeBackend:
        b = FakeBackend(spec, self.events, self.start_delay, self.fail_next_start)
        self.fail_next_start = False
        self.backends.append(b)
        return b

    async def _backend_healthy(self) -> bool:
        self.health_checks += 1
        return self.healthy


def spec(tag: str, **kw) -> ModelSpec:
    return ModelSpec(tag=tag, **kw)


@pytest.fixture
def sup():
    return FakeSupervisor(drain_timeout=0.2, startup_timeout=5.0)


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
    assert sup.upstream == "http://127.0.0.1:18080"
    await sup.acquire(spec("a:1b"))
    assert sup.upstream == "http://fake"


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
    while the fresh supervisor believes the slot is free."""
    from vramux import supervisor as sup_mod

    stopped: List[str] = []

    class RecordingDocker:
        def __init__(self, spec):
            self.spec = spec
            self._started = False

        async def stop(self):
            if self._started:
                stopped.append(self.spec.tag)

    monkeypatch.setattr(sup_mod, "DockerComposeBackend", RecordingDocker)
    s = FakeSupervisor()
    await s.reconcile([
        spec("a:1b"),
        spec("d:35b", kind=KIND_DOCKER),
        spec("e:9b", kind=KIND_DOCKER),
    ])
    assert stopped == ["d:35b", "e:9b"]


async def test_reconcile_failure_does_not_block_startup(monkeypatch):
    from vramux import supervisor as sup_mod

    class ExplodingDocker:
        def __init__(self, spec):
            raise RuntimeError("docker not installed")

    monkeypatch.setattr(sup_mod, "DockerComposeBackend", ExplodingDocker)
    await FakeSupervisor().reconcile([spec("d:35b", kind=KIND_DOCKER)])

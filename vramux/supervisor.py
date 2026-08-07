"""Supervises the single GPU slot, swapping models on demand.

llama-server serves one model per process. Ollama auto-loads/unloads. The
supervisor mimics that: when a request asks for a different model than the
one currently loaded, the running backend is torn down and the new one is
started. After `idle_timeout`, it is shut down to free GPU memory.

Two backends implement that lifecycle:

* `ProcessBackend` — spawns llama-server on a local GGUF.
* `DockerComposeBackend` — brings a compose service up and down. For models
  llama.cpp cannot load at all, which ship their own OpenAI-compatible server.

Both occupy the same slot, so a GGUF and a container never hold VRAM at once.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Callable, List, Optional

import aiohttp

from . import env
from .registry import KIND_DOCKER, ModelSpec

log = logging.getLogger("vramux.supervisor")


def resolve_llama_server_bin() -> Optional[Path]:
    """Locate the llama.cpp server binary.

    An explicit `VRAMUX_LLAMA_SERVER_BIN` wins — a build tree is the common
    case and is rarely on `$PATH`. Otherwise take whatever `$PATH` offers.
    Resolved per start rather than at import so the answer follows the
    environment the process is actually running in.
    """
    override = env.get("LLAMA_SERVER_BIN")
    if override:
        return Path(override).expanduser()
    found = shutil.which("llama-server")
    return Path(found) if found else None


async def _wait_for_health(
    upstream: str,
    deadline: float,
    *,
    died: Callable[[], Optional[str]],
    label: str,
) -> None:
    """Poll `upstream/health` until 200, the deadline, or the backend dies.

    `died()` returns a reason string once the backend is gone, else None — so a
    crashed process/container fails fast instead of burning the whole timeout.
    """
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            reason = died()
            if reason:
                raise RuntimeError(f"{label} died before becoming ready: {reason}")
            try:
                async with session.get(
                    f"{upstream}/health", timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError(f"{label} did not become ready in time")


class ProcessBackend:
    """A llama-server subprocess serving one GGUF."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.upstream = f"http://{host}:{port}"
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_fh = None

    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self, spec: ModelSpec, startup_timeout: float) -> None:
        binary = resolve_llama_server_bin()
        if binary is None:
            raise RuntimeError(
                "llama-server not found on $PATH — set VRAMUX_LLAMA_SERVER_BIN "
                "to the binary in your llama.cpp build"
            )
        if not binary.is_file():
            raise RuntimeError(f"llama-server binary not found at {binary}")
        if spec.gguf_path is None or not spec.gguf_path.is_file():
            raise RuntimeError(f"GGUF not found for {spec.tag}: {spec.gguf_path}")

        args = [
            str(binary),
            "-m", str(spec.gguf_path),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(spec.ctx_size),
            "--alias", spec.tag,
            "--jinja",  # apply chat template from GGUF metadata
        ]
        if spec.n_gpu_layers is not None:
            args.extend(["-ngl", str(spec.n_gpu_layers)])
        # else: omit -ngl entirely so llama-server's auto-fit picks the
        # largest layer count that fits in free VRAM.
        if spec.is_embedding:
            args.append("--embeddings")
        args.extend(spec.extra_args)

        # Stream stderr to a per-load file so segfaults / OOMs are diagnosable.
        stderr_path = Path(
            f"/tmp/llama-server-{spec.tag.replace('/', '_').replace(':', '_')}.stderr"
        )
        log.info(
            "starting llama-server for %s (ctx=%d) — stderr at %s",
            spec.tag, spec.ctx_size, stderr_path,
        )
        self._stderr_fh = stderr_path.open("wb")
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=self._stderr_fh,
            preexec_fn=os.setsid,
        )

        def died() -> Optional[str]:
            rc = self._proc.returncode if self._proc else None
            return None if rc is None else f"exited rc={rc} (see {stderr_path})"

        await _wait_for_health(
            self.upstream,
            time.monotonic() + startup_timeout,
            died=died,
            label=f"llama-server for {spec.tag}",
        )

    async def stop(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self._proc.wait()
        self._proc = None
        if self._stderr_fh:
            try:
                self._stderr_fh.close()
            except OSError:
                pass
            self._stderr_fh = None


class DockerComposeBackend:
    """A compose service that ships its own OpenAI-compatible server.

    `up -d` is idempotent and reuses the existing container, so a stopped
    container restarts without being recreated — that keeps warm JIT/kernel
    caches in its volumes and makes restart much cheaper than a cold build.
    """

    def __init__(self, spec: ModelSpec) -> None:
        if spec.port is None or spec.compose_file is None or not spec.compose_service:
            raise RuntimeError(
                f"docker spec {spec.tag} needs compose_file, compose_service and port"
            )
        self.spec = spec
        self.upstream = f"http://127.0.0.1:{spec.port}"
        self._started = False

    def alive(self) -> bool:
        return self._started

    def _compose(self, *args: str) -> List[str]:
        return [
            "docker", "compose",
            "-f", str(self.spec.compose_file),
            *args,
        ]

    async def _run(self, *args: str, timeout: float = 120.0) -> tuple:
        proc = await asyncio.create_subprocess_exec(
            *self._compose(*args),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"`docker compose {' '.join(args)}` timed out after {timeout}s")
        return proc.returncode, (out or b"").decode(errors="replace").strip()

    async def start(self, spec: ModelSpec, startup_timeout: float) -> None:
        service = spec.compose_service
        log.info("starting container %s for %s (ctx=%d)", service, spec.tag, spec.ctx_size)
        rc, out = await self._run("up", "-d", service)
        if rc != 0:
            raise RuntimeError(f"docker compose up failed for {spec.tag}: {out}")
        self._started = True

        async def _died_check() -> Optional[str]:
            rc, out = await self._run("ps", "-q", "--status", "running", service, timeout=15)
            return None if (rc == 0 and out) else "container not running"

        # Container health is polled over HTTP; a crash-looping container is
        # caught by the compose ps check every few seconds rather than every
        # poll, since shelling out is far more expensive than a GET.
        deadline = time.monotonic() + startup_timeout
        last_ps = 0.0
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                try:
                    async with session.get(
                        f"{self.upstream}/health", timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            log.info("container ready for %s", spec.tag)
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                now = time.monotonic()
                if now - last_ps > 10:
                    last_ps = now
                    reason = await _died_check()
                    if reason:
                        await self.stop()
                        raise RuntimeError(f"container for {spec.tag} {reason}")
                await asyncio.sleep(0.5)
        await self.stop()
        raise RuntimeError(f"container for {spec.tag} did not become ready within {startup_timeout}s")

    async def stop(self) -> None:
        if not self._started:
            return
        service = self.spec.compose_service
        # `stop`, not `down`: keeps the container (and its warm caches) around
        # so the next load is a restart rather than a recreate.
        rc, out = await self._run("stop", service, timeout=90)
        if rc != 0:
            log.warning("docker compose stop failed for %s: %s", self.spec.tag, out)
        self._started = False


class LlamaServerSupervisor:
    """Owns the single GPU slot: one backend loaded at a time."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18080,
        idle_timeout: float = 900.0,  # 15min, matches OLLAMA_KEEP_ALIVE=15m
        startup_timeout: float = 180.0,
        drain_timeout: float = 120.0,
    ) -> None:
        self.host = host
        self.port = port
        self.idle_timeout = idle_timeout
        self.startup_timeout = startup_timeout
        self.drain_timeout = drain_timeout

        self._backend = None
        self._current: Optional[ModelSpec] = None
        self._last_use: float = 0.0
        self._lock = asyncio.Lock()
        self._idle_task: Optional[asyncio.Task] = None
        # In-flight request accounting so a model swap never tears down a
        # llama-server that is still streaming a response. `_drained` is set
        # exactly when `_inflight == 0`.
        self._inflight: int = 0
        self._drained = asyncio.Event()
        self._drained.set()

    @property
    def current_tag(self) -> Optional[str]:
        return self._current.tag if self._current else None

    @property
    def current_spec(self) -> Optional[ModelSpec]:
        """The ModelSpec currently loaded (running), or None when idle."""
        if self._backend is not None and self._backend.alive():
            return self._current
        return None

    @property
    def upstream(self) -> str:
        """Base URL of the loaded backend.

        Backend-dependent: llama-server binds our fixed upstream port, a docker
        model publishes its own. Only meaningful between acquire() and
        release(); falls back to the llama-server port when nothing is loaded.
        """
        if self._backend is not None:
            return self._backend.upstream
        return f"http://{self.host}:{self.port}"

    async def _backend_healthy(self) -> bool:
        """One cheap GET against the loaded backend's /health.

        Only consulted when nothing is in flight, so a busy-but-fine backend is
        never torn down for being slow to answer a health probe.
        """
        if self._backend is None:
            return False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._backend.upstream}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def reconcile(self, specs) -> None:
        """Stop any router-managed container left running by a previous process.

        Without this, a router restart while a docker model is loaded leaves the
        container holding ~20 GB: the fresh supervisor believes the slot is free
        and happily starts a llama-server alongside it, and both OOM.
        """
        for spec in specs:
            if spec.kind != KIND_DOCKER:
                continue
            try:
                backend = DockerComposeBackend(spec)
                backend._started = True  # force the stop path
                await backend.stop()
                log.info("reconcile: ensured %s container is stopped", spec.tag)
            except Exception as exc:  # never block startup on cleanup
                log.warning("reconcile: could not stop %s: %s", spec.tag, exc)

    def _effective_idle_timeout(self) -> float:
        if self._current is not None and self._current.idle_timeout is not None:
            return self._current.idle_timeout
        return self.idle_timeout

    async def acquire(self, spec: ModelSpec) -> None:
        """Ensure `spec` is loaded and register one in-flight request on it.

        If a *different* model is loaded, the swap is deferred until all
        in-flight requests on the current model have drained — so a streaming
        response is never killed underneath its caller. Every successful
        acquire() MUST be paired with exactly one release().
        """
        async with self._lock:
            already = (
                self._backend is not None
                and self._backend.alive()
                and self._current is spec
            )
            # "Running" is not "working". A backend can wedge with its process
            # or container still up — one container backend does exactly this if its
            # detokenizer stalls, answering /health with 503 while accepting
            # requests that never return. Reusing it would feed every later
            # request into a black hole, so verify health and recycle if red.
            if already and self._inflight == 0 and not await self._backend_healthy():
                log.warning("loaded backend %s is unhealthy — recycling", self.current_tag)
                await self._stop_unlocked()
                already = False
            if not already:
                if self._backend is not None and self._backend.alive():
                    if self._inflight > 0:
                        log.info(
                            "deferring swap %s -> %s until %d in-flight request(s) drain",
                            self.current_tag, spec.tag, self._inflight,
                        )
                        try:
                            await asyncio.wait_for(self._drained.wait(), timeout=self.drain_timeout)
                        except asyncio.TimeoutError:
                            log.warning(
                                "drain timed out after %.0fs with %d in-flight; forcing swap",
                                self.drain_timeout, self._inflight,
                            )
                    log.info("swapping model: %s -> %s", self.current_tag, spec.tag)
                    await self._stop_unlocked()
                await self._start_unlocked(spec)
            self._inflight += 1
            self._drained.clear()
            self._last_use = time.monotonic()

    def release(self) -> None:
        """Mark one in-flight request complete; wake a waiting swap at zero."""
        self._inflight = max(0, self._inflight - 1)
        if self._inflight == 0:
            self._drained.set()

    def _make_backend(self, spec: ModelSpec):
        if spec.kind == KIND_DOCKER:
            return DockerComposeBackend(spec)
        return ProcessBackend(self.host, self.port)

    async def _start_unlocked(self, spec: ModelSpec) -> None:
        backend = self._make_backend(spec)
        self._backend = backend
        self._current = spec
        # A docker model can legitimately take minutes on a cold container;
        # give it its own budget rather than the llama-server one.
        startup_timeout = self.startup_timeout
        if spec.kind == KIND_DOCKER:
            startup_timeout = max(startup_timeout, 600.0)
        try:
            await backend.start(spec, startup_timeout)
        except Exception:
            await self._stop_unlocked()
            raise
        self._ensure_idle_watcher()

    async def _stop_unlocked(self) -> None:
        if self._backend is None:
            return
        try:
            await self._backend.stop()
        finally:
            self._backend = None
            self._current = None

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_unlocked()

    def touch(self) -> None:
        self._last_use = time.monotonic()

    def _ensure_idle_watcher(self) -> None:
        if self._idle_task and not self._idle_task.done():
            return
        self._idle_task = asyncio.create_task(self._idle_watch())

    async def _idle_watch(self) -> None:
        while True:
            await asyncio.sleep(30)
            if self._backend is None or not self._backend.alive():
                return
            timeout = self._effective_idle_timeout()
            if self._inflight == 0 and time.monotonic() - self._last_use > timeout:
                log.info("idle timeout (%.0fs), unloading %s", timeout, self.current_tag)
                async with self._lock:
                    if self._inflight == 0:  # re-check under lock
                        await self._stop_unlocked()
                return

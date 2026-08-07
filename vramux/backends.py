"""What it takes to be a backend, and the two kinds that exist.

A backend is one loaded model with an OpenAI-compatible HTTP server in front
of it. The arbiter decides *which* backends are resident; this module is only
concerned with getting one up, knowing whether it is working, and getting it
down again.

* `ProcessBackend` — spawns llama-server on a local GGUF.
* `DockerComposeBackend` — brings a compose service up and down. For models
  llama.cpp cannot load at all, which ship their own server.

`Backend` is the contract, pinned deliberately: vLLM and TensorRT-LLM are the
obvious next kinds, and the interface should stop moving before they arrive. A
new kind is a new class in this file — the arbiter does not learn about it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Callable, List, Optional, Protocol, runtime_checkable

import aiohttp

from . import env
from .registry import ModelSpec

log = logging.getLogger("vramux.backends")


@runtime_checkable
class Backend(Protocol):
    """One model, running, reachable over HTTP.

    Everything here is per-instance: no implementation may assume it is the
    only backend on the card, because from Stage 6 it will not be.
    """

    #: Base URL of this backend's server. Valid once `start()` returns.
    upstream: str

    def alive(self) -> bool:
        """Whether this backend believes its server is up.

        Cheap and local — no I/O. "Up" is not "working": see `healthy()`.
        """
        ...

    async def start(self, spec: ModelSpec, startup_timeout: float) -> None:
        """Bring the model up and return only once it answers /health.

        Raises on failure, having cleaned up whatever it started. Must not
        return early: the caller treats a return as "ready to serve".
        """
        ...

    async def stop(self) -> None:
        """Release the VRAM. Idempotent, and must not raise on an already-stopped
        backend — it is called on the error path of `start()`."""
        ...

    async def healthy(self) -> bool:
        """One cheap probe: is this backend actually answering?

        Lives here rather than on the arbiter because what "healthy" means is
        the kind's business — an HTTP GET today, possibly not for the next
        kind. The arbiter owns the *policy* of when to ask and what to do with
        the answer, which is why this returns a bool and never acts on it.
        """
        ...

    async def pids(self) -> List[int]:
        """Host PIDs holding this backend's VRAM, best effort.

        Used only for attributing an NVML reading to an owner. An empty list
        means "could not tell", never "holding nothing", and must never affect
        whether a load proceeds.
        """
        ...

    def adopt(self) -> None:
        """Declare that an instance started by a *previous* process is running,
        so that `stop()` will tear it down.

        Used by reconcile at startup. Kinds with no way to re-attach to an
        orphan raise; that is a property of the kind, not a failure.
        """
        ...


async def http_healthy(upstream: str, timeout: float = 5.0) -> bool:
    """One GET against `upstream/health`. Any error is a no, never an exception."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{upstream}/health", timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                return resp.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False


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

    async def healthy(self) -> bool:
        return await http_healthy(self.upstream)

    async def pids(self) -> List[int]:
        """The process holding VRAM is the one we spawned."""
        return [self._proc.pid] if self._proc is not None else []

    def adopt(self) -> None:
        """Cannot re-attach: a subprocess handle does not survive our restart.

        An orphaned llama-server is a different problem from an orphaned
        container — there is no handle to reclaim, only a PID we never recorded.
        """
        raise NotImplementedError(
            "a llama-server subprocess cannot be adopted after a restart"
        )

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


def parse_compose_top(out: str) -> List[int]:
    """Pull host PIDs out of `docker compose top`.

    The column layout is not stable across compose versions — some emit the
    bare `ps` header (`UID PID PPID ...`), newer ones prefix `SERVICE` and a
    replica number. Reading a fixed column index silently collects the wrong
    number, so the header is what decides: find `PID` in it, and use that
    position until the next header.
    """
    pids: List[int] = []
    pid_column = None
    for line in out.splitlines():
        fields = line.split()
        if not fields:
            continue
        if "PID" in fields:
            pid_column = fields.index("PID")
            continue
        if pid_column is None or len(fields) <= pid_column:
            continue
        try:
            pids.append(int(fields[pid_column]))
        except ValueError:
            continue
    return pids


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
        self._running = False

    def alive(self) -> bool:
        return self._running

    async def healthy(self) -> bool:
        return await http_healthy(self.upstream)

    def adopt(self) -> None:
        """Claim a container this process did not start.

        Compose is addressed by file and service name, not by a handle, so an
        instance left behind by a previous router is reachable by exactly the
        same commands. Adopting it is therefore only a matter of admitting it
        exists, after which `stop()` does the real work.
        """
        self._running = True

    async def pids(self) -> List[int]:
        """Host PIDs of the container's processes, best effort.

        `docker compose top` reports host PIDs, and NVML reports host PIDs for
        container compute processes, so the two meet — verified against a real
        container backend. This only affects whether a load is attributed or
        counted as foreign, never whether it proceeds.
        """
        if not self._running:
            return []
        rc, out = await self._run("top", self.spec.compose_service, timeout=15)
        if rc != 0:
            return []
        return parse_compose_top(out)

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
        self._running = True

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
        if not self._running:
            return
        service = self.spec.compose_service
        # `stop`, not `down`: keeps the container (and its warm caches) around
        # so the next load is a restart rather than a recreate.
        rc, out = await self._run("stop", service, timeout=90)
        if rc != 0:
            log.warning("docker compose stop failed for %s: %s", self.spec.tag, out)
        self._running = False

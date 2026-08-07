"""Reading the true state of the card.

Everything here is read-only and best-effort. A machine with no GPU, no driver,
or an `nvidia-smi` that answers something unexpected must degrade to "I cannot
see the card" rather than raising — the observer is not allowed to be the reason
the router stops working.

`nvidia-smi` rather than a Python NVML binding: it ships with the driver, so
there is nothing to install on a machine that can run a model at all. The shape
of this module is the shape of NVML, so swapping the probe later is local.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger("vramux.nvml")

_SMI = "nvidia-smi"

_DEVICE_QUERY = "index,name,memory.total,memory.used,memory.free"
_PROCESS_QUERY = "pid,used_memory,process_name"

# nvidia-smi answers with these when a field is unavailable rather than
# omitting it — MIG devices, permission-limited containers, older drivers.
_UNAVAILABLE = ("[N/A]", "[Not Supported]", "[Unknown Error]", "")


@dataclass(frozen=True)
class GpuProcess:
    """One compute process holding memory on the device."""

    pid: int
    used_mb: int
    name: str

    @property
    def short_name(self) -> str:
        """Command basename, without the arguments nvidia-smi includes."""
        head = self.name.split()[0] if self.name.split() else self.name
        return head.rsplit("/", 1)[-1] or self.name


@dataclass(frozen=True)
class DeviceState:
    """A point-in-time reading of one device."""

    index: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    processes: List[GpuProcess] = field(default_factory=list)

    @property
    def accounted_mb(self) -> int:
        """Sum of what individual processes admit to holding.

        Usually less than `used_mb`: driver and context overhead is real memory
        that belongs to no single process. The gap is why a budget built from
        process sums alone runs optimistic.
        """
        return sum(p.used_mb for p in self.processes)

    @property
    def unattributed_mb(self) -> int:
        return max(0, self.used_mb - self.accounted_mb)


def available() -> bool:
    return shutil.which(_SMI) is not None


def _parse_int(raw: str) -> Optional[int]:
    raw = raw.strip()
    if raw in _UNAVAILABLE:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _parse_devices(out: str) -> List[DeviceState]:
    devices: List[DeviceState] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        # The device name can itself contain a comma, so bound the split by the
        # number of fields we asked for and let the name absorb the remainder.
        parts = line.split(",")
        if len(parts) < 5:
            log.debug("unparseable device line: %r", line)
            continue
        index = _parse_int(parts[0])
        total = _parse_int(parts[-3])
        used = _parse_int(parts[-2])
        free = _parse_int(parts[-1])
        name = ",".join(parts[1:-3]).strip()
        if index is None or total is None or used is None:
            log.debug("device line missing required fields: %r", line)
            continue
        devices.append(DeviceState(
            index=index,
            name=name,
            total_mb=total,
            used_mb=used,
            free_mb=free if free is not None else max(0, total - used),
        ))
    return devices


def _parse_processes(out: str) -> List[GpuProcess]:
    procs: List[GpuProcess] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        # A process command line is full of commas. Only the first two fields
        # are ours; everything after the second comma is the name.
        parts = line.split(",", 2)
        if len(parts) < 3:
            continue
        pid = _parse_int(parts[0])
        used = _parse_int(parts[1])
        if pid is None:
            continue
        # A process we cannot size still holds memory and must not vanish from
        # the picture; it contributes 0 to sums and is visible in the listing.
        procs.append(GpuProcess(pid=pid, used_mb=used or 0, name=parts[2].strip()))
    return procs


def _run(query: str, what: str) -> Optional[str]:
    try:
        res = subprocess.run(
            [_SMI, f"--query-{what}={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("nvidia-smi %s failed: %s", what, exc)
        return None
    if res.returncode != 0:
        log.debug("nvidia-smi %s exited %d: %s", what, res.returncode, res.stderr.strip())
        return None
    return res.stdout


async def _arun(query: str, what: str) -> Optional[str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            _SMI, f"--query-{what}={query}", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=10)
    except (OSError, asyncio.TimeoutError) as exc:
        log.debug("nvidia-smi %s failed: %s", what, exc)
        return None
    if proc.returncode != 0:
        log.debug("nvidia-smi %s exited %s", what, proc.returncode)
        return None
    return out.decode(errors="replace")


def _assemble(dev_out: Optional[str], proc_out: Optional[str], index: int) -> Optional[DeviceState]:
    if dev_out is None:
        return None
    devices = _parse_devices(dev_out)
    device = next((d for d in devices if d.index == index), None)
    if device is None:
        return None
    procs = _parse_processes(proc_out) if proc_out else []
    return DeviceState(
        index=device.index,
        name=device.name,
        total_mb=device.total_mb,
        used_mb=device.used_mb,
        free_mb=device.free_mb,
        processes=procs,
    )


def probe(index: int = 0) -> Optional[DeviceState]:
    """Read the device, or None if it cannot be read. Never raises."""
    if not available():
        return None
    return _assemble(_run(_DEVICE_QUERY, "gpu"), _run(_PROCESS_QUERY, "compute-apps"), index)


async def aprobe(index: int = 0) -> Optional[DeviceState]:
    """`probe()` without blocking the event loop."""
    if not available():
        return None
    dev_out, proc_out = await asyncio.gather(
        _arun(_DEVICE_QUERY, "gpu"),
        _arun(_PROCESS_QUERY, "compute-apps"),
    )
    return _assemble(dev_out, proc_out, index)

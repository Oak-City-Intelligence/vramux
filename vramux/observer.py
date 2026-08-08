"""What is on the card, and what each managed load actually costs.

The observer only watches. It never blocks a load, never evicts anything, and
nothing consults its numbers to make a decision yet. That is deliberate: the
cost model is the largest OOM risk in the design, and it should be answering
from weeks of recorded measurements by the time anything depends on it.

Two jobs:

* **Attribution.** Split what the device reports into memory vramux started and
  memory it did not. The second kind is the honest part — a compositor, a
  browser, someone else's training run. It is never reclaimed and never
  pretended away.
* **Measurement.** Sample around every managed load, and write the delta to a
  cache keyed by the configuration that determines footprint. A measurement is
  discarded rather than recorded when something else moved during the window.
* **History.** Record what the card looked like on a timer, so foreign usage
  over time is something that can be read back rather than something that
  scrolled past in the journal.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import env, nvml
from .nvml import DeviceState, GpuProcess

log = logging.getLogger("vramux.observer")

CACHE_VERSION = 1

# A measurement window is only clean if foreign usage held still through it.
# Slack for a compositor repainting, not for another model loading.
FOREIGN_DRIFT_TOLERANCE_MB = 64


# A history row is ~110 bytes, so this is a couple of megabytes at most and
# roughly a month of five-minute samples. Bounded on purpose: an unbounded log
# in a cache directory is a disk-filler waiting for a long-lived service.
HISTORY_MAX_ROWS = 20000


def _cache_dir() -> Path:
    override = env.get("CACHE_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(base).expanduser() / "vramux"


def _cache_path() -> Path:
    return _cache_dir() / "costs.json"


def _history_path() -> Path:
    return _cache_dir() / "usage.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_EPOCH = datetime.fromtimestamp(0, timezone.utc)


def _parse_ts(value) -> Optional[datetime]:
    """A recorded timestamp as a datetime, or None if it is not one.

    Rows are written by `_now()` and are always tz-aware, but the file is a
    plain log a human may have edited, so a naive timestamp is read as UTC
    rather than raising on a comparison.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def cost_key(spec) -> str:
    """Identity of everything that determines a model's footprint.

    Two configurations that differ here have genuinely different costs, and
    two that match are interchangeable. Context length matters as much as the
    weights: the same GGUF at 16 K and at 128 K differ by many gigabytes.
    """
    parts = [
        spec.kind,
        spec.tag,
        str(spec.ctx_size),
        str(spec.n_gpu_layers),
        spec.quantization or "",
        " ".join(spec.extra_args or []),
        str(spec.gguf_path or spec.compose_service or ""),
    ]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def known_cost_mb(cache, spec) -> Optional[int]:
    """What this model costs, or `None` when nobody actually knows.

    Two sources and no third one. A measurement of this exact configuration is
    authoritative; a `vram_mb:` in config is the operator's word for backends
    whose internals cannot be introspected. There is deliberately no estimate:
    `DESIGN.md` §4.2 describes one, and an estimate is exactly what must not
    decide whether a second model joins a card that is already holding one —
    an underestimate is an OOM that takes the innocent resident with it. A
    model with no known cost is served alone, which is what vramux did for its
    whole life before this, and it becomes packable the first time it loads.
    """
    entry = cache.get(cost_key(spec)) if cache is not None else None
    if entry and entry.get("measured_mb"):
        return int(entry["measured_mb"])
    declared = getattr(spec, "vram_mb", None)
    if declared:
        return int(declared)
    return None


@dataclass
class Attribution:
    """One process, labelled with who vramux thinks it belongs to."""

    process: GpuProcess
    owner: Optional[str]  # None means foreign

    @property
    def is_foreign(self) -> bool:
        return self.owner is None


@dataclass
class Snapshot:
    """A reading of the card with ownership resolved."""

    device: DeviceState
    attributions: List[Attribution] = field(default_factory=list)
    # Managed residents whose processes could not be located on the device.
    # Container backends land here: their GPU work runs under host PIDs vramux
    # never learned, so their memory is counted as foreign. Correct but
    # pessimistic, and worth seeing rather than hiding.
    unlocated_owners: List[str] = field(default_factory=list)

    @property
    def recognised_mb(self) -> int:
        return sum(a.process.used_mb for a in self.attributions if not a.is_foreign)

    @property
    def foreign_mb(self) -> int:
        return sum(a.process.used_mb for a in self.attributions if a.is_foreign)

    @property
    def foreign(self) -> List[Attribution]:
        return [a for a in self.attributions if a.is_foreign]

    def render(self) -> str:
        d = self.device
        lines = [
            f"device {d.index}: {d.name}",
            f"  total {d.total_mb} MiB   used {d.used_mb} MiB   free {d.free_mb} MiB",
            f"  recognised {self.recognised_mb} MiB   foreign {self.foreign_mb} MiB"
            f"   unattributed {d.unattributed_mb} MiB",
        ]
        if self.attributions:
            lines.append("")
            lines.append(f"  {'PID':>8}  {'MiB':>7}  {'OWNER':<18} PROCESS")
            for a in sorted(self.attributions, key=lambda x: -x.process.used_mb):
                owner = a.owner or "— foreign —"
                lines.append(
                    f"  {a.process.pid:>8}  {a.process.used_mb:>7}  {owner:<18} {a.process.short_name}"
                )
        else:
            lines.append("  no compute processes on the device")
        if self.unlocated_owners:
            lines.append("")
            lines.append(
                "  resident but not located on the device (counted as foreign): "
                + ", ".join(self.unlocated_owners)
            )
        return "\n".join(lines)


class CostCache:
    """Measured footprints, keyed by configuration. Written, never yet read
    by anything that decides."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or _cache_path()
        self._entries: Dict[str, dict] = {}
        self._loaded = False

    def load(self) -> None:
        self._loaded = True
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            return
        entries = data.get("entries")
        if isinstance(entries, dict):
            self._entries = entries

    def get(self, key: str) -> Optional[dict]:
        if not self._loaded:
            self.load()
        return self._entries.get(key)

    def record(self, key: str, spec, measured_mb: int) -> None:
        if not self._loaded:
            self.load()
        prior = self._entries.get(key) or {}
        self._entries[key] = {
            "tag": spec.tag,
            "kind": spec.kind,
            "ctx": spec.ctx_size,
            "measured_mb": measured_mb,
            "samples": int(prior.get("samples", 0)) + 1,
            "previous_mb": prior.get("measured_mb"),
            "updated": _now(),
        }
        self._flush()

    def all(self) -> Dict[str, dict]:
        if not self._loaded:
            self.load()
        return dict(self._entries)

    def _flush(self) -> None:
        payload = json.dumps(
            {"version": CACHE_VERSION, "entries": self._entries}, indent=2, sort_keys=True
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Rename over the old file so a crash mid-write cannot truncate it.
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(payload)
            tmp.replace(self.path)
        except OSError as exc:
            log.debug("could not write cost cache at %s: %s", self.path, exc)


class UsageLog:
    """What the card looked like, over time.

    The cost cache answers "what does this model cost"; this answers "what was
    the rest of the machine doing", which is the number a budget is wrong about
    when it is wrong. Append-only JSON lines, trimmed to a bound rather than
    rotated — one file, readable with `tail`, and incapable of filling a disk.
    """

    def __init__(self, path: Optional[Path] = None, max_rows: int = HISTORY_MAX_ROWS) -> None:
        self.path = path or _history_path()
        self.max_rows = max_rows
        self._rows: Optional[int] = None

    def record(self, snap: "Snapshot") -> None:
        row = {
            "t": _now(),
            "used_mb": snap.device.used_mb,
            "free_mb": snap.device.free_mb,
            "recognised_mb": snap.recognised_mb,
            "foreign_mb": snap.foreign_mb,
            "unattributed_mb": snap.device.unattributed_mb,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError as exc:
            log.debug("could not write usage history at %s: %s", self.path, exc)
            return
        self._rows = (self._count_rows() if self._rows is None else self._rows) + 1
        # Rewrite in one go when the file has drifted well past the bound,
        # rather than on every append: trimming is the rare path.
        if self._rows > self.max_rows * 1.25:
            self._trim()

    def rows(self, limit: Optional[int] = None) -> List[dict]:
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return []
        if limit is not None:
            lines = lines[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line is not worth failing over
        return out

    def recent(self, minutes: Optional[float] = None, limit: Optional[int] = None) -> List[dict]:
        """The tail of the history, oldest first.

        Age is filtered after the read rather than by seeking to a timestamp:
        the file is bounded at `max_rows`, so the whole of it is a couple of
        megabytes and one pass over it costs less than being clever. A row
        whose timestamp cannot be parsed is dropped when a window was asked
        for — it cannot be shown to fall inside one.
        """
        rows = self.rows()
        if minutes is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            rows = [r for r in rows if (_parse_ts(r.get("t")) or _EPOCH) >= cutoff]
        if limit is not None and limit > 0:
            rows = rows[-limit:]
        return rows

    def _count_rows(self) -> int:
        try:
            with self.path.open() as fh:
                return sum(1 for _ in fh)
        except OSError:
            return 0

    def _trim(self) -> None:
        try:
            with self.path.open() as fh:
                kept = fh.readlines()[-self.max_rows:]
            tmp = self.path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(kept))
            tmp.replace(self.path)
            self._rows = len(kept)
        except OSError as exc:
            log.debug("could not trim usage history at %s: %s", self.path, exc)


class Observer:
    """Read-only accounting for one device."""

    def __init__(
        self,
        device_index: int = 0,
        cache: Optional[CostCache] = None,
        probe=nvml.aprobe,
        history: Optional[UsageLog] = None,
    ) -> None:
        self.device_index = device_index
        self.cache = cache if cache is not None else CostCache()
        self.history = history if history is not None else UsageLog()
        self._probe = probe
        # owner label -> pids vramux started for it
        self._owned: Dict[str, List[int]] = {}

    # ---- ownership -------------------------------------------------------

    def claim(self, owner: str, pids=()) -> None:
        """Record that `owner`'s work runs under `pids`.

        A backend with no PIDs to give still claims, with an empty list. Its
        memory then reads as foreign — pessimistic, but the honest answer, and
        it shows up in `unlocated_owners` rather than being quietly assumed.
        """
        self._owned.setdefault(owner, [])
        self._owned[owner].extend(p for p in pids if p is not None)

    def release(self, owner: str) -> None:
        self._owned.pop(owner, None)

    # ---- reading ---------------------------------------------------------

    async def snapshot(self) -> Optional[Snapshot]:
        device = await self._probe(self.device_index)
        if device is None:
            return None
        by_pid = {pid: owner for owner, pids in self._owned.items() for pid in pids}
        attributions = [
            Attribution(process=p, owner=by_pid.get(p.pid)) for p in device.processes
        ]
        located = {a.owner for a in attributions if a.owner}
        unlocated = sorted(o for o in self._owned if o not in located)
        return Snapshot(device=device, attributions=attributions, unlocated_owners=unlocated)

    async def sample(self) -> Optional[Snapshot]:
        """Take a reading and store it. Called on the broker's timer.

        Failing to record a sample is a missing data point, never an error
        anybody hears about — the same rule the rest of this module runs under.
        """
        snap = await self._safe_snapshot()
        if snap is not None:
            self.history.record(snap)
        return snap

    async def log_snapshot(self, note: str = "") -> Optional[Snapshot]:
        snap = await self.snapshot()
        if snap is None:
            return None
        log.info(
            "gpu%s: used %d/%d MiB, free %d, recognised %d, foreign %d across %d process(es)",
            f" ({note})" if note else "",
            snap.device.used_mb, snap.device.total_mb, snap.device.free_mb,
            snap.recognised_mb, snap.foreign_mb, len(snap.attributions),
        )
        return snap

    # ---- measuring -------------------------------------------------------

    @asynccontextmanager
    async def measuring(self, spec):
        """Bracket a managed load and record what it cost.

        Nothing raised in here can stop the load: a failed measurement is a
        missing data point, not a failed request.
        """
        before = await self._safe_snapshot()
        started = time.monotonic()
        try:
            yield
        except Exception:
            raise
        else:
            after = await self._safe_snapshot()
            self._record(spec, before, after, time.monotonic() - started)

    async def _safe_snapshot(self) -> Optional[Snapshot]:
        try:
            return await self.snapshot()
        except Exception as exc:  # never let observation break a load
            log.debug("snapshot failed: %s", exc)
            return None

    def _record(self, spec, before: Optional[Snapshot], after: Optional[Snapshot],
                elapsed: float) -> None:
        if before is None or after is None:
            return
        drift = abs(after.foreign_mb - before.foreign_mb)
        delta = after.device.used_mb - before.device.used_mb
        if drift > FOREIGN_DRIFT_TOLERANCE_MB:
            log.info(
                "not recording cost for %s: foreign usage moved %d MiB during the load",
                spec.tag, drift,
            )
            return
        if delta <= 0:
            log.debug("not recording cost for %s: delta %d MiB", spec.tag, delta)
            return
        key = cost_key(spec)
        prior = self.cache.get(key)
        self.cache.record(key, spec, delta)
        if prior and prior.get("measured_mb"):
            log.info(
                "measured %s at %d MiB in %.1fs (was %d MiB)",
                spec.tag, delta, elapsed, prior["measured_mb"],
            )
        else:
            log.info("measured %s at %d MiB in %.1fs (first measurement)",
                     spec.tag, delta, elapsed)

    async def observe_unload(self, tag: str, before: Optional[Snapshot]) -> None:
        after = await self._safe_snapshot()
        if before is None or after is None:
            return
        reclaimed = before.device.used_mb - after.device.used_mb
        log.info("unloaded %s, %d MiB returned, %d MiB now free",
                 tag, reclaimed, after.device.free_mb)

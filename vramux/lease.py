"""Granting memory to consumers vramux does not run.

A lease is a promise that `mb` megabytes stay available to its holder until it
is released or expires. It is *bookkeeping, not the allocation* — granting one
moves nothing, and dropping one frees nothing. What it buys is that vramux will
not promise the same memory to somebody else.

Three things in here are load-bearing and none of them is obvious:

* **TTL is the guarantee, the wrapper is only the fast path.** A holder killed
  with `SIGKILL` runs no cleanup, by definition. Server-side expiry is what
  stops a dead holder stranding the card forever, so it is built and tested
  first and the CLI's release-on-exit is an optimisation on top of it.
* **A grant is charged only for the shortfall** (`DESIGN.md` §5.2). A holder
  re-acquiring after a broker restart already has its memory on the card, where
  vramux sees it as foreign. Charging the full request again would shrink the
  card by the size of the holder's own allocation. See `budget.py`.
* **Nothing is granted while a model is loading.** A load in flight has not
  allocated yet, so the device reads free when it is about to be anything but.
  The request waits — and gets an honest `408` if it runs out of patience —
  rather than being handed memory that is already spoken for.

Broker restart drops every lease. Holders demote to foreign: vramux stops
knowing whose memory that is, still sees it via NVML, and still subtracts it.
The budget stays true, which is the invariant that matters.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

from . import budget as budget_mod
from .budget import DEFAULT_RESERVE_MB, Budget, LeaseAccount

log = logging.getLogger("vramux.lease")

# Longest TTL a holder may ask for in one go. Renewal is unlimited; a single
# very long TTL is not, because the TTL is the only thing standing between a
# holder that dies badly and a permanently reserved card.
MAX_TTL = 3600.0
DEFAULT_TTL = 300.0

# How often expiry is checked. Short, because an expired lease holds real
# budget, and the sweep is a dictionary scan.
SWEEP_INTERVAL = 5.0

# How often the same timer records what the card looked like. The observer has
# always logged foreign usage and never stored it, so drift over time was
# unrecoverable; one timer, two jobs.
SAMPLE_INTERVAL = 300.0

# Cadence for re-testing the budget while a request waits for room.
POLL_INTERVAL = 2.0
WAIT_LOG_INTERVAL = 15.0

# Ancestry walks stop here. A pid chain on a healthy system is a handful of
# levels; a longer one means something is wrong and is not worth chasing.
_MAX_ANCESTRY_DEPTH = 32


class LeaseError(RuntimeError):
    """Base for everything the broker refuses to do."""


class InvalidRequest(LeaseError):
    """Malformed request — 400."""


class NotGrantable(LeaseError):
    """More than the card could ever provide, empty — 413."""


class NoRoom(LeaseError):
    """Not grantable within the caller's `wait` — 408."""


class UnknownLease(LeaseError):
    """No such lease; it was released, expired, or predates a restart — 404."""


class CardUnreadable(LeaseError):
    """The device cannot be read, so no honest answer exists — 503."""


def _iso(seconds_from_now: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now)).isoformat(
        timespec="seconds"
    )


def ppid_of(pid: int) -> Optional[int]:
    """Parent of `pid` from procfs, or None if it cannot be read.

    `/proc/<pid>/stat` field 4 is the ppid, but field 2 is the executable name
    in parentheses and may itself contain spaces and parentheses. Splitting
    after the *last* `)` is the only parse that survives a process called
    `(my program)`.
    """
    try:
        with open(f"/proc/{pid}/stat", "r") as fh:
            raw = fh.read()
    except (OSError, ValueError):
        return None
    tail = raw.rpartition(")")[2].split()
    if len(tail) < 2:
        return None
    try:
        return int(tail[1])
    except ValueError:
        return None


def descends_from(pid: int, roots, parent_of=ppid_of) -> bool:
    """Whether `pid` sits inside any of the process trees rooted at `roots`.

    The ordinary case for the CLI wrapper: it acquires the lease, then runs a
    command, so the process that allocates VRAM is its child or grandchild.
    Without this the holder's own memory reads as foreign and its grant gets
    charged on top of an allocation the device is already reporting.
    """
    seen = set()
    current = pid
    for _ in range(_MAX_ANCESTRY_DEPTH):
        if current in roots:
            return True
        if current <= 1 or current in seen:
            return False
        seen.add(current)
        parent = parent_of(current)
        if parent is None:
            return False
        current = parent
    return False


@dataclass
class Lease:
    """One grant, alive until released or expired."""

    id: str
    owner: str
    mb: int
    priority: int
    ttl: float
    pids: List[int] = field(default_factory=list)
    granted_at: float = 0.0
    expires_at: float = 0.0
    granted_iso: str = ""
    expires_iso: str = ""
    # What the holder was already holding when the grant was made. Kept for
    # reporting: a grant of 18 GB that charged 0 is not a bug, and the log
    # line should be able to say so.
    covered_mb: int = 0

    def expired(self, now: float) -> bool:
        return now >= self.expires_at

    def age(self, now: float) -> float:
        return max(0.0, now - self.granted_at)

    def to_json(self) -> dict:
        return {
            "lease": self.id,
            "owner": self.owner,
            "granted_mb": self.mb,
            "priority": self.priority,
            "ttl": self.ttl,
            "pids": list(self.pids),
            "granted_at": self.granted_iso,
            "expires_at": self.expires_iso,
            "covered_mb": self.covered_mb,
        }


class Broker:
    """Hands out memory, and takes it back when nobody says otherwise.

    `observer` supplies the card: anything with an async `snapshot()` and an
    async `sample()`. `loading` answers "is a model load in flight" — the
    arbiter's property, passed as a callable so the broker does not import
    residency and residency does not have to know leases exist.
    """

    def __init__(
        self,
        observer,
        reserve_mb: int = DEFAULT_RESERVE_MB,
        loading=None,
        max_ttl: float = MAX_TTL,
        sweep_interval: float = SWEEP_INTERVAL,
        sample_interval: float = SAMPLE_INTERVAL,
        poll_interval: float = POLL_INTERVAL,
        parent_of=ppid_of,
    ) -> None:
        self.observer = observer
        self.reserve_mb = reserve_mb
        self.max_ttl = max_ttl
        self.sweep_interval = sweep_interval
        self.sample_interval = sample_interval
        self.poll_interval = poll_interval
        self._loading = loading
        self._parent_of = parent_of
        self._leases: Dict[str, Lease] = {}
        self._lock = asyncio.Lock()
        self._timer: Optional[asyncio.Task] = None
        self._last_sample = 0.0
        # Why the last attempt did not grant, so a wait and a 408 can both say
        # something more useful than "busy".
        self._blocked_reason = "the card is busy"

    # ---- what is held ---------------------------------------------------------

    @property
    def leases(self) -> List[Lease]:
        return list(self._leases.values())

    def get(self, lease_id: str) -> Lease:
        lease = self._leases.get(lease_id)
        if lease is None:
            raise UnknownLease(f"no such lease: {lease_id}")
        return lease

    # ---- the budget -----------------------------------------------------------

    async def budget(self, snapshot=None) -> Budget:
        """Current budget, from a fresh reading unless one is supplied.

        Foreign usage moves without warning, so an admission decision reads the
        card rather than a cache — a stale read is exactly the case that OOMs.
        """
        if snapshot is None:
            snapshot = await self.observer.snapshot()
        if snapshot is None:
            raise CardUnreadable("the GPU cannot be read")
        return budget_mod.compute(snapshot, self._accounts(snapshot), self.reserve_mb)

    def _accounts(self, snapshot) -> List[LeaseAccount]:
        return [
            LeaseAccount(
                id=lease.id,
                owner=lease.owner,
                granted_mb=lease.mb,
                observed_mb=self._observed_for(snapshot, lease.pids),
            )
            for lease in self._leases.values()
        ]

    def _observed_for(self, snapshot, pids: Sequence[int]) -> int:
        return budget_mod.observed_mb(
            snapshot,
            pids,
            ancestor_of=lambda pid, roots: descends_from(pid, roots, self._parent_of),
        )

    # ---- acquire --------------------------------------------------------------

    async def acquire(
        self,
        mb: int,
        owner: str,
        ttl: float = DEFAULT_TTL,
        priority: int = 5,
        pids: Sequence[int] = (),
        wait: float = 0.0,
    ) -> Lease:
        """Grant `mb` to `owner`, waiting up to `wait` seconds for room.

        Raises `NotGrantable` at once when the request exceeds what the card
        could ever provide, and `NoRoom` when it merely does not fit yet —
        `413` versus `408`, which is the difference between a configuration
        error and a busy card.
        """
        mb = self._require_positive_int(mb, "mb")
        owner = (owner or "").strip()
        if not owner:
            raise InvalidRequest("owner is required")
        ttl = self._validated_ttl(ttl)
        wait = max(0.0, float(wait))
        pids = [self._require_positive_int(p, "pid") for p in pids]

        deadline = time.monotonic() + wait
        last_log = time.monotonic()
        while True:
            async with self._lock:
                lease = await self._try_grant(mb, owner, ttl, priority, pids)
                if lease is not None:
                    return lease
                blocked = self._blocked_reason
            now = time.monotonic()
            if now >= deadline:
                raise NoRoom(
                    f"{mb} MiB not available within {wait:.0f}s ({blocked})"
                )
            if now - last_log >= WAIT_LOG_INTERVAL:
                last_log = now
                log.info("lease for %s waiting: %s", owner, blocked)
            await asyncio.sleep(min(self.poll_interval, max(0.05, deadline - now)))

    async def _try_grant(
        self, mb: int, owner: str, ttl: float, priority: int, pids: Sequence[int]
    ) -> Optional[Lease]:
        """One admission attempt against a fresh reading. Caller holds the lock."""
        snapshot = await self.observer.snapshot()
        if snapshot is None:
            raise CardUnreadable("the GPU cannot be read")
        current = budget_mod.compute(snapshot, self._accounts(snapshot), self.reserve_mb)
        if mb > current.ceiling_mb:
            raise NotGrantable(
                f"{mb} MiB exceeds the {current.ceiling_mb} MiB this card can ever "
                f"grant ({current.total_mb} MiB total, {current.reserve_mb} MiB reserved)"
            )
        load = self._loading() if self._loading is not None else None
        if load is not None:
            # A load in flight has not allocated yet, so the card reads freer
            # than it is about to be. Waiting is the honest answer.
            self._blocked_reason = f"{load.get('tag', 'a model')} is loading"
            return None
        observed = self._observed_for(snapshot, pids)
        charge = budget_mod.charge_for(mb, observed)
        if not current.can_grant(charge):
            self._blocked_reason = (
                f"{charge} MiB needed, {current.free_mb} MiB free"
            )
            return None
        return self._record(mb, owner, ttl, priority, pids, observed, charge)

    def _record(
        self,
        mb: int,
        owner: str,
        ttl: float,
        priority: int,
        pids: Sequence[int],
        observed: int,
        charge: int,
    ) -> Lease:
        now = time.monotonic()
        lease = Lease(
            id="lse_" + secrets.token_hex(6),
            owner=owner,
            mb=mb,
            priority=priority,
            ttl=ttl,
            pids=list(pids),
            granted_at=now,
            expires_at=now + ttl,
            granted_iso=_iso(0),
            expires_iso=_iso(ttl),
            covered_mb=min(observed, mb),
        )
        self._leases[lease.id] = lease
        if charge < mb:
            log.info(
                "granted %s to %s (%s), %d MiB — %d MiB of it already on the card",
                lease.id, owner, f"ttl {ttl:.0f}s", mb, mb - charge,
            )
        else:
            log.info("granted %s to %s (ttl %.0fs), %d MiB", lease.id, owner, ttl, mb)
        return lease

    # ---- release, renew -------------------------------------------------------

    async def release(self, lease_id: str) -> Lease:
        async with self._lock:
            lease = self._leases.pop(lease_id, None)
        if lease is None:
            raise UnknownLease(f"no such lease: {lease_id}")
        log.info(
            "released %s (%s), %d MiB after %.0fs",
            lease.id, lease.owner, lease.mb, lease.age(time.monotonic()),
        )
        return lease

    async def renew(self, lease_id: str, ttl: Optional[float] = None) -> Lease:
        """Extend a lease by its TTL. Renewal is a heartbeat, not a courtesy."""
        async with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise UnknownLease(f"no such lease: {lease_id}")
            if ttl is not None:
                lease.ttl = self._validated_ttl(ttl)
            lease.expires_at = time.monotonic() + lease.ttl
            lease.expires_iso = _iso(lease.ttl)
            return lease

    # ---- expiry ---------------------------------------------------------------

    def start(self) -> None:
        """Start the sweep. Idempotent."""
        if self._timer is not None and not self._timer.done():
            return
        self._timer = asyncio.create_task(self._tick())

    async def close(self) -> None:
        if self._timer is None:
            return
        self._timer.cancel()
        try:
            await self._timer
        except asyncio.CancelledError:
            pass
        self._timer = None

    async def _tick(self) -> None:
        """One timer, two jobs: expire leases, and record what the card looks like."""
        while True:
            await asyncio.sleep(self.sweep_interval)
            try:
                await self.sweep()
            except Exception as exc:  # a sweep that raises must not end the timer
                log.warning("lease sweep failed: %s", exc)
            now = time.monotonic()
            if now - self._last_sample >= self.sample_interval:
                self._last_sample = now
                try:
                    await self.observer.sample()
                except Exception as exc:
                    log.debug("usage sample failed: %s", exc)

    async def sweep(self) -> List[Lease]:
        """Drop every lease whose holder stopped saying it was still there.

        Expiry is loud on purpose: a lease that expires under a live holder is
        a bug in that holder, and a quiet log line would hide it.

        Candidates are chosen outside the lock and re-checked inside it — a
        renewal can land in between, and dropping a freshly renewed lease would
        pull the card out from under a holder that did everything right.
        """
        candidates = [lease for lease in self.leases if lease.expired(time.monotonic())]
        if not candidates:
            return []
        expired: List[Lease] = []
        async with self._lock:
            now = time.monotonic()
            for lease in candidates:
                if self._leases.get(lease.id) is not lease:
                    continue
                if not lease.expired(now):  # renewed while we were deciding
                    continue
                del self._leases[lease.id]
                expired.append(lease)
                log.warning(
                    "lease %s (%s) expired holding %d MiB after %.0fs — its holder "
                    "did not renew or release",
                    lease.id, lease.owner, lease.mb, lease.age(now),
                )
        return expired

    # ---- validation -----------------------------------------------------------

    def _validated_ttl(self, ttl) -> float:
        try:
            ttl = float(ttl)
        except (TypeError, ValueError):
            raise InvalidRequest("ttl must be a number of seconds") from None
        if ttl <= 0:
            raise InvalidRequest("ttl must be positive — there is no infinite lease")
        if ttl > self.max_ttl:
            raise InvalidRequest(
                f"ttl {ttl:.0f}s exceeds the {self.max_ttl:.0f}s maximum — renew instead"
            )
        return ttl

    @staticmethod
    def _require_positive_int(value, name: str) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise InvalidRequest(f"{name} must be an integer") from None
        if number <= 0:
            raise InvalidRequest(f"{name} must be positive")
        return number


def self_pids() -> List[int]:
    """The calling process, as a lease's pid list.

    A wrapper's own pid is the right root even though the wrapper allocates
    nothing: the command it runs is inside its process tree, and attribution
    walks the tree.
    """
    return [os.getpid()]

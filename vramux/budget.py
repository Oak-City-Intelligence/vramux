"""How much of the card is available to hand out.

The whole module is pure arithmetic over one NVML reading and the current set
of grants. No clock, no lock, no I/O — because this is the number every later
stage trusts, and a number that can only be produced by a live GPU is a number
that never gets tested.

The formula is anchored on what the device *reports as used* rather than on a
sum of what vramux believes it handed out:

```
free = total - reserve - used - outstanding
```

`used` is ground truth: it already contains residents, leaseholders' real
allocations, foreign processes and the driver's own unattributed overhead.
`outstanding` is the part of a grant its holder has not allocated yet:

```
outstanding(lease) = max(0, granted_mb - observed_mb)
```

That subtraction is the whole defence against the trap in `DESIGN.md` §5.2. A
holder whose memory is already on the card is *observed*, so it is already in
`used`; charging its grant again on top would shrink the card by the size of
its own allocation. As the holder allocates, its outstanding share falls to
zero and nothing moves in the total. A holder that has allocated nothing yet is
charged in full, which is exactly the reservation it asked for.

`reserve` is not the unattributed overhead already on the card — that is inside
`used` and needs no help. It is headroom for the overhead a *new* allocation
creates and does not declare: a fresh CUDA context, compute buffers, the
allocator's own fragmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

# Headroom kept back from every grant. A new process brings its own CUDA
# context and compute buffers, and declares none of it. Measured unattributed
# overhead on the development card sat at 775-790 MiB with one model resident,
# which is the order this default is chosen against — it is a config value
# (`VRAMUX_RESERVE_MB`) precisely because it is a property of the card, not a
# constant of the design.
DEFAULT_RESERVE_MB = 1024


@dataclass(frozen=True)
class LeaseAccount:
    """One grant, with the memory its holder has actually been seen using.

    `observed_mb` is resolved by the caller, because deciding which NVML
    processes belong to a holder means looking at process ancestry, and that is
    not arithmetic.
    """

    id: str
    owner: str
    granted_mb: int
    observed_mb: int = 0

    @property
    def outstanding_mb(self) -> int:
        """The part of this grant that is promised but not yet allocated."""
        return max(0, self.granted_mb - self.observed_mb)


@dataclass(frozen=True)
class Budget:
    """What the card looks like to an admission decision."""

    total_mb: int
    reserve_mb: int
    used_mb: int
    recognised_mb: int
    foreign_mb: int
    unattributed_mb: int
    leases: Tuple[LeaseAccount, ...] = ()

    @property
    def outstanding_mb(self) -> int:
        return sum(lease.outstanding_mb for lease in self.leases)

    @property
    def granted_mb(self) -> int:
        return sum(lease.granted_mb for lease in self.leases)

    @property
    def ceiling_mb(self) -> int:
        """The most that could ever be granted, on a completely empty card.

        A request above this is a configuration error and must fail at once
        (`413`) rather than blocking for two minutes first.
        """
        return max(0, self.total_mb - self.reserve_mb)

    @property
    def free_mb(self) -> int:
        return max(0, self.ceiling_mb - self.used_mb - self.outstanding_mb)

    def can_grant(self, charge_mb: int) -> bool:
        return charge_mb <= self.free_mb

    def to_json(self) -> dict:
        return {
            "total_mb": self.total_mb,
            "reserve_mb": self.reserve_mb,
            "used_mb": self.used_mb,
            "recognised_mb": self.recognised_mb,
            "foreign_mb": self.foreign_mb,
            "unattributed_mb": self.unattributed_mb,
            "granted_mb": self.granted_mb,
            "outstanding_mb": self.outstanding_mb,
            "ceiling_mb": self.ceiling_mb,
            "free_mb": self.free_mb,
        }


def charge_for(requested_mb: int, observed_mb: int) -> int:
    """What a request actually costs the budget.

    Only the shortfall is new. A holder re-acquiring after a broker restart
    already has its memory on the card and is already counted as foreign; the
    lease *covers* that allocation rather than adding to it.
    """
    return max(0, requested_mb - observed_mb)


def compute(
    snapshot,
    leases: Sequence[LeaseAccount] = (),
    reserve_mb: int = DEFAULT_RESERVE_MB,
) -> Budget:
    """Build a `Budget` from an observer snapshot and the current grants."""
    device = snapshot.device
    return Budget(
        total_mb=device.total_mb,
        reserve_mb=reserve_mb,
        used_mb=device.used_mb,
        recognised_mb=snapshot.recognised_mb,
        foreign_mb=snapshot.foreign_mb,
        unattributed_mb=device.unattributed_mb,
        leases=tuple(leases),
    )


def observed_mb(snapshot, pids: Iterable[int], ancestor_of=None) -> int:
    """Memory on the card belonging to `pids`, or to their descendants.

    The descendant case is the ordinary one, not an exotic one: a wrapper
    acquires a lease and then runs a command, so the process that allocates is
    a child — often a grandchild. Attributing only the exact pid would leave
    the holder's own memory reading as foreign, and then its grant would be
    charged on top of an allocation already inside `used`.

    `ancestor_of(pid, roots)` answers "is this pid inside one of these process
    trees"; it is injected so this is testable without a real process tree.
    """
    roots = {p for p in pids if p is not None}
    if not roots:
        return 0
    total = 0
    for process in snapshot.device.processes:
        if process.pid in roots:
            total += process.used_mb
        elif ancestor_of is not None and ancestor_of(process.pid, roots):
            total += process.used_mb
    return total

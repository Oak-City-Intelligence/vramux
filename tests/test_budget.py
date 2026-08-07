"""Budget arithmetic.

Pure functions over a fake reading, so every case that matters — including the
ones a real card only produces once a month — is a two-line test.
"""

from __future__ import annotations

from vramux import budget
from vramux.budget import Budget, LeaseAccount
from vramux.nvml import DeviceState, GpuProcess
from vramux.observer import Attribution, Snapshot


def snapshot(used=1000, total=24564, procs=(), owners=()):
    """A reading with `procs` on the card; `owners` names the recognised ones."""
    processes = list(procs)
    owned = dict(owners)
    device = DeviceState(
        index=0, name="Test GPU", total_mb=total, used_mb=used,
        free_mb=total - used, processes=processes,
    )
    return Snapshot(
        device=device,
        attributions=[Attribution(process=p, owner=owned.get(p.pid)) for p in processes],
    )


def test_free_is_the_card_minus_reserve_and_what_is_on_it():
    snap = snapshot(used=8000, total=24564)
    view = budget.compute(snap, reserve_mb=1024)
    assert view.ceiling_mb == 24564 - 1024
    assert view.free_mb == 24564 - 1024 - 8000


def test_an_unallocated_grant_is_charged_in_full():
    """A reservation is a promise about memory nobody has taken yet."""
    snap = snapshot(used=1000, total=24564)
    view = budget.compute(
        snap, [LeaseAccount("lse_a", "batch", granted_mb=18000, observed_mb=0)],
        reserve_mb=1024,
    )
    assert view.outstanding_mb == 18000
    assert view.free_mb == 24564 - 1024 - 1000 - 18000


def test_a_grant_stops_being_charged_as_its_holder_allocates():
    """The holder's memory moves from promised to present. It must not be
    counted in both places, or the card appears to shrink as work starts."""
    before = budget.compute(
        snapshot(used=1000), [LeaseAccount("lse_a", "batch", 18000, 0)], reserve_mb=1024
    )
    half = budget.compute(
        snapshot(used=10000, procs=[GpuProcess(4127, 9000, "worker")]),
        [LeaseAccount("lse_a", "batch", 18000, 9000)], reserve_mb=1024,
    )
    full = budget.compute(
        snapshot(used=19000, procs=[GpuProcess(4127, 18000, "worker")]),
        [LeaseAccount("lse_a", "batch", 18000, 18000)], reserve_mb=1024,
    )
    # The 1000 MiB of unrelated usage is constant, so free memory should be
    # constant too across the holder's entire allocation.
    assert before.free_mb == 24564 - 1024 - 1000 - 18000
    assert half.free_mb == 24564 - 1024 - 10000 - 9000
    assert full.free_mb == 24564 - 1024 - 19000 - 0
    assert before.free_mb == half.free_mb == full.free_mb


def test_a_holder_over_its_grant_is_never_credited():
    """Allocating more than was granted does not hand budget back."""
    view = budget.compute(
        snapshot(used=20000), [LeaseAccount("lse_a", "batch", 18000, 20000)],
        reserve_mb=1024,
    )
    assert view.outstanding_mb == 0
    assert view.free_mb == 24564 - 1024 - 20000


def test_charge_is_only_the_shortfall():
    """`DESIGN.md` §5.2: re-acquiring after a restart charges nothing new."""
    assert budget.charge_for(18000, observed_mb=18000) == 0
    assert budget.charge_for(18000, observed_mb=12000) == 6000
    assert budget.charge_for(18000, observed_mb=0) == 18000
    assert budget.charge_for(8000, observed_mb=20000) == 0


def test_unattributed_overhead_is_already_subtracted():
    """Driver overhead belongs to no process but is real memory. Anchoring on
    `used_mb` counts it without anyone having to remember to."""
    snap = snapshot(used=7800, procs=[GpuProcess(1, 6588, "llama-server")])
    view = budget.compute(snap, reserve_mb=1024)
    assert view.unattributed_mb == 7800 - 6588
    assert view.free_mb == 24564 - 1024 - 7800


def test_free_never_goes_negative():
    view = budget.compute(snapshot(used=24000), [LeaseAccount("a", "x", 8000, 0)],
                          reserve_mb=1024)
    assert view.free_mb == 0


def test_ceiling_is_what_413_is_measured_against():
    view = budget.compute(snapshot(used=20000), reserve_mb=1024)
    assert view.ceiling_mb == 23540
    assert not view.can_grant(23540)      # not right now
    assert 24000 > view.ceiling_mb        # and never


def test_observed_memory_counts_the_holders_own_pids():
    snap = snapshot(procs=[GpuProcess(10, 500, "a"), GpuProcess(11, 300, "b")])
    assert budget.observed_mb(snap, [10]) == 500
    assert budget.observed_mb(snap, [10, 11]) == 800
    assert budget.observed_mb(snap, []) == 0
    assert budget.observed_mb(snap, [99]) == 0


def test_observed_memory_follows_a_holders_children():
    """The wrapper acquires, then runs a command: the process that allocates is
    a descendant, not the pid on the lease."""
    snap = snapshot(procs=[GpuProcess(4130, 18000, "worker")])
    tree = {4130: 4129, 4129: 4127}

    def ancestor_of(pid, roots):
        while pid in tree:
            pid = tree[pid]
            if pid in roots:
                return True
        return False

    assert budget.observed_mb(snap, [4127], ancestor_of=ancestor_of) == 18000
    assert budget.observed_mb(snap, [9999], ancestor_of=ancestor_of) == 0


def test_budget_json_carries_the_arithmetic_not_just_the_answer():
    view = Budget(
        total_mb=24564, reserve_mb=1024, used_mb=8000, recognised_mb=6000,
        foreign_mb=2000, unattributed_mb=0,
        leases=(LeaseAccount("a", "x", 4000, 1000),),
    )
    payload = view.to_json()
    assert payload["free_mb"] == view.free_mb
    assert payload["granted_mb"] == 4000 and payload["outstanding_mb"] == 3000
    assert payload["reserve_mb"] == 1024

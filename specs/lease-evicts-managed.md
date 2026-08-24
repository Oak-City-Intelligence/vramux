# Spec: lease admission evicts managed residents

Status: BUILT 2026-08-24 (ROADMAP Stage 8). One deviation from the text
below: `make_room_for_lease` returns the eviction *count*, not megabytes —
an unmeasured victim frees an amount only the card can report, and the count
is the honest answer to the caller's actual question (did anything change,
i.e. is a re-grant attempt worth making before asking anyone to yield).
Companion: `specs/serving-queue.md` — the two are one priority rule read from
both directions, and should land in this order (this one first; it is smaller
and the other one's tests want it in place).

## Problem

A lease request that does not fit today has exactly one lever: ask *other
leaseholders* below its priority to yield (`Broker.acquire` →
`request_yield`). It never touches managed residents, so a 20 GB language
model sitting idle blocks an 18 GB generation lease until the model's
15-minute idle timeout fires or a client hand-calls `POST /gpu/evict` first.

Every serious lease client on this machine has learned the workaround:
the image stack's VRAM helper unloads the model itself before it waits. That is the
pre-vramux poll loop wearing a vramux hat — the client is doing residency's
job because the broker won't route the request to the one component that can.

`DESIGN.md` §3 already grants the permission this spec uses: managed
consumers are the tier vramux may **evict silently**. They pay a reload, they
are never corrupted, and they never even learn it happened. Leases were built
after that table and simply never got wired to it.

## Behavior

A lease acquire that is short on budget may evict managed residents to cover
the shortfall, under one rule:

> A lease may evict a resident whose effective priority is **strictly below**
> the lease's priority.

A resident's effective priority is `ModelSpec.priority` when set, else
`serving_priority` (default 7). Leases default to 5. Consequences, in the
order they matter:

- **A default lease still cannot evict anything.** 5 < 7. Every existing
  client behaves exactly as it does today. The feature is opt-in per request,
  by asking for a priority above serving.
- **A generation lease at priority 8 evicts the idle chat model** instead of
  waiting out its idle timer. This is the case the spec exists for.
- The rule is the mirror image of `_ask_leases_to_yield`, where serving asks
  leases *below its* priority to yield. After this spec, priority means one
  thing in both directions: the higher number gets the card; the lower number
  is evicted (managed), asked (leaseholder), or waits (see companion spec).

What eviction means here is exactly what it means when serving does it:
`ResidencyArbiter._evict` — wait for the resident's in-flight requests to
drain (`Resident.drained`, bounded by `drain_timeout`), then stop the
backend. A streaming response is never cut mid-token. Nothing new is built at
the eviction layer.

### What this does not do

- Never evicts a resident at or above the lease's priority. Pin a model
  against eviction with `priority:` in `models.yml`.
- Never touches leaseholders (that is yield, unchanged) or foreign memory
  (tier 3 is observe-only, forever).
- Never evicts a model that is mid-load. `arbiter.loading` already answers
  this; a load in flight is waited out, not raced.
- A `wait: 0` acquire does not evict. Eviction takes seconds (drain +
  process exit + the card actually freeing); a caller who declared it will
  not wait cannot be given a benefit that only arrives by waiting. Same
  contract as yield, which also only runs under a deadline.

## Mechanics

### Wiring

The broker does not import residency and must not start. The injection
pattern already exists in `__main__.py` and gets its fourth line:

```python
arbiter.use_budget(broker.budget)
arbiter.use_yield(broker.request_yield)
broker.use_make_room(arbiter.make_room_for_lease)   # new
```

`Broker.use_make_room(callable)` stores it; `None` (the default, and the
state in any embedding that never calls it) means the feature is off and
`acquire` behaves exactly as today. Tests that construct a bare `Broker`
change nothing.

### Broker side

`Broker.acquire`'s wait loop already computes the shortfall
(`_blocked_short_mb`) and already fires yield once per acquire, not once per
poll. Eviction slots into the same place with the same once-per-acquire
discipline, and runs **before** yield:

```
while not granted:
    if not asked_to_evict and short > 0 and now < deadline:
        asked_to_evict = True
        await self._make_room(short, owner, priority)     # evict managed, bounded
    if not asked_to_yield and still short and now < deadline:
        asked_to_yield = True
        await self.request_yield(short, ...)              # then ask leaseholders
    ...poll until deadline...
```

Evict-before-yield is deliberate: eviction is free to its victim in every
sense that matters (a reload later), while yield interrupts somebody's
running work. The cheap disruption goes first, and often makes the expensive
ask unnecessary.

`_make_room` awaits the injected callable with the shortfall, the owner
string (for the log), and the lease priority. It treats any exception as "no
room made" and logs it — a broken residency layer must degrade to today's
behavior, never take `acquire` down.

### Residency side

New method on `ResidencyArbiter`:

```python
async def make_room_for_lease(self, mb: int, by: str, priority: int) -> int:
    """Evict managed residents below `priority` until `mb` MiB is freed
    or no eligible resident remains. Returns the number evicted."""
```

- Takes the arbiter lock (`_admit`), so it cannot race a load or another
  eviction, and a serving request arriving mid-eviction queues behind it
  exactly as it queues behind a swap today.
- Victim order: cheapest-first among eligible residents — the same reasoning
  `request_yield` documents. Evicting a 2 GB embedder must be tried before a
  20 GB chat model when 2 GB covers the shortfall.
- Stops as soon as the freed cost covers `mb`. Uses measured cost
  (`known_cost_mb`) for the running total, and re-reads the budget at the end
  rather than trusting arithmetic — the card is the truth.
- Logs loudly, once per victim:
  `evicting qwen3.8:27b (19 442 MiB) for lease owner=image-stack priority=8 — resident priority 7`.

### Accounting

Nothing changes. Eviction returns memory to the same pool `_try_grant` reads;
the next poll of the acquire loop sees it. The evicted model's measured cost
stays in `costs.json`, so the reload after the lease releases is admitted on
a measurement, not an estimate.

## Config

None. The priority gate is the whole safety story, and it defaults closed
(lease 5 < serving 7). Adding a kill-switch env var would be a second way to
say what `priority:` in the request and `priority:` in `models.yml` already
say between them.

## Tests

1. Default-priority lease beside a resident, short on room: **no eviction**,
   behavior identical to today (waits, then `NoRoom`).
2. Priority-8 lease, resident at serving priority 7, shortfall: resident is
   drained then evicted, lease granted within `wait`.
3. Resident pinned at `priority: 9`: priority-8 lease does not evict it,
   falls through to yield/timeout as today.
4. Two residents (2 GB and 19 GB), shortfall 1.5 GB: only the 2 GB resident
   is evicted.
5. Resident with an in-flight streaming request: eviction defers until
   drained; forced after `drain_timeout`, matching serving-side eviction.
6. `wait: 0` with a shortfall: immediate `NoRoom`, no eviction.
7. `use_make_room` never called (bare Broker): all lease tests from Stage 4
   pass unmodified.
8. Callable raises: acquire degrades to today's wait-then-`NoRoom`, error
   logged.

## Docs

- `README.md` lease rules gain one bullet: a lease above serving priority
  may evict idle-or-drained managed models; below it, never.
- `DESIGN.md` §5 lease protocol notes the rule and points at the §3 tier
  table it is derived from.
- `ROADMAP.md` gets the stage entry; this also part-answers the standing
  "Open" item *whether anything should enforce priority* — for managed
  residents the answer is yes, because evicting its own children silently is
  the one power vramux has always claimed.

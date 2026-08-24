# Spec: serving queues behind a higher-priority lease

Status: spec. Not built.
Companion: `specs/lease-evicts-managed.md` — one priority rule, two
directions. Build that one first.

## Problem

A model load that cannot fit asks leaseholders **below its priority** to
yield, waits `yield_wait` (default 30 s), then raises `InsufficientVRAM`.
The router turns that into a 503 with `Retry-After: 10`
(`router._still_loading`).

That is the right shape when the blocker is a peer that was asked and
declined. It is the wrong shape when the blocker is a lease that **outranks
serving** — which, once the companion spec lands, is the normal state of the
card during a generation run: an agent's pipeline holds an 18 GB lease at
priority 8 precisely so that serving (priority 7) cannot pull VRAM out from
under a running FLUX job.

The operator arranged that ranking on purpose. A chat request arriving
mid-generation is not hitting an error; it is *outranked*, and the correct
behavior is to wait its turn. Today it gets a hard 503, which most
OpenAI-compatible clients — an agent harness above all — surface as a failed
turn. The one client this machine runs all day cannot survive the one state
this machine is designed to be in.

Two facts make waiting safe to promise:

1. **Every lease has a TTL and there is no infinite lease** (`DESIGN.md`
   §5). The blocker *will* end, and the broker knows the latest moment it
   can: `expires_at`, extended only by explicit renewal.
2. The 503-with-`Retry-After` path stays fully intact for every other cause.
   This spec narrows one branch; it deletes nothing.

## Behavior

When admission fails **because of leases the requesting model does not
outrank** — and only then — the load request parks instead of raising:

> An outranked serving request waits, up to `VRAMUX_QUEUE_WAIT` seconds, for
> the blocking lease(s) to release or expire. The caller sees a slow first
> token, not an error.

Every other `InsufficientVRAM` cause is unchanged: foreign memory, a
same-or-lower-priority lease that was asked to yield and did not, a cost that
never fits. Those still fail after `yield_wait` exactly as today, because for
them there is no bounded event to wait for.

Precisely: after `_make_room_for` has evicted what it can, the outranked
branch is taken when the shortfall is covered by
`sum(outstanding + held of leases with priority >= requesting priority)` —
i.e. releasing the outrankers would make the load fit. If it would not
(foreign grew, cost too big), fail now; queueing cannot help.

### The wait itself

- Deadline: `min(now + VRAMUX_QUEUE_WAIT, latest expires_at among blocking
  leases + grace)` — no request ever waits past the moment the last blocker
  must either renew or die. A renewal that pushes `expires_at` out extends
  the horizon, but never past `VRAMUX_QUEUE_WAIT` total.
- Wakeup: poll the budget on the lease sweep cadence (5 s). Lease release
  already triggers a sweep; a subscription mechanism is not worth its
  machinery at this timescale.
- On room appearing: proceed into the normal load path. The measured-cost
  admission check runs again from scratch — the world moved while we slept.
- On deadline: raise `InsufficientVRAM` with a message that names the
  blocker: `qwen3.8:27b needs 19442 MiB; lease lse_04c1 (image-stack,
  priority 8, ttl 240s) holds the card and outranks serving — gave up after
  600s`. The router's existing 503 path carries it out.

### Honesty while parked

A silent socket is indistinguishable from a wedge — the same argument
`_admit`'s docstring already makes for slow loads, so the same remedies:

- The parked load registers as the in-flight load (`_loading`) with a state
  the wait message can render: `_wait_message` learns a third case,
  `"qwen3.8:27b is queued behind lease lse_04c1 (image-stack), up to Ns"`.
  A second chat request arriving during the park then queues on the arbiter
  lock and — if it times out — gets *that* message in its 503, which is the
  truthful one.
- `GET /gpu/state` grows an optional `queued` block: model tag, blocking
  lease ids, queued-since. The console renders it — a card that looks idle
  while requests stack up behind a lease is exactly the "where did my memory
  go" moment the console exists for.
- Log on a cadence (`_report_wait` pattern, reuse it): once per interval,
  `queued: qwen3.8:27b behind the image stack (lease lse_04c1, 183s of ttl
  left)`.

### Lock discipline

The parked request holds the arbiter lock, deliberately. Concurrent chat
requests for *any* model cannot be served anyway — the card is spoken for —
and letting them stack on `_admit` gives them the honest wait message and a
bounded timeout for free, instead of each independently discovering the
lease. `_admit`'s bound for waiters must account for the queue: `bound =
queue remaining + _WAIT_SLACK` when the in-flight "load" is a park.

One necessary exception: `keep_alive: 0` / evict and all `/gpu/*` state
endpoints stay lock-free as they are today. An operator watching a parked
card must be able to see it and free it.

## Config

| Env var | Default | Meaning |
|---|---|---|
| `VRAMUX_QUEUE_WAIT` | `600` | max seconds an outranked load request parks; `0` restores today's fail-fast |

Default on. The failure mode of `0` (agent turn dies mid-generation) is
strictly worse than the failure mode of `600` (a caller that wanted to fail
fast waits — and a caller with a shorter patience already enforces it with
its own client timeout, which closing the socket honors: a parked request
whose client disconnects abandons the park and releases the lock).

`yield_wait` is untouched and still governs the ask-to-yield phase; the park
begins only after that phase concludes without room.

## Interaction with the companion spec

Together the two specs make priority a total story:

| requester | blocker | outcome |
|---|---|---|
| lease p8 | resident p7 | resident evicted (companion spec) |
| serving p7 | lease p8 | serving parks (this spec) |
| serving p7 | lease p5 | lease asked to yield, then fail (today, unchanged) |
| lease p5 | resident p7 | wait then `NoRoom` (today, unchanged) |

Nothing anywhere revokes or takes. The ROADMAP "Open" item — *whether
anything should enforce priority* — closes as: enforcement is eviction for
vramux's own children, and patience for everyone else.

## Tests

1. Load blocked by a priority-8 lease: parks; lease released at t+20s; load
   proceeds; client got its response on one connection, no 503.
2. Same, lease expires instead of releasing: park ends on the expiry sweep,
   load proceeds.
3. Park exceeds `VRAMUX_QUEUE_WAIT`: 503, message names lease, owner and
   remaining ttl.
4. Blocker is priority 5 (does not outrank): today's behavior byte-for-byte
   — yield asked, `InsufficientVRAM` after `yield_wait`, no park.
5. Blocker outranks but releasing it would still not cover the shortfall
   (foreign grew): immediate fail, no park.
6. Second request during a park: waits on the lock, its timeout message
   names the queue, not a phantom slow load.
7. Client disconnect mid-park: lock released, next waiter proceeds to park
   with the remaining horizon.
8. `VRAMUX_QUEUE_WAIT=0`: today's behavior across all of the above.
9. Renewal during park extends the expiry horizon but total park still caps
   at `VRAMUX_QUEUE_WAIT`.

## Docs

- `README.md` env table gains `VRAMUX_QUEUE_WAIT`; the lease section gains a
  paragraph: "a lease above serving priority makes serving wait, which is
  the point — take one when the card must not be disturbed."
- `DESIGN.md` §6 gains the four-row table above.
- `ROADMAP.md` stage entry; closes the priority "Open" item jointly with the
  companion spec.

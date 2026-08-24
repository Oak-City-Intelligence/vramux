# Roadmap

How to get from a single-slot model router to a real VRAM broker, without
breaking the machine it runs on along the way.

Companion to `DESIGN.md` (what vramux is). This is the sequence.

---

## The diagnosis

A unifying control console for the machine's GPU tools was attempted once, and
abandoned. The individual launchers survived it; the layer that was supposed to
unify them did not.

It did not fail for lack of ability. It failed because **it was a launcher layer
with nothing underneath it.** Each tool knows how to start its own workload.
None of them can know whether starting it is *safe*, because no component owns
that question. Three separate tools have since grown their own VRAM-polling
workaround — the same hole, discovered independently three times.

So the sequence below does not rebuild the console. It builds the floor, then
lets the existing tools stand on it. Those tools are the client set, not the
thing being replaced.

## Principles

These constrain every stage. They are why the ordering looks conservative.

1. **The daily driver never breaks.** The router on :11434 is load-bearing —
   every local-LLM client on the machine depends on it. Every stage ends with a
   working router or it is not finished.
2. **Each stage is independently valuable.** No stage exists only to enable the
   next one. If work stops after any stage, what shipped was worth shipping.
3. **Each stage is reversible.** Until clients migrate, vramux's new behaviour
   is additive. Reverting means turning something off, not unwinding a rewrite.
4. **Observe before controlling.** Every mechanism that will eventually make
   decisions first ships in a mode where it only watches and logs. The decisions
   turn on later, against real data.
5. **Behaviour changes and structure changes never land together.** A commit
   either moves code or changes what it does. Never both — that is the only way
   a regression stays diagnosable.

## Current state (2026-08-07)

**Stages 0 to 4 are done.** Stage 5, migrating clients, is next — and most of
it is work in other repos.

- ~3,400 lines across twelve modules; 171 tests, all GPU-less
- the broker grants: leases with mandatory TTL, server-side expiry, reclaim by
  process tree, and `vramux lease -- <cmd>` holding one for a command's lifetime
- nothing requires a grant to be served: unmigrated consumers are foreign,
  which is a correct state
- residency-shaped arbiter with per-resident in-flight counting; admission is
  pinned at one, so behaviour is still one model at a time
- swap verified working both directions against the live service after the
  refactor (24 s cold container, 6 s container→GGUF)
- under git at its final location, serving `:11434` from there as
  `vramux.service` (the old unit name kept as an alias)
- no path resolves to this machine; a clean checkout with an empty environment
  starts and serves
- `VRAMUX_*` env prefix; the old `MYLLAMA_*` names shim with a one-time warning
- the GPU is free again — Stage 6 is unblocked whenever the Stage 2 dataset
  justifies starting it

---

## Stage 0 — Safety net — DONE (2026-08-07)

**Do this first, and it needs no GPU.**

Every later stage edits load-bearing infrastructure. Right now there is no way
to know whether an edit broke something except noticing that a client stopped
working, which is both slow and infuriating.

- SSE→wire-format translation: fragmented tool-call arguments, missing finish
  chunk, parallel calls, malformed argument JSON, plain content, thinking/content
  split. These are the exact cases that regressed last session and were verified
  with throwaway inline asserts.
- Registry loading: both backend kinds, aliasing, dedup, YAML-over-scan
  precedence.
- Swap and drain logic against a fake backend — no GPU, no subprocess.

`pytest` + `aiohttp.test_utils`, entirely GPU-less, which is also what makes it
CI-able later.

**Exit:** a red test can be produced by reverting any of last session's fixes.
That is the real acceptance criterion, not coverage percentage.

Met. Each of the five fixes was reverted in turn and the suite went red each
time: the `[DONE]` guard (7 failures), the tool-call accumulator (6), the
health recycle (1), the drain wait (1), and `reconcile()`'s docker filter (1).

## Stage 1 — Move and identity — DONE (2026-08-07)

### 1a. Move, do not fork

The repo relocates to its final location **now**, early, and development
continues there against the live service.

The tempting alternative is to copy the code somewhere clean, build vramux
there while the old router keeps serving :11434, and swap at the end. Do not do
this. It produces two codebases, and:

- **Nothing is dogfooded.** The version under development is not the version
  running, so bugs are found by inspection instead of by use.
- **Fixes diverge.** Anything repaired in the live router must be re-applied to
  the fork or silently lost.
- **All risk concentrates at cutover** — every bug from every stage arriving at
  once, on the day load-bearing infrastructure is swapped.
- **The machine gets nothing until the end.** Stages 2, 4 and 5 each pay off
  immediately; a fork defers all of it.

The thing that motivates forking — wanting a clean room for the scrub — is not
worth it. The scrub is five hardcoded paths and an env prefix.

```
mv <old-checkout> <final-location>
git init                      # not a repo today
# repoint the user unit at the new path, restart, verify :11434
```

It lives at its final address from day one and is the thing actually running the
whole time. There is never a moment where it gets "turned around and installed" —
it was never anywhere else.


### 1b. Identity

Mechanical, low risk, and it stops the old name from spreading into new code.

- `MYLLAMA_*` → `VRAMUX_*`, with a shim that reads the old names and warns.
- `the old unit` → `vramux.service`.
- Strip absolute machine-specific defaults: resolve the inference binary from
  `$PATH`, model dir from env or `./models`, blob roots only if they exist.
- `ctx_overrides` moves out of `registry.py` into shipped example config — it is
  one machine's configuration wearing a module as a disguise.
- `models.yml` stays local and gitignored; `models.example.yml` ships.

Port 11434 does not change, so every client is unaffected. The unit file is the
only disruption, and it is a single restart.

**On publishing.** It is not a driver of any decision in this roadmap, but the
destination is claimed at 1a and being publish-shaped — no absolute paths, real
tests, honest docs — is hygiene that makes the machine's own infrastructure
better. It comes as a side effect, not a goal, and there is no deadline on it.

**Exit:** clean checkout runs on a machine that is not this one. No test asserts
a path under `/home`.

Met. A fresh clone started under `env -i` with a foreign `HOME`, no config and
no model directory: it came up and served `/api/tags` as an empty list rather
than failing. No tracked file contains an absolute path to the machine it was
written on.

## Stage 2 — The observer — DONE (2026-08-07)

**The highest-leverage stage, and it changes no behaviour at all.**

Add NVML accounting that only watches: total, used, free, per-process attribution,
foreign versus recognised. Log it. Sample it around every managed load and unload.
Write what it sees to the cost cache. Decide nothing.

Two reasons this comes before anything that uses it:

- **Zero risk.** A read-only observer cannot break the router. It can ship the
  same day it is written.
- **It builds the dataset the hard parts need.** By the time admission control
  needs to know what a model actually costs at a given context length, there are
  weeks of real measurements instead of a formula and a prayer. The cost model is
  the single largest OOM risk in `DESIGN.md`; this is how that risk gets retired
  before it is load-bearing.

A foreign trainer holding a large slice of the card is exactly the tier-3 case
that has to work, and it is the observer's most useful first subject.

**Exit:** `vramux state` prints a true picture of the card, including processes
vramux has never heard of. Cost cache has real entries for every model that has
been loaded since the observer landed.

Met, and it settled an open question early. Container PID attribution — the
assumption flagged in `DESIGN.md` §13 and deferred to Stage 5 — was tested here
because implementing attribution required it: `docker compose top` reports host
PIDs, NVML reports host PIDs for container compute processes, and they match.
Both backend kinds are now attributed and measured. The first parser read the
wrong column, which is worth remembering: a fixed column index silently
collects a plausible wrong number, so the header decides.

## Stage 3 — Structure

Pure refactor. Tests from Stage 0 are what make it safe, and they must stay green
throughout without being edited to fit.

- `Backend` `Protocol` pinned, with `adopt()` replacing the
  `backend._started = True` hack in `reconcile()`.
- `LlamaServerSupervisor` → residency-shaped arbiter. Not `SlotSupervisor` — the
  whole point is that "slot" stops being singular.
- In-flight counting becomes **per-resident**. This is the largest structural
  change: evicting model A must not wait on requests in flight against model B.
- Translation helpers split out of `router.py` into their own module.
- Bounded fast-fail on slow container loads, so a legitimate 600 s wait reports
  as a wait instead of reading like a hang.

Admission stays tuned to exactly one resident. Behaviour is identical to today.
The structure that permits multi-residency exists; the budget stays shut.

**Exit:** tests green, unedited. `git diff` shows no change in observable
behaviour. A third backend kind could be added without touching the arbiter.

### Done, 2026-08-07

`supervisor.py` became three files: `backends.py` (the pinned `Backend`
protocol and both kinds), `residency.py` (`Resident` + `ResidencyArbiter`), and
`translate.py` (the wire-format functions lifted out of `router.py`, which
dropped from 617 lines to 392). `healthy()` moved onto the protocol — the
decision the stage was required to make and not defer, recorded with its
reasoning in `DESIGN.md` §7.1, along with why `vram_hint()` was left off.

All 119 tests stayed green with **every assertion unchanged**. What did change:
imports, and the test doubles that implement the very interface this stage
pinned — a fake backend has to grow `healthy()` and `adopt()` when the contract
grows them. That is the one place the "unedited" rule could not hold literally,
and it holds where it matters: no test was weakened or retuned to pass.

Verified against the live service, not just the fakes — swap both directions
with measurement and unload accounting, the streaming tool call (exactly one
terminal line, carrying `tool_calls`), container reconcile through the new
`adopt()`, and wedge recovery via `docker pause`, which is what proves
`healthy()` still works from its new home.

## Stage 4 — Leases

Additive. The broker starts granting; nothing yet requires a grant.

- Lease API: acquire, release, renew, mandatory TTL, `413` versus `408`.
- CLI wrapper — `vramux lease --mb N --owner X -- <cmd>` — releasing on exit
  including crash. This is the whole adoption story: the correct version has to
  be shorter than the broken version or nothing will migrate.
- Reclaim by PID, so a re-acquire after broker restart does not double-count
  against memory already observed as foreign (`DESIGN.md` §5.2).
- Leases dropped on broker restart; holders demote to foreign, budget stays true.

Unmigrated consumers keep working untouched. They are foreign, which is a correct
state, not a broken one.

**Exit:** a lease can be taken and released by hand, survives a broker restart
without corrupting the budget, and expires loudly when its holder dies.

### Done, 2026-08-07

Two new modules, kept flat rather than split into the `core/ lease/` directories
`DESIGN.md` §8 eventually wants: `budget.py` is pure arithmetic over one reading
and the current grants, `lease.py` is the broker — grants, expiry, and the pid
attribution behind reclaim.

The decision worth recording is **what the budget is anchored on**. Summing what
vramux believes it handed out would have needed a separate rule for residents,
for leaseholders, for foreign processes and for driver overhead, and any one of
them wrong is an OOM. Instead:

```
free = total - reserve - used - outstanding
```

`used` is what the device reports, which already contains all four populations.
`outstanding` is the part of each grant its holder has not allocated yet,
`max(0, granted - observed)`. Reclaim then needs no special case at all: a
holder whose memory is already on the card is already inside `used`, so its
grant charges nothing new, and as it allocates, the same memory simply moves
from promised to present. That was verified against the live card — a 6000 MiB
grant naming a resident already holding 6588 MiB moved free memory by zero.

Three things the stage learned that were not in the plan:

- **Attribution has to follow the process tree.** The wrapper acquires the
  lease and then runs a command, so the process that allocates is a child or a
  grandchild. Matching the exact pid would have left the holder's own memory
  reading as foreign — which is the double-count the stage exists to prevent,
  arriving through the front door instead of after a restart.
- **Nothing may be granted while a model is loading.** A load in flight has not
  allocated yet, so the card reads freer than it is about to be. The request
  waits and gets an honest `408`.
- **`reserve` is not the unattributed overhead already on the card.** That is
  inside `used` and needs no help. The reserve covers what a *new* allocation
  creates and never declares: its own CUDA context and compute buffers. It is
  `VRAMUX_RESERVE_MB`, defaulting to 1024.

The CLI is stdlib-only, deliberately — a client that needs a virtualenv to
release a lease will not be installed on the machine that needs it. Expiry was
built and tested before the wrapper, because a wrapper killed with `SIGKILL`
runs no cleanup and server-side expiry is the only thing that makes the
guarantee real.

The stage also closed the standing gap from Stage 2: the sweep timer's second
job records the card to `~/.cache/vramux/usage.jsonl`, so foreign drift over
time is readable rather than merely logged.

52 new tests, 171 total, still GPU-less and ~0.9 s. Verified against the live
service: `413` immediately versus `408` after a wait, expiry logging loudly
under a dead holder, reclaim against a real resident, grants refused during a
cold container load, and the wrapper propagating its child's exit code.

## Stage 5 — Clients

One at a time, most pain first. Each migration is a separate, revertible change,
and each one is the first time a real person gets a real benefit.

| order | client | what changes | why this order |
|---|---|---|---|
| 1 | batch pipeline | poll loop → `vramux lease -- ./stage2.sh` | worst hack, clearest win, easy rollback |
| 2 | `vram-free` helper | deleted or reduced to `vramux free` | proves the primitive replaces the workaround |
| 3 | image stack | leaseholder; yields via its unload endpoint | first cooperative yielder, first container PID attribution test |
| 4 | captioning tool | drop the GPU advisor, ask the broker | removes the third reimplementation |

Stage 5 is where the untested assumption in `DESIGN.md` §13 gets tested — whether
NVML attributes container processes to host PIDs the way reclaim needs. Client 3
is deliberately the one that proves it, with clients 1 and 2 already delivering
value if it turns out to be harder than expected.

**Exit:** no project on the machine reimplements VRAM reasoning. The
`vram-free` helper and the GPU advisor are gone.

### In progress, 2026-08-07

All four clients now ask vramux instead of reading the device themselves. Two
of them could take a commit; the rest live in trees that are not under version
control, which is recorded here because that work is otherwise invisible.

| client | was | now |
|---|---|---|
| batch pipeline | polled its own toolbox for free VRAM against a hardcoded floor | gates on the smaller of that reading and vramux's budget |
| video pipeline | `keep_alive:0`, then 30x2s of `nvidia-smi` | `vramux evict` (which drains first) + `vramux free --wait` |
| the `vram-free` helper | `nvidia-smi` free + `keep_alive:0` + a 2 s poll loop | the same three calls, answered by the broker |
| captioning tool | ~360 lines inferring container VRAM by subtraction | the broker's per-process attribution and measured costs |
| image stack | nothing at all | warns when the card is too full to bother starting; opt-in leaseholder |

**The container question is answered.** The image stack took a 12 000 MB lease
naming the host PID from `docker compose top` while the container already held
980 MB; `outstanding` came back 11 020. Reclaim needs nothing beyond `pids` on
the acquire request, so the compose-aware path in vramux that §13 anticipated is
not required.

The captioning tool is the clearest case for the whole project: its own comments
said container VRAM "cannot be isolated", so it guessed by subtracting an
estimate from the total. The broker measures it, and measures what each model
costs, so two guesses became two readings.

What the migrations did *not* do: nothing holds a lease for its entire working
period by default. Leaseholder mode in the image stack is opt-in, because that
container idles for long stretches with its models unloaded and a standing
reservation over an idle container would starve the box while looking
principled. Turning it on by default wants the cooperative-yield path, which is
Stage 6.

## Stage 6 — Multi-residency

**Needs a free GPU and the Stage 2 dataset. Do not start it without both.**

The dataset is closed as of 2026-08-07: every registered tag has a measured
cost, taken one at a time on an otherwise idle card.

Model tags on this machine are private, so the table is by shape. The point is
the spread, not the catalogue:

| params | ctx | measured |
|---|---|---|
| 35B | 16384 | 21 405 MiB |
| 27B | 65536 | 18 970 MiB |
| 35B | 131072 | 18 916 MiB |
| 14B | 81920 | 18 635 MiB |
| 27B | 16384 | 17 555 MiB |
| 12B | 81920 | 10 993 MiB |
| 9B | 98304 | 9 231 MiB |
| 9B | 16384 | 6 591 MiB |
| 9B | 4096 | 6 195 MiB |

**Context dominates, not parameter count.** A 14B at 81 920 tokens costs 18 635
MiB — more than a 27B at 16 384 — and the same 9B weights span 6 195 to 9 231
MiB across three windows. Admission cannot reason about a model; it has to
reason about a model *at a context*, which is what the cost cache is keyed on
and why raising `_ADMITTED_RESIDENTS` from a parameter count would be wrong.

Against the 23 540 MiB ceiling that gives six admissible pairs, all of them
drawn from the four cheapest entries: 6 195 + 6 591, 6 195 + 9 231, 6 591 +
9 231, 6 195 + 10 993, 6 591 + 10 993, and 9 231 + 10 993 at 20 224 MiB, which
is the tightest of them. Everything from 17 555 upward is a single-resident
model on this card, and `exclusive: true` is the honest way to say so.

- Measure-and-learn feeding admission, replacing estimates with recorded costs.
- Port pool, so more than one local backend can run.
- LRU eviction under pressure; `exclusive: true` for models that want the card.
- Budget opens.

The first real win is the small one: an embedding model resident alongside a
chat model, so a 512 MB request stops evicting 20 GB. A 9B and a 12B at working
context are roughly 17 GB resident together and genuinely fit on a 24 GB card.

**Exit:** two models resident and serving. No OOM across a week of normal use.

### What normal use actually looks like, and why it will not test this

A full batch-pipeline run was measured end to end at one sample per second. It
peaked at **22 937 MiB of a 24 564 MiB card** — one consumer, 93% of the card,
climbing 15 GB in seconds as it moved between a language model and a diffusion
stage. The card was never shared during it, because nothing was left to share.

That is worth stating plainly: **the workload this machine runs today gains
nothing from multi-residency.** Packing helps traffic made of several modest
consumers, and the daily traffic here is one large one. The stage is aimed at
the shape the machine should be able to run, not the shape it currently does.

The consequence is that normal use will never exercise this. The scenario has
to be built deliberately, and both halves of it exist:

- **Two residents, no downloads.** The registry's smallest model at a small
  context is 6 195 MiB (measured); the same weights at 16K are 6 591 MiB.
  Registered as separate tags, that is 12 786 MiB resident together, which
  fits with room to spare. Same weights on disk, two genuine residents, and
  it works only once `_make_room_for()` evicts on cost rather than on count.
- **Several claimants at once.** `tools/hold_vram.py` allocates real VRAM
  through the driver and holds it, optionally under a lease with its own owner
  and renewal loop. `tools/scenario_small_loads.sh` runs several at staggered
  arrivals and checks the budget through the whole cycle. This is the part
  serving cannot test: many holders, arrivals interleaved with allocations,
  and reclaim when a holder dies without cleanup.

Both were used to check the Stage 4 accounting under four concurrent holders:
grants tracked, `outstanding` fell to zero as each holder allocated, the card
returned exactly to its starting numbers on release, and a request larger than
the card failed 413 immediately rather than waiting out its timeout.

One reporting defect surfaced and has been fixed: `covered_mb` was a
grant-time snapshot, so a holder that leases before it allocates — the correct
order — reported `covered_mb: 0` for its whole life while the budget correctly
saw its memory. The arithmetic was right and the field was misleading. It is
now `covered_at_grant_mb`, which says what it is, and every lease payload
carries live `observed_mb` and `outstanding_mb` beside it, taken from the same
accounting `budget.py` computes from rather than worked out a second time. A
console can draw from those.

The other thing that run argued for landed with it: `VRAMUX_SAMPLE_INTERVAL`.
The history sampled every five minutes, and a measured batch run took the card
to within 1 627 MiB of full during a stage that lasted fourteen seconds — an
event the history could not see at all. The knob clamps up to the 5 s lease
sweep it rides on, and says so when it does.

### What Stage 6 landed, 2026-08-07

Admission is open. `_ADMITTED_RESIDENTS = 1` is gone; what decides now is
measured cost against `budget.free_mb` — the same arithmetic a lease is granted
from, not a second accounting built from declared costs.

Four rules, in the order admission applies them:

1. **`exclusive: true`, either way round.** A model that says it wants the card
   gets it alone, and nothing joins a resident that said so.
2. **No known cost, no sharing.** Measured or declared in config, or the model
   is served alone. There is deliberately no estimate in this path: §4.2
   describes one and admission does not use it, because an underestimate is an
   OOM that takes the innocent resident with it. A model becomes packable the
   first time it loads, since that load measures it.
3. **The resident ceiling** (`VRAMUX_MAX_RESIDENTS`, default 2). A bound on
   backend processes and upstream ports, not on memory.
4. **The budget.** Evict least-recently-used until the incoming cost fits in
   what is free.

Everything else the stage needed came with it: a port pool so two
llama-servers can coexist, per-tag upstream routing (a global `upstream`
would have proxied every request to whichever model was admitted last),
`/api/ps` reporting a list, `keep_alive: 0` unloading the model named rather
than the card, and `auto` resolving to the most recently *used* resident
rather than the most recently admitted one.

The failure path §11 asked for exists: a load that fails beside a peer evicts
the peers and retries once, alone. Free memory is not always allocatable
memory, and one retry turns that into a served request rather than an error. A
load that fails on an empty card is not retried — that is a broken model, and
looping on it turns a clear failure into a hung request.

**Verified on the card**, not just in tests: two 9B models resident and
serving at 14 009 MiB; the tight pair — 9 231 + 10 993 — resident at 21 449
MiB with no OOM; a 27B correctly evicting both; and a 12 000 MiB leaseholder
being enough to stop a second model being admitted, which is residency and
leasing agreeing through one budget rather than two.

**Exit met:** two models resident at once, admitted from measured costs. The
week of normal use is still ahead of it.

## Stage 7 — The console

Only now, and only because everything under it is real.

`vramux top` — live view of the card: what is resident, who holds leases, what
is foreign, what is queued and waiting. The thing that was wanted at the start.

The existing tools stay exactly as they are. They gain a floor, not a rewrite.

### What landed, 2026-08-07

**A terminal console, streamed.** `vramux top` draws the card from
`/gpu/events`, a server-sent event stream carrying the same body `/gpu/state`
returns. Both open questions in this file are answered by it:

- **Streaming, not polling** — and the reason it is worth an endpoint is that
  it costs *less* than polling did. Every watcher shares one reading, so a
  second console is a socket rather than a second `nvidia-smi`, and nothing
  samples at all while nobody is attached. A console that cannot stream falls
  back to polling `/gpu/state` and says so on screen, which is also what
  happens against a router older than this file.
- **A TUI first, and then a web view** — `vramux top` is stdlib `curses`, no
  dependency, and works over the SSH connection you open when the box is in
  trouble. `/gpu/console` serves the same view as one self-contained HTML
  file that reads the same stream: no build step, no framework, and nothing
  fetched from a network the machine may not have.

Frames go out **on change**, which is why the payload carries absolute
timestamps and no ages: an idle counter computed server-side ticks every
second and would turn "publish when the card moved" into a per-second
broadcast of nothing. Ages and TTLs are worked out by the console against its
own clock — the same machine's clock, since a consumer of this endpoint is on
the card's box. A card doing nothing sends one frame and then a comment line
every fifteen seconds, so a dead socket surfaces as a write error rather than
as a console quietly watching a router it lost.

`/gpu/state` grew `resident_detail` alongside the `residents` tag list it has
always had — port, in-flight requests, last use, and what the model is
believed to cost. Additive on purpose: clients on this machine already read
`residents` as a list of tags.

Two things the live run corrected, neither of them arithmetic:

- A leaseholder's process was being drawn under FOREIGN while its memory was
  already on screen as HELD. Same 2 386 MiB, twice, which reads as two
  allocations — the exact confusion one accounting exists to prevent.
- HELD above the grant is normal and stays unmarked: 2 000 MiB requested is
  2 386 MiB on the card, the CUDA context being the difference. What is worth
  marking is OUT — a grant nobody has allocated against.

Rendering is a pure function from state to lines, so every layout decision is
tested without a terminal, a card, or a router. 30 tests came with it.

## After the stages — yield, 2026-08-07

The numbered roadmap ended with the console. This is the first thing built past
it, and it closes the gap every earlier stage was careful to name: **a lease was
never revoked, so a big model waited for one to expire rather than asking for
it back.** `DESIGN.md` §6 eviction tier 3, now §6.2.

What it is: a request with a deadline, delivered on the heartbeat the holder is
already sending. What it is not, and must never become: a way to take memory
back. The deadline decides when vramux logs that a holder ignored it — nothing
else. A holder that has never heard of yield behaves identically with it on,
which is why it could be turned on against the live router the day it was
written.

It also had to answer the question this file listed as open since Stage 4:
**priority is higher-wins**, with three named points (`batch` 1, default 5,
`interactive` 7), asked strictly downward so equal never yields. `models.yml`
grew `priority:`, and a holder opts out of ever being asked by taking the
serving priority or above — an honour system, on a box with one operator, said
out loud rather than pretended otherwise.

Verified live, both directions:

| what | result |
|---|---|
| a 16 000 MiB batch lease, then a model needing 272 MiB more | asked for **272 MiB**, not the model's whole cost |
| the holder releasing | "the memory came back, 22 319 MiB free", load resumed in 13 s |
| a holder that ignored it | "nothing yielded within 30s", then one warning naming the holder — at the time the load went ahead anyway; that ending has since become a retryable refusal |
| `vramux lease --on-yield term` | forwarded SIGTERM to its command, which exited and released — the whole cooperative loop, end to end |
| `vramux top` during a request | the lease row went red with "asked for 4 273 MiB by serving:qwen3.5:9b — 23s left" |

18 tests, and the one that matters is `test_a_yield_request_takes_no_memory_and
_shortens_no_lease`.

---

## After the stages — history on the console, 2026-08-07

The sampler has written `usage.jsonl` since Stage 4 and nothing could read it
back over HTTP, so both consoles could draw the card as it is and never as it
has been. `GET /gpu/history` closes that, and the page grew a sparkline of the
last hour: used solid, foreign dashed, scaled against the card's total rather
than against its own extent, so a quiet hour looks quiet instead of looking
like a busy one at a different zoom.

Three decisions worth keeping:

- **A separate endpoint, not a field on `/gpu/state`.** History changes once a
  sample interval. Shipping it in a frame that goes out on every change would
  put an hour of unchanged rows on the wire whenever a single number moved.
- **It reads the file and never the card.** A console redrawing its chart costs
  no `nvidia-smi` call, which is the same argument `StateFeed` makes for
  streaming rather than polling.
- **`interval_s` travels with the rows.** Twelve points is an hour or a minute
  depending on the cadence, and the page uses it twice: to label the chart, and
  to break the line across a gap wider than a couple of intervals. Joining
  across the hours a router was down would draw a confident straight line
  through memory nobody measured.

The live check was a throwaway router on another port with
`VRAMUX_SAMPLE_INTERVAL=5` and its own cache directory, plus
`tools/hold_vram.py` — 9 000 MiB appearing and going away, on screen, without
touching the running service. 10 tests came with it.

`vramux top` did not get the sparkline. The TUI reads one stream and nothing
else, and giving it a second transport to draw a chart at 5 s samples in
`curses` buys less than it costs.

## Stage 8 — A lease can evict a model — DONE (2026-08-24)

Spec: `specs/lease-evicts-managed.md`. The §3 tier table finally wired to
leases: managed consumers were always the tier vramux may evict silently, and
until now only serving ever exercised it. A short lease that waits and
*outranks* an idle resident — strictly above its `priority:`, else
`VRAMUX_SERVING_PRIORITY` — has it drained and evicted instead of waiting out
a 15-minute idle timer or making the client hand-call `/gpu/evict` first,
which is what every serious lease client had learned to do.

What landed:

- `ResidencyArbiter.make_room_for_lease(mb, by, priority)` — cheapest-first
  among eligible residents, drain-first via the same `_evict` a swap uses,
  stops when the measured cost covers the shortfall. Returns the eviction
  *count*, not megabytes: an unmeasured victim frees an amount only the card
  can report, so counting stops there and the caller re-reads the budget.
- `Broker.use_make_room()` — wired in `__main__.py` beside `use_budget` /
  `use_yield`, the same one-callable-each-way shape, so the broker still
  never imports residency and a bare `Broker` behaves exactly as before.
- In `Broker.acquire`: evict once per acquire, *before* yield — eviction
  costs its victim a reload later, yield stops running work now, and the
  cheap disruption often makes the expensive ask unnecessary.
- No new configuration. The gate is the priority pair already in every
  request and every model entry, and it defaults closed: lease 5 < serving 7.

12 tests (`test_lease_evict.py`); the compatibility half of the suite pins
that a default-priority lease, a `wait: 0` lease, an unwired broker, and a
residency layer that raises all leave the world exactly as Stage 4 built it.

This also part-answers the standing open item below — *whether anything
should enforce priority*: for managed residents, yes, eviction; for everyone
else, still nothing takes.

## Stage 9 — Serving queues behind a lease it does not outrank — DONE (2026-08-24)

Spec: `specs/serving-queue.md`. Stage 8's rule read back from serving's
side, and the half that makes the pair usable on an agent machine: a lease
taken above serving's priority exists precisely so the card is not disturbed
mid-generation, and a chat request arriving during it is not failing — it is
*outranked*. It used to get a hard 503 after `yield_wait`; most
OpenAI-compatible clients surface that as a failed turn, so the one client
the machine runs all day could not survive the one state the machine was
designed to be in. Now it parks: slow first token, same connection, no error.

What landed:

- `Broker.outrankers(priority)` — the leases at or above a priority, each
  with `mb` (the larger of grant and observed) and `remaining_s` to expiry.
  Injected into residency as `use_outrankers`, the fourth and last pairing.
- `ResidencyArbiter._park_behind_leases` — entered from every refusal site
  in `_ask_leases_to_yield`, and deliberately narrow: it parks only when the
  blockers outrank the model AND releasing them would make the load fit.
  Foreign growth, an unco-operative peer below serving, a cost that never
  fits — all still refuse exactly as before.
- Bounded twice: `VRAMUX_QUEUE_WAIT` (default 600, `0` restores fail-fast)
  is a hard cap renewals cannot push past, and the blocking lease's own TTL
  plus a sweep-width grace bounds it from the other side — TTL being
  mandatory is what makes the wait a promise rather than a hope.
- Honest while parked: the load registers as in flight with a `behind`
  field, so `/gpu/state`, the admission wait message a second caller gets,
  and the cadence log all say "queued behind lease X", not "still loading".
  A client that disconnects abandons the park, surrenders the lock, and
  leaves no phantom load.

10 tests (`test_serving_queue.py`). With Stage 8 this closes the open item
below: priority is enforced by eviction for vramux's own children and by
patience for everyone else, and nothing anywhere revokes or takes.

---

## Working around a busy GPU

For most of this work the card was occupied by a training run. That constrains
*nothing* important:

| stage | needs GPU? |
|---|---|
| 0 safety net | no |
| 1 identity | no |
| 2 observer | no — and it benefits from the training run being there to watch |
| 3 structure | no |
| 4 leases | no |
| 5 clients | partially |
| 6 multi-residency | **yes, exclusively** |
| 7 console | no |

Six of eight stages are GPU-free. Stage 6 is the only one that must wait, and it
is the one that most wants the measurement history Stage 2 will have collected by
then. A long-running foreign job is not a delay; it is the observer's first
subject.

## Risk register

| risk | stage | mitigation |
|---|---|---|
| refactor breaks the daily driver | 3 | Stage 0 tests land first and stay unedited |
| cost underestimate → OOM kills innocent resident | 6 | Stage 2 collects real data for weeks first; budget shut until then |
| ~~container PID attribution doesn't work~~ | ~~5~~ | **retired at Stage 2** — verified working against a real container backend |
| clients never migrate | 5 | CLI wrapper must be shorter than the hack it replaces |
| codebase forks and diverges | 1a | move, never copy; develop against the live service |
| publishing pressure distorts priorities | any | publishing has no deadline and drives no decision |

## Settled

- Name `vramux`, verified unclaimed on PyPI, GitHub, npm
- One repo: broker core, serving layer as its first client
- Move to the final location early; develop live, never fork
- Leases dropped on broker restart; holders demote to foreign
- Publishing is a side effect of good hygiene, not a driver, and has no deadline
- The machine's existing GPU tools are clients, not migration targets

## Open

- Multi-GPU placement policy (device index threaded through, nothing designed)
- ~~Whether anything should *enforce* priority~~ — **answered at Stages 8–9**:
  eviction for vramux's own children, patience for everyone else. Yield still
  asks; nothing anywhere takes

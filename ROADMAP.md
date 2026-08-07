# llama-router → vramux

How to get from a single-slot model router to the control console this machine
was always supposed to have, without breaking the machine on the way.

Companion to `DESIGN.md` (what vramux is). This is the sequence.

---

## The diagnosis

The control console was attempted once. Its remains are still on disk:

```
May 26 17:05  a local tool the captioning tool the image stack a local tool the TTS tool    ← console, set up in one sitting
Jun 16 15:51  vram-free                                   ← three weeks later
```

`<console-dir>` is referenced by four READMEs and no longer
exists. The launchers survived; the directory that was supposed to unify them
did not.

It did not fail for lack of ability. It failed because **it was a launcher layer
with nothing underneath it.** Every `my*` tool knows how to start its workload.
None of them can know whether starting it is *safe*, because no component owns
that question. `vram-free` is the console discovering its own missing floor, and
`the batch pipeline`'s poll loop and `the captioning tool`'s `the GPU advisor` are two more
independent discoveries of the same hole.

So the sequence below does not rebuild the console. It builds the floor, then
lets the existing tools stand on it. The `my*` family is the client set, not the
thing being replaced.

## Principles

These constrain every stage. They are why the ordering looks conservative.

1. **The daily driver never breaks.** The router on :11434 is load-bearing —
   a client, an editor client, the captioning tool, the batch pipeline, a dependent project all depend on it. Every
   stage ends with a working router or it is not finished.
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

## Current state

- 1,370 lines across four modules, no tests
- single slot, one backend at a time, swap verified working both directions
  (17 s cold container, 7 s container→GGUF, 16 s back)
- hardcoded paths to `<home>` and `<data-mount>`
- `MYLLAMA_*` env prefix, `the old unit` user unit
- GPU currently occupied by a training run — see "Working around the GPU"

---

## Stage 0 — Safety net

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

## Stage 1 — Move and identity

### 1a. Move, do not fork

The repo relocates to `<final-location>/` **now**, early, and
development continues there against the live service.

The tempting alternative is to copy the code to `oci-public`, build vramux there
while `llama-router` keeps serving :11434, and swap at the end. Do not do this.
It produces two codebases, and:

- **Nothing is dogfooded.** The version under development is not the version
  running, so bugs are found by inspection instead of by use.
- **Fixes diverge.** Anything repaired in the live router must be re-applied to
  the fork or silently lost.
- **All risk concentrates at cutover** — every bug from every stage arriving at
  once, on the day load-bearing infrastructure is swapped.
- **The machine gets nothing until the end.** Stages 2, 4 and 5 each pay off on
  this box immediately; a fork defers all of it.

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

Match how `talkrec` and `tmux-kit` wire into `oci-public/.githooks/`.

### 1b. Identity

Mechanical, low risk, and it stops the old name from spreading into new code.

- `MYLLAMA_*` → `VRAMUX_*`, with a shim that reads the old names and warns.
- `the old unit` → `vramux.service`.
- Strip `<home>` and `<data-mount>` defaults: resolve the inference binary from
  `$PATH`, model dir from env or `./models`, blob roots only if they exist.
- `ctx_overrides` moves out of `registry.py` into shipped example config — it is
  this box's configuration wearing a module as a disguise.
- `models.yml` stays local and gitignored; `models.example.yml` ships.

Port 11434 does not change, so every client is unaffected. The unit file is the
only disruption, and it is a single restart.

**On publishing.** It is not a driver of any decision in this roadmap, but the
destination is claimed at 1a and being publish-shaped — no absolute paths, real
tests, honest docs — is hygiene that makes the machine's own infrastructure
better. It comes as a side effect, not a goal, and there is no deadline on it.

**Exit:** clean checkout runs on a machine that is not this one. No test asserts
a path under `/home`.

## Stage 2 — The observer

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

It also starts recording the training run currently holding 12 GB, which is
exactly the tier-3 case that has to work.

**Exit:** `vramux state` prints a true picture of the card, including processes
vramux has never heard of. Cost cache has real entries for every model that has
been loaded since the observer landed.

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

## Stage 5 — Clients

One at a time, most pain first. Each migration is a separate, revertible change,
and each one is the first time a real person gets a real benefit.

| order | client | what changes | why this order |
|---|---|---|---|
| 1 | `the batch pipeline` | poll loop → `vramux lease -- ./stage2.sh` | worst hack, clearest win, easy rollback |
| 2 | `the print pipeline` | `vram-free` deleted or reduced to `vramux free` | proves the primitive replaces the workaround |
| 3 | `the image stack` | leaseholder; yields via its unload endpoint | first cooperative yielder, first container PID attribution test |
| 4 | `the captioning tool` | drop `the GPU advisor`, ask the broker | removes the third reimplementation |

Stage 5 is where the untested assumption in `DESIGN.md` §13 gets tested — whether
NVML attributes container processes to host PIDs the way reclaim needs. Client 3
is deliberately the one that proves it, with clients 1 and 2 already delivering
value if it turns out to be harder than expected.

**Exit:** no project on this box reimplements VRAM reasoning. `vram-free` and
`the GPU advisor` are gone.

## Stage 6 — Multi-residency

**Needs a free GPU and the Stage 2 dataset. Do not start it without both.**

- Measure-and-learn feeding admission, replacing estimates with recorded costs.
- Port pool, so more than one local backend can run.
- LRU eviction under pressure; `exclusive: true` for models that want the card.
- Budget opens.

The first real win is the small one: an embedding model resident alongside a chat
model, so a 512 MB request stops evicting 20 GB. On this box `qwen3.5:9b` plus
`gemma4-coder:12b` is roughly 17 GB resident together and genuinely fits.

**Exit:** two models resident and serving. No OOM across a week of normal use.

## Stage 7 — The console

Only now, and only because everything under it is real.

`vramux top` — live view of the card: what is resident, who holds leases, what is
foreign, what is queued and waiting. The thing that was wanted at OS install.

The `my*` tools stay exactly as they are. They gain a floor, not a rewrite.

---

## Working around the GPU

The card is busy with training. That constrains *nothing* important:

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
then. The training run is not a delay; it is the observer's first subject.

## Risk register

| risk | stage | mitigation |
|---|---|---|
| refactor breaks the daily driver | 3 | Stage 0 tests land first and stay unedited |
| cost underestimate → OOM kills innocent resident | 6 | Stage 2 collects real data for weeks first; budget shut until then |
| container PID attribution doesn't work | 5 | client 3, after 1 and 2 have already paid out |
| clients never migrate | 5 | CLI wrapper must be shorter than the hack it replaces |
| codebase forks and diverges | 1a | move, never copy; develop against the live service |
| publishing pressure distorts priorities | any | publishing has no deadline and drives no decision |

## Settled

- Name `vramux`, verified unclaimed on PyPI, GitHub, npm
- One repo: broker core, serving layer as its first client
- Move to `<final-location>/` early; develop live, never fork
- Leases dropped on broker restart; holders demote to foreign
- Publishing is a side effect of good hygiene, not a driver, and has no deadline
- The `my*` tools are clients, not migration targets

## Open

- Multi-GPU placement policy (device index threaded through, nothing designed)
- Whether `/gpu/state` needs streaming or polling suffices
- Priority granularity: per-consumer, or just interactive versus batch
- Whether Stage 7 is a TUI or a web view

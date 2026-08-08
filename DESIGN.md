# vramux — design

**vramux is a VRAM broker.** It arbitrates one GPU between every consumer that
wants a piece of it: language models, image models, TTS, trainers, and whatever
else is on the box. Serving language models is the first thing built on top of
it, not the point of it.

Status: design. Most of this is not implemented yet; `ROADMAP.md` is the
sequence and records what is done. Supersedes the single-slot model-router
framing this project started from.

---

## 1. Why

One 24 GB card, many consumers, no coordinator. Today every consumer on the
machine solves that alone, and each one solves it wrong.

A batch pipeline's unload helper — evict, poll, give up:

```bash
ollama_unload(){
  curl -s .../api/generate -d '{"model":"...","keep_alive":0,"prompt":""}'
  for _ in $(seq 1 30); do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "${free_mb:-0}" -gt 18000 ] && break
    sleep 2
  done
}
```

A hand-written `vram-free` script — stops the image stack, unloads the language
model, and prints VRAM before and after.

A captioning tool's GPU advisor module — a third one, which at least knows what
it is up against:

```python
# NOTE: nvidia-smi shows aggregate VRAM usage from ALL processes (host + Docker containers).
```

Three tools independently reimplementing the same reasoning about one card.
All three share the same defects:

- **No reservation.** Observing 18 GB free does not make it yours. Two callers
  observe it simultaneously, both proceed, both die.
- **Eviction is total.** The only lever is "unload everything," so a 512 MB
  embedding request costs a full reload of a 20 GB model.
- **Failure is silent.** The poll loop times out after 60 s and continues
  anyway, into a card that does not have room.

Nothing accounts for consumers that were never asked. The desktop compositor
holds a couple of hundred MiB right now; every budget on the machine is wrong by
that much and by whatever else drifts in.

The fix is a single process that knows the true state of the card and is the
only thing allowed to hand out room on it.

## 2. What vramux is not

- Not a scheduler for compute. It arbitrates **memory residency**. Two resident
  models contending for SMs is expected and fine.
- Not a sandbox. It cannot stop a process from allocating. A consumer that
  ignores vramux wins, and vramux's job is then to notice and stay out of the way.
- Not multi-GPU in v1. The design keeps a device index throughout so this is an
  extension rather than a rewrite, but nothing is built or tested for it.

## 3. Consumer tiers

The trust model has exactly three tiers, and the distinction is *how much
vramux can do to them*, not how much it trusts them.

| tier | vramux can | examples |
|---|---|---|
| **managed** | start, stop, evict silently | language-model backends vramux itself spawns |
| **leaseholder** | grant, deny, request yield | batch pipelines, image stacks, TTS, trainers |
| **foreign** | observe only | desktop compositor, stray scripts, anything unaware |

**Managed** consumers are the easy case. vramux owns the process lifecycle, so
eviction is transparent — the consumer never learns it happened, it just pays a
reload next time. All current backends are here.

**Leaseholders** are cooperative external processes. They ask before allocating
and release when done. vramux can *ask* one to yield but cannot force it.

**Foreign** is everything else, and it is the tier that makes the design honest.
vramux enumerates compute processes on the device via NVML, subtracts anything
it does not recognise from the budget, and treats that memory as gone. It never
tries to reclaim it. Without this tier the budget is fiction.

## 4. Accounting

### 4.1 The budget

```
budget = device_total
       - reserve                  # config, default 512 MB
       - sum(foreign_process_mb)  # NVML, refreshed on a timer and before admission
```

Everything vramux grants comes out of `budget`. Managed residents and active
leases both draw down the same pool; there is no separate allowance.

Foreign usage is re-read immediately before any admission decision, not from
cache. It moves without warning, and a stale read is exactly the case that OOMs.

### 4.2 Cost estimation

Admission needs to know a cost before the allocation exists. Three sources, in
order of preference:

1. **Measured.** vramux recorded what this exact configuration actually used
   last time. Authoritative.
2. **Declared.** `vram_mb:` in config. Required for container backends — their
   internals are not introspectable.
3. **Estimated.** Computed for local model files from header metadata:

   ```
   weights   = file size on disk
   kv_cache  = 2 * layers * kv_heads * head_dim * ctx * bytes_per_element
   overhead  = cuda_context + compute_buffers    # measured constant, per backend kind
   ```

   The KV term dominates at long context and is the reason file size alone is a
   useless estimate: the same weights at 16 K and at 128 K differ by many GB.

Estimates carry a configurable safety margin (default 15%). Measurements do not.

### 4.3 Measure and learn

Every managed load is measured:

1. Sample device used-memory before start.
2. Start, wait for ready.
3. Sample again. Delta is the real cost.
4. Write to `~/.cache/vramux/costs.json`, keyed by a hash of the configuration
   that determines footprint (model identity, context length, backend kind,
   quantization, offload settings).

Later loads of the same key use the measured value with no margin. A key whose
inputs change falls back to estimation and re-measures.

This is what makes the budget tighten over time instead of staying permanently
conservative. It is also the only part of the cost model that is trustworthy
enough to pack two models onto one card.

Measurement caveat: the delta includes anything else that allocated during the
window. vramux takes the sample under its own lock and discards a measurement
if foreign usage moved during it.

## 5. Lease protocol

A lease is a promise that `mb` megabytes will remain available to the holder
until released or expired.

```
POST   /gpu/lease            {"mb": 18000, "owner": "batch-pipeline",
                              "priority": 5, "ttl": 300, "wait": 120}
       → 200 {"lease": "lse_...", "granted_mb": 18000, "expires_at": "..."}
       → 408 if not grantable within `wait`
       → 413 if never grantable (exceeds budget even when empty)

DELETE /gpu/lease/{id}        release
POST   /gpu/lease/{id}/renew  extend by ttl
GET    /gpu/state             residents, leases, foreign, free
```

**TTL is mandatory and there is no infinite lease.** A holder that dies without
releasing must not strand the card. Renewal is a heartbeat: hold longer than
`ttl` and you must say so, repeatedly. Expiry logs loudly — a lease that expires
under a live holder is a bug in that holder, and silence would hide it.

`413` versus `408` matters: asking for more than the card can ever provide is a
configuration error and should fail immediately, not block for two minutes first.

### 5.1 Broker restart, and why dropping leases is safe

On restart vramux drops every lease. It does not persist them. This is correct
rather than merely convenient, and the reason is tier 3.

**A lease is bookkeeping, not the allocation.** Dropping one frees nothing. A
holder that was mid-work keeps every byte it had; nothing reached into its
process. What actually happens is a silent demotion:

```
leaseholder  --[broker restart]-->  foreign
```

vramux no longer knows who that memory belongs to, but it still *sees* it via
NVML and still subtracts it from the budget. The holder loses its guarantee and
its priority standing. It does not lose memory, and vramux does not hand that
memory to someone else.

So the invariant that matters survives a restart: **the budget always reflects
the real card**, whether or not vramux remembers who asked for what. Persisting
leases would add a second source of truth that can disagree with NVML, which is
strictly worse — a restarting broker has already lost track of the card, and
pretending otherwise is how you get a budget that believes 18 GB is free while
an image stack is sitting on it.

vramux is not the controller for these consumers and does not need to be. It
needs to be the only thing that *hands out* room, and to be honest about room it
did not hand out.

### 5.2 Reclaim: re-acquiring without double-counting

The recovery path is a trap. A holder's `renew` fails `404 unknown lease`, so it
re-acquires — but vramux is already counting that same memory as foreign.
Reserving it again charges the budget twice and the card appears to shrink.

Therefore a lease request carries the holder's `pid`, and admission
**reconciles against observed usage rather than adding to it**:

```
POST /gpu/lease {"mb": 18000, "owner": "image-stack", "pid": 4127, ...}

observed = foreign usage attributable to pid (and its container, if any)
charge   = max(0, mb - observed)      # only the shortfall is new
```

The lease then *covers* the existing allocation: that memory moves from foreign
back to accounted-for, and the holder is a leaseholder again with its guarantee
restored. If it asks for more than it currently holds, only the difference goes
through admission.

Container-resident consumers are the common case here — image stacks and
captioning backends typically run their GPU work in containers — so PID attribution
must map a container's processes to the lease. NVML reports host PIDs for
container processes, which makes this tractable.

**Tested, not assumed.** A container stack took a 12 000 MB lease naming the
host PID from `docker compose top` while its process already held 980 MB on the
card; the broker charged the difference and reported `outstanding` as 11 020.
No mechanism beyond `pids` in the acquire request was needed. Two things make
it work and both are load-bearing: the PID column is read from the header
rather than by index, and attribution matches a process *or any of its
descendants*, so a wrapper's pid covers the command it runs.

Same mechanism covers first-acquisition by a consumer that already allocated
before ever talking to vramux. It is not a special case.

### 5.3 CLI

The wrapper is what makes leases adoptable, because it makes the correct
version shorter than the broken version:

```bash
vramux lease --mb 18000 --owner batch-pipeline -- ./stage2.sh
```

Acquire, run, release on exit — including crash, including SIGKILL of the child,
including the wrapper being killed. Renewal happens on a background thread for
the command's lifetime. A twelve-line poll loop becomes this.

Also:

```bash
vramux state                       # what is on the card and who owns it
vramux top                         # the same, live, until you press q
vramux free --mb 8000              # block until 8 GB is available, then exit
vramux evict <tag>                 # drop a managed resident by hand
```

## 6. Residency and eviction

vramux holds a set of resident managed backends and a set of active leases.
Both consume budget. A request arrives for either.

Admission:

1. Refresh foreign usage.
2. If the request is a managed model already resident and healthy — reuse, done.
3. Compute cost. If `cost <= free` — admit.
4. Otherwise select eviction candidates until `cost <= free`, or fail.

Eviction order, cheapest regret first:

1. **Idle managed residents, LRU.** Zero in-flight requests. Cost of eviction is
   a future reload, nothing more. vramux does this without telling anyone.
2. **Managed residents with in-flight work**, oldest last-use first, after
   draining. Drain is bounded; a stream is never killed mid-response inside the
   drain window.
3. **Leaseholders below the requester's priority**, via yield request. Voluntary,
   with a deadline. A holder that does not yield keeps its memory and the
   requester waits or fails. **Built** — §6.2.
4. **Never foreign.** Not evictable by definition.

`exclusive: true` on a model config means it takes the whole card: admission
evicts every managed resident and blocks on all leases. A 20 GB model on a 24 GB
card is exclusive in practice, and declaring it is cheaper than discovering it
through a failed cost estimate.

### 6.2 Yield: the tier vramux cannot perform itself

Tiers 1 and 2 are vramux stopping things vramux started. Tier 3 is not: a
leaseholder is somebody else's process holding somebody else's allocation,
and nothing in a broker can free it. So the mechanism is a **request with a
deadline**, and the deadline is a reporting boundary rather than a taking one.

**Asking is not taking, and the code must never learn how.** When the deadline
passes with the lease still held, one warning is logged and the requester goes
back to waiting or failing exactly as it would have before yield existed. This
is why the feature was safe to turn on against a live machine on the day it was
written: a holder that has never heard of it behaves identically with it on.

**The transport is the heartbeat.** A renewal response carries a `yield` object
— who wants it, how much, by when — or `null`. No callback URL, no second
connection, nothing for a holder behind a container to publish, and no way for
a holder to be asked but not hear: it is already talking three times per TTL,
and one that has stopped talking is about to expire anyway.

```
POST /gpu/lease/{id}/renew
  → 200 {"lease": "lse_...", ..., "yield": {"wanted_mb": 4273,
                                            "by": "serving:qwen3.5:9b",
                                            "priority": 7,
                                            "deadline": "..."}}
```

**Priority is higher-wins**, which this document finally has to say because the
field was accepted and inert from Stage 4 until now. Three named points —
`batch` 1, default 5, `interactive` 7 — so that the two cases people actually
have are sayable without every client inventing a scale. Yield is asked of
holders **strictly below** the requester: equal never yields, or with a default
of 5 on both sides every ordinary holder would be asking every other ordinary
holder to get off the card.

Two consequences worth stating plainly:

- **The shortfall is what gets asked for, not the request.** A model needing
  272 MiB more than is free asks for 272 MiB, and the holders asked are the
  *cheapest* that cover it. Stopping a twelve-gigabyte pipeline to make room
  for a rounding error is worse than the contention.
- **A holder opts out by outranking the asker.** Taking `priority: 7` or above
  means serving never asks. That is an honour system on a single-operator box,
  and it is one on purpose — the alternative is enforcement, and vramux does
  not have any (§5, and "a lease is a promise, not a fence").

The serving side waits `VRAMUX_YIELD_WAIT` (30 s, `0` disables asking
entirely) and then loads anyway. Waiting indefinitely for an answer that may
never come turns a cooperative gesture into a hang, and the load was going to
happen regardless — the only question was whether the holder got told first.

### 6.1 The in-flight problem

Residency changes the drain semantics. In-flight counting is per-resident, not
global — evicting model A must not wait on requests in flight against model B.
This was the single largest structural change of Stage 3, and it is done: a
`Resident` owns its backend, its counter and its own drain gate, so an eviction
waits on its victim alone. With admission at one the distinction is invisible;
it exists so that opening admission is a budget change, not a re-reasoning of
every drain.

## 7. Serving language models

The serving layer becomes a **client of the broker**, one consumer among
several. It contributes:

- a registry of servable models and their configurations
- backends that know how to start one: a local inference subprocess, or a
  container that ships its own server
- translation between the **chat wire format** clients already speak on :11434
  (`/api/chat`, `/api/tags`, `/api/generate`, `/api/embeddings`) and the
  OpenAI-shaped API the backends actually expose

That wire format is a compat surface and nothing more. It exists because the
existing clients already speak it; it is not a dependency and not part of the
stack.

### 7.1 Backend contract

Pinned as a `Protocol` in `backends.py`, before a third implementation lands:

```python
class Backend(Protocol):
    upstream: str
    def alive(self) -> bool: ...
    async def start(self, spec: ModelSpec, startup_timeout: float) -> None: ...
    async def stop(self) -> None: ...
    async def healthy(self) -> bool: ...
    async def pids(self) -> list[int]: ...   # for attribution, best effort
    def adopt(self) -> None: ...             # claim a pre-existing instance
```

`adopt()` replaced the `backend._started = True` hack in `reconcile()`. Only
the container kind can honour it: compose addresses an orphan by file and
service name, so adopting one is just admitting it exists. A subprocess handle
does not survive our restart, so `ProcessBackend.adopt()` raises — a property
of the kind, not a failure.

`healthy()` sits here rather than on the arbiter, which is the decision Stage 3
was required to make and not defer. What "healthy" means is the kind's
business; *when to ask and what to do with the answer* stays the arbiter's, so
this returns a bool and never acts on it. Both current kinds share one
`http_healthy()` helper, which is exactly the shape a third kind would need to
override.

`vram_hint()` is deliberately **not** pinned. Nothing implements it and nothing
calls it: the observer measures real cost instead of asking a backend to guess,
and Stage 6 owns whatever the cost model turns out to need. Pinning a method
with no caller would freeze a guess.

Two implementations exist. vLLM and TensorRT-LLM are the obvious next ones and
the interface should stop moving before they arrive.

### 7.2 Ports

A single fixed upstream port only works when only one backend runs. Multiple
residents need a pool, allocated at start and returned at stop. Container
backends publish their own and are exempt.

## 8. Repo shape

**One repo. Broker as the core, serving as a component on top.**

```
vramux/
  core/        budget, residency, NVML accounting, eviction, cost model
  lease/       lease HTTP API + CLI wrapper
  serve/       model registry, backends, the :11434 compat surface
  backends/    local subprocess, container
examples/
  container-backend/    worked example of the backend contract
```

The alternative — publishing the broker alone and keeping serving private — is a
cleaner boundary but publishes the half nobody can evaluate. The broker's claim
is "this arbitrates a real GPU between real consumers," and the serving layer is
the proof. They ship together.

Two entry points: `vramux serve` (broker + serving) and `vramux` (the client
CLI). The broker can run without the serving layer; the serving layer cannot run
without the broker.

## 9. Migration

Existing consumers, in dependency order:

| consumer | today | after |
|---|---|---|
| clients on :11434 | chat wire format | unchanged — same port, same shape |
| batch pipeline | unload + poll + hope | `vramux lease -- ./stage2.sh` |
| `vram-free` helper | stop stack, unload, print | `vramux free --mb N`, or deleted |
| image stack | unmanaged, container | leaseholder; has an unload endpoint to yield with |
| captioning tool | GPU advisor reads aggregate | leaseholder; drop the advisor, ask the broker |

The image and captioning stacks run their GPU work in containers and neither is
controlled by vramux. They become leaseholders, not managed consumers — vramux
grants and asks them to yield, it never starts or stops them. Until they adopt
leases they are simply foreign, which is a correct and safe state, not a broken
one.

The `keep_alive: 0` unload convention stays supported — it is a crude lease
release and several things use it.

Env prefix moves `MYLLAMA_*` → `VRAMUX_*`, with a shim that reads the old names
and warns, because the unit file and several dependent projects reference them.

## 10. Security

Stated plainly, because the honest version is alarming:

- **No authentication.** Anything that reaches the port can load models, run
  inference, take leases, and evict other consumers' work. Loopback by default.
- **The config file is a trust boundary.** Container backends shell out to
  compose. Whoever writes model config chooses what containers start.
- **Container backends require membership in the `docker` group, which is
  root-equivalent on most systems.** Say it in `SECURITY.md`, not in a footnote.
- Lease denial-of-service is trivial and unmitigated: one client can take a
  large lease and renew forever. Acceptable for a single-operator box; it is a
  reason this is not a multi-tenant tool.

## 11. Risks

- **An underestimate is an OOM, and an OOM on a shared card can kill the
  innocent resident too.** Mitigations: measure-and-learn, safety margin on
  estimates only, `exclusive` for models known to want everything, and a
  conservative default budget in v1.
- **Fragmentation.** Free memory is not always allocatable memory. Budget
  arithmetic will sometimes admit something that then fails. Handle the failure
  path explicitly rather than pretending.
- **Cooperative eviction depends on clients that may not answer.** The image
  stack has an unload endpoint; an arbitrary training script has nothing. Yield
  requests have deadlines and failure is a normal outcome.
- **Adoption cost is outside this repo.** Leases are worthless until consumers
  take them, and every consumer is a separate change.

## 12. Phasing

**v0.1 — broker core, conservative.**
NVML accounting including foreign. Budget and cost model with estimates only.
Residency structure with per-resident in-flight counting. Admission tuned so one
managed resident at a time, matching today's behaviour exactly. Rename and
de-personalize. GPU-less test suite. This ships and is safe.

**v0.2 — leases.**
Lease API, TTL, renewal, CLI wrapper. Migrate the batch pipeline and the
`vram-free` helper, which is where the value first shows up outside this repo.

**v0.3 — real mux.** *Landed 2026-08-07.*
Measure-and-learn feeding admission. Multi-resident, port pool, LRU eviction,
`exclusive`. The budget opens only once the cost cache has real numbers in it.

One thing was decided during the build and is not in §4.2's three-source
ladder: **admission uses measured and declared costs only, never an estimate.**
An estimate is good enough to size a card, and not good enough to decide
whether a second model joins one that is already loaded — the failure mode is
§11's first risk, and it kills the innocent resident as well as the greedy one.
A model with no known cost is served alone, which is what vramux did for its
whole life before this, and it becomes packable the moment it loads.

The ordering is deliberate: the structure that makes multi-residency possible
lands in v0.1, but the budget stays closed until v0.3 has measurements to open it
with. Single-resident is the degenerate case of the general design, not a
different implementation, so nothing is thrown away.

## 13. Open questions

- ~~Does `/gpu/state` need to be streamable for a live view, or is polling
  fine?~~ **Answered.** It is streamable: `GET /gpu/events` pushes the same
  body as server-sent events, on change, with a keepalive comment while the
  card is still. Streaming won on cost rather than on latency — one reading of
  the card serves every watcher, and nothing is read at all when nobody is
  watching, where polling consoles multiply `nvidia-smi` calls by however many
  are open. Ages and TTLs are absent from the payload on purpose: a
  server-side counter ticks every second and would make every frame a change.
- ~~PID attribution for container-resident leaseholders (§5.2) assumes NVML
  reports host PIDs for container processes.~~ **Answered.** Verified against a
  real container backend: `docker compose top` and NVML agree on host PIDs, and
  the container's compute process attributes correctly. Note that the compose
  column layout varies by version, so the PID column must be read from the
  header rather than by index.
- ~~Is per-consumer priority worth configuring, or is a two-level
  interactive/batch split enough?~~ **Answered by building yield.** Neither, or
  both: the field stays a plain integer, higher-wins, with three named points
  (`batch` 1, default 5, `interactive` 7) so the two-level case needs no
  configuration and a third level needs no new mechanism. What is still
  undecided is whether anything should *enforce* it, and nothing does.
- Multi-GPU: device index is threaded through from the start, but placement
  policy is undesigned.

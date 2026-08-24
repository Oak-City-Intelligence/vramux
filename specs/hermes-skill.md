# Spec: a published hermes skill for vramux

Status: BUILT 2026-08-24 — `examples/hermes/vramux/` (SKILL.md + README.md).
The acceptance run below still stands as the live checklist; the serving
process now runs Stages 8–9, and the core loop — a lease evicting the
resident model, generation parking behind it and resuming — has been
exercised live end to end.

Original text follows. It depended on both companion specs
(`lease-evicts-managed.md`, `serving-queue.md`) — without them the skill
would have to teach the evict-first dance and mid-generation turns would
die on 503s, and a published skill must not encode workarounds for bugs we
intend to fix.

## Why a skill, and why it ships here

The integration problem is not code, it is *convention*: an agent whose own
brain is served by vramux can drive GPU-heavy generation on the same card,
but only if its tools take leases at the right priority and the agent expects
the right latency shape. That knowledge has to live somewhere a harness can
load it.

[hermes-agent] installs capabilities as skill directories — a folder with a
`SKILL.md` (YAML frontmatter: `name`, `description`; body: instructions the
agent reads). No harness source is modified; a user drops the directory into
`~/.hermes/skills/` and the agent has it. That makes a skill the publishable
unit: it rides in this repo, and anyone running hermes + vramux integrates by
copying one folder.

Ships as `examples/hermes/vramux/SKILL.md` (plus a `README.md` beside it for
humans). `examples/` because that is what it is — the first documented
client integration, and the template for writing one for any other harness.

## What the skill teaches

Four conventions. Everything else is commentary.

### 1. Never allocate VRAM bare

Any tool the agent runs that will touch the GPU — image gen, video gen, mesh,
TTS training, anything — runs under the wrapper:

```bash
vramux lease --mb <cost> --priority 8 --owner <tool-name> -- <command>
```

Acquire, run, heartbeat in the background, release on exit — the wrapper is
already the whole protocol. The skill forbids the alternatives by name:
no `nvidia-smi` free-memory polling, no `keep_alive: 0` unload hacks, no
"looks free, go". Observing room does not reserve it.

### 2. The priority convention

| priority | who | meaning |
|---|---|---|
| 8 | interactive generation the agent is driving now | evicts the idle brain; serving parks behind it |
| 7 | (serving — reserved) | the agent's own brain lives here |
| 5 | background/batch work | waits its turn; never disturbs the brain |

Priority 8 is a claim that a human is watching the output. The skill tells
the agent to use 5 for anything it queued for later and 8 only for the task
it is actively holding a conversation about. Nothing above 8 without the
operator saying so — 9+ is the operator's own override band, and an agent
that self-escalates has defeated the ranking.

### 3. The latency shape is normal

During a lease-held generation the agent's own model is evicted. The skill
sets expectations rather than letting the agent diagnose them as failures:

- First token after a generation completes is slow (model reload, tens of
  seconds). Not an outage. Do not retry, do not fail over.
- A turn that needs the LLM *while* a priority-8 lease is held parks until
  the lease releases (`VRAMUX_QUEUE_WAIT`). Also not an outage.
- Corollary the agent must respect: **think before you lease.** Do all
  prompt expansion, planning, and parameter math first; take the lease;
  run the tool without needing another LLM call until it releases. The
  turn-boundary structure of tool-calling agents gives this for free —
  the skill just names it so the agent does not schedule an LLM-dependent
  step mid-lease.

### 4. Short jobs block, long jobs background

- **Short** (an image batch, a mesh, under ~5 min): run the leased command
  synchronously inside the tool call. The agent's turn is parked anyway.
- **Long** (video, training, 10 min+): do not hold a turn open for it.
  Submit the leased command as a background job, end the turn, let the
  job's completion notify the agent (hermes has cron and pending-message
  plumbing; other harnesses have equivalents — the skill states the
  pattern, not the hermes internals). This keeps the window in which the
  brain is evicted-and-parked as short as each conversational turn needs,
  instead of the length of the render.

### Also in the body

- `vramux free --mb N` to wait for room, `vramux leases` / `GET /gpu/state`
  to see who holds the card, `vramux top --once` when reporting state to
  the operator. Read state from vramux, never from `nvidia-smi` — the
  device's free number cannot see reservations.
- Cost numbers: start from the pipeline's measured peak (the operator
  usually knows it; `/gpu/history` shows it after one run) and round up.
  A lease that is too small OOMs *you*; too big merely queues others.
- What errors mean, verbatim from the protocol: `413` = asked for more than
  the card has, fix the number, do not retry. `408` = busy, either wait
  longer (`--wait`) or come back. A `yield` field appearing on your
  heartbeat = someone outranks you; finish the current unit of work and
  release.

## Skill frontmatter (draft)

```yaml
---
name: vramux
description: >
  Share one GPU between this agent's own model and the generation tools it
  drives. Run any VRAM-consuming command under `vramux lease -- <cmd>`,
  follow the priority convention (8 interactive / 5 background), expect a
  slow first token after generation, and background any job over ~5 minutes.
  Use whenever a task involves image/video/mesh/audio generation, model
  training, or any GPU tool on a vramux machine.
---
```

## Boundaries

- **The skill is prose and a wrapper invocation. It contains no patches.**
  If integrating a harness requires editing that harness, that is a bug in
  vramux or the harness, not a skill body. (Hermes specifically: source
  edits break `hermes update`; nothing in the skill may suggest one.)
- Harness-agnostic core: sections 1–4 mention hermes only in the
  install line and the background-job example. The same file should read as
  integration doc for any tool-calling agent; a second harness gets a
  sibling directory, not a fork of the conventions.
- The skill does not configure vramux. Priorities, `VRAMUX_QUEUE_WAIT`,
  model pinning stay operator decisions in the service config; the skill
  teaches the client side only.

## Acceptance

Run on the reference machine (24 GB card, 27B brain via vramux, hermes):

1. Agent asked for an image: expands prompt, takes a priority-8 lease, brain
   evicted, generation runs, lease released, agent's summary turn pays one
   reload. No 503 surfaced to the user.
2. Second hermes session pings mid-generation: its turn parks and completes
   after release. Slow, not failed.
3. Agent asked for a long video: submits background job, turn ends promptly,
   completion notification arrives, agent reports the result.
4. Agent asked "what's on the GPU": answers from `/gpu/state`, not
   `nvidia-smi`.
5. The image pipeline driven by the agent end-to-end under one lease — the
   pilot client for the whole stack.

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

This machine has one GPU and a broker, vramux, that arbitrates it. The model
answering right now is served *by that broker* on the same card your tools
want. These conventions are how you drive GPU work without evicting your own
brain mid-thought — and without your thoughts blocking the work.

## 1. Never allocate VRAM bare

Any command that will touch the GPU — image generation, video, mesh, TTS,
training, anything — runs under the wrapper:

```bash
vramux lease --mb <cost> --priority 8 --owner <tool-name> --wait 120 -- <command>
```

That is the whole protocol: acquire, run, heartbeat in the background,
release on the way out. `--wait` matters — eviction of an idle model to make
room only happens for a lease that is willing to wait for it; a lease with
no wait fails fast instead.

Forbidden, by name:

- No `nvidia-smi` free-memory checks followed by "looks free, go". Observing
  room does not reserve it; two observers both proceed and both die.
- No `keep_alive: 0` unload hacks against the serving API.
- No allocating first and hoping. If the tool cannot be wrapped, it is not
  ready to run on this machine.

## 2. The priority convention

| priority | who | meaning |
|---|---|---|
| 8 | interactive generation you are driving now | evicts the idle serving model; serving parks behind it |
| 7 | (serving — reserved) | your own model lives here |
| 5 | background / batch work | waits its turn; never disturbs serving |

Priority 8 is a claim that a person is watching the output. Use it only for
the task you are actively holding a conversation about; anything you queued
for later runs at 5. Never take a priority above 8 on your own judgment —
9 and up is the operator's override band, and an agent that self-escalates
has defeated the ranking it depends on.

## 3. The latency shape is normal

While your priority-8 lease is held, your own model is evicted. Expect it,
and do not diagnose it as an outage:

- The first token after a generation completes is slow — the model reloads,
  tens of seconds. Do not retry, do not fail over.
- A request that needs the model *while* the lease is held parks until the
  lease releases (the broker's queue, bounded by its TTL). Also not an
  outage.
- Corollary — **think before you lease.** Do all prompt expansion, planning,
  and parameter math first; take the lease; run the tool to completion
  without needing another model call until it releases. Tool-calling turn
  structure gives you this for free; just do not schedule a model-dependent
  step mid-lease.

## 4. Short jobs block, long jobs background

- **Short** (an image batch, a mesh — under ~5 minutes): run the leased
  command synchronously inside the tool call. Your turn is parked anyway.
- **Long** (video, training — 10 minutes and up): do not hold a turn open.
  Submit the leased command as a background job, end the turn, and let the
  job's completion notify you (cron, a pending-message queue — whatever
  your harness provides). This keeps the window in which your model is
  evicted as short as each conversational turn needs, instead of the length
  of the render.

## Reading the card

Ask vramux, never the device — `nvidia-smi`'s free number cannot see
reservations:

```bash
vramux top --once            # one frame: residents, leases, foreign
vramux leases                # who holds what right now
vramux free --mb 8000        # block until 8 GB could be granted
curl -s localhost:11434/gpu/state   # the same, as JSON
```

## Cost numbers

Start from the pipeline's measured peak — `/gpu/history` shows it after one
run — and round up. A lease that is too small OOMs *you*; too big merely
queues others.

## What the errors mean

- `413` — you asked for more than the card can ever provide. Fix the
  number. Do not retry.
- `408` — not available within your `--wait`. Either wait longer or come
  back; the card is busy, not broken.
- A `yield` field on your lease heartbeat — something outranks you. Finish
  the current unit of work and release; the wrapper warns by default and
  can forward a signal with `--on-yield`.

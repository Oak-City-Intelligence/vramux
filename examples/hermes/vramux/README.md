# A vramux skill for agent harnesses

An agent whose own model is served by vramux can drive GPU-heavy generation
on the same card — the broker evicts the idle model for an outranking lease
and parks the model's next request until the lease releases. What makes that
work is not code, it is *convention*: tools must take leases at the right
priority, and the agent must expect the right latency shape instead of
diagnosing it as an outage. `SKILL.md` is those conventions, written for the
agent to read.

It is prose and a wrapper invocation. It contains no patches: if
integrating a harness requires editing that harness, that is a bug in
vramux or the harness, not something a skill body should paper over.

## Installing into hermes

[hermes-agent](https://github.com/NousResearch/hermes-agent) loads
capabilities from skill directories. Copy this directory in and the agent
has it:

```bash
cp -r examples/hermes/vramux ~/.hermes/skills/vramux
```

No hermes source is modified, so `hermes update` keeps working.

## Other harnesses

Sections 1–4 of `SKILL.md` are harness-agnostic — the lease wrapper, the
priority bands, the latency expectations, and the short-blocks / long-
backgrounds split apply to any tool-calling agent. Port them to your
harness's skill or system-prompt mechanism as a sibling directory here,
rather than forking the conventions.

## What the skill assumes about the server

- vramux ≥ the release with ROADMAP Stages 8–9: a waiting lease that
  outranks an idle model evicts it, and an outranked model load parks
  (`VRAMUX_QUEUE_WAIT`) instead of erroring. On an older server the skill
  still works, but the agent must evict by hand (`vramux evict <tag>`)
  before leasing, and requests arriving mid-lease fail fast instead of
  parking.
- Serving priority at its default of 7. If the operator moved it, the
  skill's priority table moves with it.

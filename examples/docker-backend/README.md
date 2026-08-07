# A worked `docker` backend

Some models will never be a GGUF — custom kernels, a patched runtime, a
quantisation llama.cpp does not implement. They still contend for the same
card, so vramux runs them the only way they ship: as a container that serves
its own OpenAI-compatible API.

This directory is a complete example of that. `docker-compose.yml` is the
container; the snippet below is what registers it.

## The contract

Four requirements, and nothing else is assumed about the image:

1. **It serves an OpenAI-compatible API on a published host port.** vramux
   talks `/v1/chat/completions` to it and translates for ollama clients.
2. **`GET /health` answers `200` when it is ready to serve**, and does not
   answer `200` before. vramux polls it after `up` and treats it as the load
   being finished — a health endpoint that goes green early turns into requests
   hanging on a model that is still loading.
3. **It answers to the id in `served_name`.** vramux rewrites the incoming
   model tag to that id.
4. **The compose file sets no `restart:` policy.** vramux stops this container
   to free the card. A restart policy fights it, and wins.

## Registering it

In `models.yml`:

```yaml
models:
  example-model:70b:
    kind: docker
    compose_file: ~/projects/vramux/examples/docker-backend/docker-compose.yml
    compose_service: example-model        # the service name inside that file
    port: 30000                           # the host side of the published port
    served_name: vendor/example-model     # what the container answers to
    ctx: 131072                           # must match --max-model-len
    idle_timeout: 3600                    # containers cost more to reload
    weights_dir: ~/models/example-model   # size reporting only
    family: example
```

`idle_timeout` is worth setting deliberately. The default is 900 s, which is
right for a GGUF that reloads in seconds and wrong for a container that takes
minutes: evicting it on the default schedule spends more time loading than
serving.

## Verifying it, in the order that isolates failures

Run the container by hand first. If it does not serve on its own, nothing
vramux does will help, and the error will arrive dressed as a router problem.

```bash
docker compose -f examples/docker-backend/docker-compose.yml up -d
curl -sf localhost:30000/health && echo healthy
curl -s localhost:30000/v1/models | python3 -m json.tool     # confirms served_name
docker compose -f examples/docker-backend/docker-compose.yml down
```

Then let vramux drive it:

```bash
vramux state                                   # what is on the card now
curl -s localhost:11434/api/chat -d '{"model":"example-model:70b","stream":false,
  "messages":[{"role":"user","content":"hi"}]}'
vramux state                                   # the container, attributed and measured
```

The second `vramux state` is the real check. The container's memory should
appear under the model's name rather than as **foreign**: NVML reports
container processes under their host PIDs, and vramux resolves ownership
through the process tree, so a correctly-registered container is recognised
without anything compose-aware.

Swapping back proves the other half:

```bash
curl -s localhost:11434/api/chat -d '{"model":"some-gguf-model:9b","stream":false,
  "messages":[{"role":"user","content":"hi"}]}'
journalctl --user -u vramux.service | grep -E "measured|unloaded|stopping"
```

## What vramux does to it, and why

- **Unload is `docker compose stop`, not `down`.** `down` removes the
  container and every warm cache with it — on the model this example is drawn
  from, that is a two-minute reload instead of about twenty-five seconds. The
  container staying around while stopped is deliberate.
- **A container left running by a previous router process is stopped at
  startup.** Restarting the router while a container held the card used to
  orphan twenty gigabytes: vramux no longer knew whose it was, and the
  container had no reason to exit. Startup reconciliation stops any
  `docker`-kind service it finds up.
- **A wedged container is recycled rather than reused.** Process up, `/health`
  red, requests hanging forever is a real failure mode. vramux health-checks a
  reused backend when nothing is in flight and restarts it if it does not
  answer.
- **The load budget is generous and bounded.** A cold container start is
  minutes; an in-progress load blocks admission for the rest of its budget plus
  slack, and then fails rather than hanging.

## Leases, if the container is not a vramux model

If a container serves something vramux does *not* register — a training job, an
image pipeline — it is not a backend at all, it is a lease holder. Take the
lease with the **host** PIDs, which `docker compose top` reports:

```bash
vramux lease --mb 12000 --owner image-stack --ttl 600 --pid "$(
  docker compose -f some-compose.yml top some-service | awk 'NR==3 {print $2}')"
```

Sending the PID matters. A holder that already allocated is charged only for
the shortfall, and a holder that does not identify itself is charged twice —
once as its own allocation, once as the grant — so the card appears to shrink.
This was measured: a 12 000 MB lease naming the host PID, against a container
already holding 980 MB, reported 11 020 MB outstanding.

Read the pid column from the header rather than by position. `docker compose
top` prefixes the service name and a replica number in this version, so a fixed
index quietly collects the replica number `1` — a plausible wrong answer, which
is the worst kind.

## Security

A `docker` entry in `models.yml` makes vramux run `docker compose` with the
file you name, and the docker group is root-equivalent on most systems.
Whoever can write either file chooses what runs as you. `SECURITY.md` says this
in full; it is the reason the registry is called a trust boundary rather than a
config file.

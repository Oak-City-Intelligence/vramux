# vramux

One ollama-compatible endpoint in front of several inference runtimes, sharing
a single GPU between them.

Local model runtimes each want the whole card, and none of them knows the
others exist. vramux sits on `:11434`, speaks ollama's HTTP API so existing
clients keep working unchanged, and owns the question of what is resident:
when a request arrives for a model that is not loaded, whatever is loaded comes
down first.

More than one model may be resident when the card can prove it fits: admission
compares a *measured* cost against the same budget leases are granted from, and
a model nobody has measured is served alone. Two kinds of backend sit behind the
same endpoint:

| kind | what it runs | for |
|---|---|---|
| `llama-server` | a llama.cpp subprocess on a local GGUF | most models |
| `docker` | a compose service that ships its own OpenAI-compatible server | models llama.cpp cannot load at all |

The second kind is the reason this is not just a llama.cpp wrapper. A model
with custom CUDA kernels on a patched runtime will never be a GGUF, but it
still contends for the same 24 GB, and something has to arbitrate.

**Status: pre-1.0, one operator's setup.** It has run one machine's inference
daily for months, which is a different thing from being ready for yours. Three
things to know before running it:

- **No authentication.** Anything that can reach the port can load models, run
  inference, and evict what is resident. It binds loopback by default; leave it
  there.
- **`models.yml` is a trust boundary.** A `docker` entry makes vramux run
  `docker compose` with the file you name, so whoever can write that config
  chooses what containers start.
- **The docker backend needs membership in the `docker` group**, which is
  root-equivalent on most systems.

`SECURITY.md` states the threat model in full, including what is deliberately
not defended.

## What it implements

- `/api/tags`, `/api/show`, `/api/chat`, `/api/generate`, `/api/embeddings`,
  `/api/ps`, `/api/version`, `/api/pull` (no-op), `/api/delete` (no-op)
- `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- ollama `keep_alive: 0` unloads; `format: "json"` becomes a GBNF grammar
- `reasoning_content` deltas surface as ollama's `thinking` field
- streaming tool calls are reassembled across deltas and emitted once
- a wedged backend (process up, `/health` red) is recycled rather than reused
- a container left running by a previous process is stopped at startup
- `GET /gpu/state` — what is resident, what is foreign, what each model cost
- `GET /gpu/events` — the same state as server-sent events, pushed on change
- `GET /gpu/console` — that stream drawn as a page, in one dependency-free file
- `POST /gpu/lease`, `DELETE /gpu/lease/{id}`, `POST /gpu/lease/{id}/renew` —
  memory reserved for consumers vramux does not run
- `POST /gpu/evict` — unload a named resident by hand

## Install

Requires Python 3.10+, `aiohttp`, `PyYAML`, and — for GGUF models — a
[llama.cpp](https://github.com/ggml-org/llama.cpp) build.

```bash
git clone git@github.com:Oak-City-Intelligence/vramux.git
cd vramux
cp models.example.yml models.yml   # then edit it
python3 -m vramux
```

There is nothing to build. `./install.sh` puts a `vramux` shim on `$PATH` —
which the client commands below need, since `python -m vramux` only resolves
from inside the checkout — and `--service` also installs the systemd user unit:

```bash
./install.sh --service
systemctl --user enable --now vramux
```

Or write the unit by hand:

```bash
mkdir -p ~/.config/systemd/user
sed "s|%CHECKOUT%|$PWD|" systemd/vramux.service > ~/.config/systemd/user/vramux.service
systemctl --user daemon-reload
systemctl --user enable --now vramux
```

## Configuration

Models are configured in `models.yml`. Start from `models.example.yml`, which
documents both backend kinds.

| Env var | Default | Meaning |
|---|---|---|
| `VRAMUX_HOST` | `127.0.0.1` | bind address |
| `VRAMUX_PORT` | `11434` | ollama-compatible port |
| `VRAMUX_UPSTREAM_PORT` | `18080` | first port llama-server backends bind; one per resident, consecutive |
| `VRAMUX_IDLE_TIMEOUT` | `900` | seconds before an idle model is unloaded |
| `VRAMUX_MODEL_DIR` | `./models` | scanned for `*.gguf` |
| `VRAMUX_MODELS_CONFIG` | `./models.yml` | registry file |
| `VRAMUX_LLAMA_SERVER_BIN` | found on `$PATH` | llama.cpp server binary |
| `VRAMUX_UPSTREAM_READ_TIMEOUT` | `300` | seconds of upstream silence before erroring |
| `VRAMUX_LOG_LEVEL` | `INFO` | python logging level |
| `VRAMUX_DEVICE` | `0` | GPU index to observe |
| `VRAMUX_CACHE_DIR` | `~/.cache/vramux` | where measured costs and usage history are written |
| `VRAMUX_RESERVE_MB` | `1024` | headroom held back from every lease |
| `VRAMUX_MAX_RESIDENTS` | `2` | ceiling on models resident at once; the budget is the real limit |
| `VRAMUX_SAMPLE_INTERVAL` | `300` | seconds between usage-history samples; clamped up to the 5 s lease sweep it rides on |
| `VRAMUX_EVENT_INTERVAL` | `1` | seconds between readings while a console is watching; nothing is read when none is |
| `VRAMUX_EVENT_KEEPALIVE` | `15` | seconds of an unchanged card before the event stream sends a comment line |

The older `MYLLAMA_*` names still work, warning once each.

Models are also discovered without configuration: any `*.gguf` under
`VRAMUX_MODEL_DIR`, and an ollama blob store if one exists on the machine.
Discovered models get an 8192-token window; set real sizes in the config's
`ctx_overrides` block.

## Seeing the card

```bash
python -m vramux state
```

```
device 0: NVIDIA GeForce RTX 4090
  total 24564 MiB   used 20149 MiB   free 3958 MiB
  recognised 18972 MiB   foreign 395 MiB   unattributed 782 MiB

       PID      MiB  OWNER              PROCESS
    482931    18972  a-container-model  model::scheduler
      2185      272  — foreign —        compositor
    283219      123  — foreign —        browser
```

**Foreign** is memory vramux did not hand out — a compositor, a browser,
someone else's training run. It is observed, subtracted from what is available,
and never reclaimed. A budget that ignores it is fiction.

**Unattributed** is the gap between what the device reports as used and what
individual processes admit to holding: driver and context overhead belonging to
no one process. It is real memory, which is why process sums alone run
optimistic.

Every managed load is measured and written to `~/.cache/vramux/costs.json`,
keyed by the configuration that determines footprint — context length included,
since the same weights at 16K and 128K differ by gigabytes. A measurement is
discarded rather than recorded when foreign usage moves during the load.

Admission reads those numbers, and only those: a model is packed beside another
when a measured or declared cost fits in what is free. There is deliberately no
estimate, because the cost model is the largest OOM risk in the design and an
underestimate takes the innocent resident down with the new one.

What the card looked like is also sampled on a timer into
`~/.cache/vramux/usage.jsonl`, bounded and appended one JSON object per line,
so foreign usage over time can be read back rather than scrolled past.

### Watching it live

```bash
vramux top
```

```
vramux  NVIDIA GeForce RTX 4090  (gpu0)   streaming
[#######################################::::                              ]
 9 807 / 24 564 MiB used (39%)   grantable 13 733   reserve 1 024   foreign 2 796

 RESIDENT                         COST   PORT  REQ   IDLE
 qwen3.5:9b-4k                   6 195  18080    0     1m

 LEASE OWNER                     GRANT     HELD     OUT   TTL
 batch-pipeline                  2 000    2 386       0   53s

 FOREIGN                           MiB   PID
 compositor                        240   2185
 browser                           170   283219
```

`q` quits. The bar draws the reserve as `:` — neither yours nor free.
**HELD** is what a leaseholder has on the card right now and **OUT** is the
grant it has not allocated against yet; a holder that has just taken a lease
is all OUT, and one still showing OUT an hour later is worth a look. HELD
above the grant is normal: a CUDA context costs a few hundred MiB that nobody
asks for, which is what the reserve is there to cover.

The console streams `/gpu/events` and falls back to polling `/gpu/state` when
it cannot — the header says which. Every watcher shares one reading of the
card, so a second console costs a socket and not a second `nvidia-smi`, and
nothing is sampled at all while nobody is watching.

`vramux top --once` prints a single frame and exits, for logs and pipes.

The same console is at **`http://localhost:11434/gpu/console`** for a browser:
one file, no build step, no fonts or scripts fetched from anywhere, because
the machine that wants this page is usually the machine with no network left.
It reads `/gpu/events` and shows exactly what the terminal view does.

## Leases

A lease reserves VRAM for something vramux does not run — an image stack, a
training script, a batch job. vramux does not start it, stop it or reach into
it. It only promises that the memory stays available, and refuses to promise
the same memory to anybody else.

```bash
vramux lease --mb 18000 --owner batch-pipeline -- ./stage2.sh
```

Acquire, run, renew in the background, release on the way out. That is the
whole adoption story: the correct version has to be shorter than the poll loop
it replaces.

```bash
vramux leases          # what is held right now
vramux free --mb 8000  # wait until 8 GB could be granted, then exit
vramux evict some:tag  # unload a resident model by hand
```

The rules worth knowing before writing a client:

- **TTL is mandatory and there is no infinite lease.** Renewal is a heartbeat.
  A holder killed with `SIGKILL` runs no cleanup by definition, so server-side
  expiry — not the wrapper — is what stops a dead holder stranding the card.
  Expiry logs loudly: a lease that expires under a live holder is a bug in that
  holder.
- **`413` and `408` mean different things.** `413` is "more than this card can
  ever provide", which is a configuration error and fails immediately. `408` is
  "not within the time you gave me", after actually waiting.
- **Send your `pid`.** A holder that already has memory on the card — because
  it allocated before asking, or because vramux restarted underneath it — is
  charged only for the shortfall. Attribution follows the process tree, so the
  command a wrapper runs counts as the wrapper.
- **A restart drops every lease and frees nothing.** Holders demote to foreign:
  vramux stops knowing whose memory that is, still sees it, and still subtracts
  it. The budget stays true, which is the invariant that matters.

Nothing needs a lease to be served a model. Consumers that have not migrated
are foreign, which is a correct state and not a broken one.

## Layout

```
vramux/
  __main__.py     entry point — `python -m vramux`
  env.py          settings, with the old variable names shimmed
  nvml.py         reading the device — totals and per-process usage
  observer.py     attribution, measurement, the cost cache, usage history
  budget.py       how much there is to hand out — pure arithmetic
  lease.py        the broker: grants, expiry, reclaim by process tree
  cli.py          the client side — `lease`, `free`, `evict`, stdlib only
  console.py      `vramux top` — a pure renderer and an SSE reader, stdlib only
  console.html    the same console as a page, served at /gpu/console
  registry.py     ModelSpec, YAML config, dir scan, ollama-blob discovery
  backends.py     the Backend contract: ProcessBackend + DockerComposeBackend
  residency.py    who is on the card: admission, eviction, drain, idle
  translate.py    OpenAI <-> ollama wire format
  router.py       aiohttp routes
tests/            GPU-less; no card, no llama.cpp, no docker required
examples/
  docker-backend/ a complete container backend, and the contract it satisfies
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Everything runs against fakes, so the suite needs no GPU and no model weights.

## Smoke test

```bash
curl -s localhost:11434/api/tags | jq '.models[].name'

curl -s localhost:11434/api/chat -d '{
  "model":"your-tag:9b","stream":false,
  "messages":[{"role":"user","content":"hi"}]}'
```

## Where it is going

`DESIGN.md` and `ROADMAP.md` describe the actual destination: a VRAM broker
that leases memory to any consumer on the machine, not just to model serving.
Serving is its first client.

Leases exist, the budget is honest about the card, and serving now draws on it:
a second model is admitted when its measured cost fits in what is free, and a
leaseholder taking memory is enough to keep it out. What is deliberately absent
is estimation — a model with no measured or declared cost gets the card to
itself, because an underestimate is an OOM and an OOM on a shared card can take
the innocent resident down with it.

Serving does not take leases for its own residents. It does not need to: the
budget is anchored on what the device reports, so a resident is accounted for
whether or not anybody wrote it down.

What is still missing is cooperative eviction — asking a leaseholder to yield.
Today a lease is never revoked, so a big model waits for one to expire rather
than negotiating.

## Contributing

`CONTRIBUTING.md`, and `DESIGN.md` before it. The two arguments most worth
reading first are why the budget is anchored on what the device reports, and
why admission refuses to estimate a cost it has not measured.

## License

Apache-2.0. See `LICENSE`.

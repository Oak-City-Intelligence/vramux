# vramux

One ollama-compatible endpoint in front of several inference runtimes, sharing
a single GPU between them.

Local model runtimes each want the whole card, and none of them knows the
others exist. vramux sits on `:11434`, speaks ollama's HTTP API so existing
clients keep working unchanged, and owns the question of what is resident:
when a request arrives for a model that is not loaded, whatever is loaded comes
down first.

Today that means **one model resident at a time**, and two kinds of backend
behind the same endpoint:

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

## Install

Requires Python 3.10+, `aiohttp`, `PyYAML`, and — for GGUF models — a
[llama.cpp](https://github.com/ggml-org/llama.cpp) build.

```bash
git clone git@github.com:Oak-City-Intelligence/vramux.git
cd vramux
cp models.example.yml models.yml   # then edit it
python3 -m vramux
```

As a user service:

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
| `VRAMUX_UPSTREAM_PORT` | `18080` | port llama-server backends bind |
| `VRAMUX_IDLE_TIMEOUT` | `900` | seconds before an idle model is unloaded |
| `VRAMUX_MODEL_DIR` | `./models` | scanned for `*.gguf` |
| `VRAMUX_MODELS_CONFIG` | `./models.yml` | registry file |
| `VRAMUX_LLAMA_SERVER_BIN` | found on `$PATH` | llama.cpp server binary |
| `VRAMUX_UPSTREAM_READ_TIMEOUT` | `300` | seconds of upstream silence before erroring |
| `VRAMUX_LOG_LEVEL` | `INFO` | python logging level |
| `VRAMUX_DEVICE` | `0` | GPU index to observe |
| `VRAMUX_CACHE_DIR` | `~/.cache/vramux` | where measured costs are written |

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

Nothing reads those numbers to make a decision yet. That is deliberate: the
cost model is the largest OOM risk in the design, and it should be answering
from real measurements before anything depends on it.

## Layout

```
vramux/
  __main__.py     entry point — `python -m vramux`
  env.py          settings, with the old variable names shimmed
  nvml.py         reading the device — totals and per-process usage
  observer.py     attribution, measurement, the cost cache
  registry.py     ModelSpec, YAML config, dir scan, ollama-blob discovery
  supervisor.py   the GPU slot: ProcessBackend + DockerComposeBackend
  router.py       aiohttp routes + OpenAI <-> ollama translation
tests/            GPU-less; no card, no llama.cpp, no docker required
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
Serving is its first client. Multi-residency, leases and eviction are not
implemented yet.

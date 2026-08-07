# llama-router

Tiny Python service that exposes [llama.cpp](https://github.com/ggerganov/llama.cpp)
behind **ollama's HTTP API surface** on port 11434.

The point: every existing local-LLM client on the machine (a client, an editor client,
the captioning tool, the batch pipeline, anything speaking `OLLAMA_HOST`) keeps working
unchanged. llama.cpp is the inference engine; ollama as a daemon is gone.

## What it does

- `/api/tags`, `/api/show`, `/api/chat`, `/api/generate`, `/api/embeddings`,
  `/api/ps`, `/api/version`, `/api/pull` (no-op), `/api/delete` (no-op) —
  ollama-shape responses
- `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` — OpenAI
  passthrough
- One backend loaded at a time, auto-swapped when a different model is
  requested, auto-unloaded after `MYLLAMA_IDLE_TIMEOUT` seconds. Two backend
  kinds share that single GPU slot:
  - `llama-server` (default) — a supervised subprocess on a local GGUF
  - `docker` — a compose service that ships its own OpenAI-compatible server,
    for models llama.cpp cannot load at all
- Translates ollama `keep_alive: 0` to an unload, `format: "json"` to GBNF
  grammar (and strips `<think>` blocks for thinking models)
- Forwards `reasoning_content` deltas as ollama `thinking` field

## Files

```
llama_router/
  __init__.py
  __main__.py     entry point — `python -m llama_router`
  registry.py     ModelSpec + YAML + dir-scan + ollama-blob fallback
  supervisor.py   GPU-slot lifecycle: ProcessBackend + DockerComposeBackend
  router.py       aiohttp routes + OpenAI<->ollama translation
```

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `MYLLAMA_HOST` | `127.0.0.1` | bind address |
| `MYLLAMA_PORT` | `11434` | ollama-compat port |
| `MYLLAMA_UPSTREAM_PORT` | `18080` | llama-server bind port |
| `MYLLAMA_IDLE_TIMEOUT` | `900` | seconds before unloading idle model |
| `MYLLAMA_MODEL_DIR` | `<model-dir>` | scanned for `*.gguf` |
| `MYLLAMA_LOG_LEVEL` | `INFO` | python logging level |
| `LLAMA_SERVER_BIN` | `<llama.cpp-build>/llama-server` | binary path |

The model registry YAML lives at
`<console-dir>config/the models config` *(legacy path
— move along with the rest of the cleanup; the loader reads
`Path.home() / "control/stacks/console-dir/config/the models config"`,
override by passing `config_file=` to `ModelRegistry()`)*.

Each entry:

```yaml
models:
  my-tag:9b:
    gguf: <model-dir>/some.gguf
    ctx: 16384
    family: my-tag
    n_gpu_layers: 999    # optional; omit to let llama-server auto-fit
    extra_args:          # optional
      - --no-jinja
      - --chat-template
      - command-r
```

## Run it

```bash
# one-off
cd <old-checkout>
python3 -m llama_router

# as a systemd user unit
systemctl --user start the old unit
systemctl --user status the old unit
```

## Smoke test

```bash
curl -s http://127.0.0.1:11434/api/tags | jq '.models[].name'

curl -s -X POST http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

## Built

Sep–May 2026 during the ollama → llama.cpp migration. See
`baseline-2026-05-26.md` in the config dir for the post-migration throughput
numbers.

### docker-kind entries

For a model that is not a GGUF — its own runtime, its own server — register the
compose service instead of a blob:

```yaml
models:
  container-model:35b:
    kind: docker
    compose_file: <container-project>
    compose_service: container-model
    port: 30000                        # host port the container publishes
    served_name: container-model-35b-a3b-w2   # model id the container answers to
    ctx: 131072
    idle_timeout: 3600                 # override; containers are costlier to reload
    weights_dir: <weights-dir>   # size reporting only
```

The supervisor runs `docker compose up -d <service>` to load and
`docker compose stop <service>` to unload — `stop`, not `down`, so the container
and its warm JIT/kernel caches survive (29 s reload vs. a ~2 min cold build for
the container-model image). The compose file must not set a `restart:` policy, or docker
will fight the router for the slot.

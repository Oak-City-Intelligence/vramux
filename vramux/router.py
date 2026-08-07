"""Ollama-compatible HTTP router on top of a llama-server supervisor.

Implements the subset of ollama's REST API that the callers on this machine
actually use: /api/tags, /api/show, /api/chat, /api/generate, /api/embeddings,
plus pass-through for /v1/chat/completions etc.

Responses are translated between OpenAI (what llama-server emits) and the
ollama JSON-lines shape that existing clients expect.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp
from aiohttp import web

from . import env
from .registry import ModelRegistry, ModelSpec
from .supervisor import LlamaServerSupervisor

log = logging.getLogger("vramux.router")


# Longest gap tolerated between bytes from an upstream server. Generous enough
# for a long prefill on a big context, short enough that a wedged backend
# surfaces as an error instead of an unbounded hang.
UPSTREAM_READ_TIMEOUT = env.get_float("UPSTREAM_READ_TIMEOUT", 300.0)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Router:
    def __init__(self, registry: ModelRegistry, supervisor: LlamaServerSupervisor) -> None:
        self.registry = registry
        self.supervisor = supervisor
        self._client_session: Optional[aiohttp.ClientSession] = None

    async def startup(self, _app: web.Application) -> None:
        # No total timeout — a long generation is legitimate — but a read
        # timeout is essential: a wedged upstream that accepts the connection
        # and then never writes would otherwise hang the caller forever.
        self._client_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=UPSTREAM_READ_TIMEOUT)
        )
        await self.supervisor.reconcile(self.registry.all())

    async def cleanup(self, _app: web.Application) -> None:
        if self._client_session:
            await self._client_session.close()
        await self.supervisor.stop()

    # ---- ollama API surface ---------------------------------------------------

    async def health(self, _request: web.Request) -> web.Response:
        return web.Response(text="Ollama is running")

    async def list_tags(self, _request: web.Request) -> web.Response:
        models = [spec.to_ollama_tag_entry() for spec in self.registry.all()]
        return web.json_response({"models": models})

    async def models(self, _request: web.Request) -> web.Response:
        """OpenAI-compatible `/v1/models` — lets a client verify the endpoint and
        enumerate available models. ollama exposes this too."""
        data = [
            {"id": spec.tag, "object": "model", "created": 0, "owned_by": "vramux"}
            for spec in self.registry.all()
        ]
        # Synthetic "auto" id: resolves to whatever model is currently loaded.
        # Lets side-task callers (auxiliary agent tasks) target the live model
        # without forcing a swap, and keeps endpoint-verification probes happy.
        data.append({"id": "auto", "object": "model", "created": 0, "owned_by": "vramux"})
        return web.json_response({"object": "list", "data": data})

    async def show(self, request: web.Request) -> web.Response:
        body = await request.json()
        name = body.get("name") or body.get("model") or ""
        spec = self._resolve_spec(name)
        if not spec:
            return web.json_response({"error": f"model '{name}' not found"}, status=404)
        return web.json_response({
            "modelfile": f"FROM {spec.gguf_path or spec.weights_dir or spec.tag}\n",
            "parameters": f"num_ctx {spec.ctx_size}\n",
            "template": "",
            "details": spec.to_ollama_tag_entry()["details"],
            "model_info": {"general.architecture": spec.family or ""},
        })

    async def version(self, _request: web.Request) -> web.Response:
        return web.json_response({"version": "vramux-1"})

    async def ps(self, _request: web.Request) -> web.Response:
        """Mimic ollama's `/api/ps` — what model (if any) is currently loaded."""
        cur = self.supervisor.current_spec
        if not cur:
            return web.json_response({"models": []})
        entry = cur.to_ollama_tag_entry()
        entry["size_vram"] = entry["size"]
        entry["expires_at"] = "0001-01-01T00:00:00Z"
        return web.json_response({"models": [entry]})

    async def gpu_state(self, _request: web.Request) -> web.Response:
        """What is on the card right now, with ownership resolved.

        Read-only, and it is the only place ownership can be resolved: the
        observer that knows which PIDs vramux started lives in this process.
        A CLI asking from outside would see every process as foreign.
        """
        observer = self.supervisor.observer
        if observer is None:
            return web.json_response({"error": "observation disabled"}, status=503)
        snap = await observer.snapshot()
        if snap is None:
            return web.json_response({"error": "no GPU visible"}, status=503)
        return web.json_response({
            "device": {
                "index": snap.device.index,
                "name": snap.device.name,
                "total_mb": snap.device.total_mb,
                "used_mb": snap.device.used_mb,
                "free_mb": snap.device.free_mb,
                "unattributed_mb": snap.device.unattributed_mb,
            },
            "recognised_mb": snap.recognised_mb,
            "foreign_mb": snap.foreign_mb,
            "processes": [
                {"pid": a.process.pid, "used_mb": a.process.used_mb,
                 "name": a.process.name, "owner": a.owner}
                for a in snap.attributions
            ],
            "unlocated_owners": snap.unlocated_owners,
            "costs": observer.cache.all(),
        })

    async def pull(self, _request: web.Request) -> web.Response:
        # Models are managed out-of-band (GGUFs are local files). No-op success.
        return web.json_response({"status": "success"})

    async def delete(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "success"})

    def _resolve_spec(self, model: str):
        """Map a request's model name to a ModelSpec.

        The sentinel ``"auto"`` means "serve with whatever model is currently
        loaded, without forcing a swap." Auxiliary/side-task callers
        (compression, title-gen, session-search) use it so they ride the live
        model instead of thrashing the single GPU slot. When the server is idle
        (nothing loaded) it falls back to the first registered model — which is
        the small/fast one by convention, a sensible default for side tasks."""
        if model == "auto":
            cur = self.supervisor.current_spec
            if cur is not None:
                return cur
            allspecs = self.registry.all()
            return allspecs[0] if allspecs else None
        return self.registry.get(model)

    async def chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        model = body.get("model", "")
        spec = self._resolve_spec(model)
        if not spec:
            return web.json_response({"error": f"model '{model}' not found"}, status=404)
        if _is_unload_request(body):
            return await self._handle_unload(model)
        await self.supervisor.acquire(spec)
        try:
            stream = bool(body.get("stream", True))
            upstream_payload = {
                "model": spec.served_name,
                "messages": body.get("messages", []),
                "stream": stream,
                **_ollama_options_to_openai(body.get("options") or {}),
            }
            if body.get("format") == "json":
                upstream_payload["response_format"] = {"type": "json_object"}
            # ollama's `think: false` disables qwen3-style thinking mode. llama-server
            # exposes this via the chat template's `enable_thinking` kwarg (qwen3
            # templates honor it); forward unconditionally when the caller asks.
            if body.get("think") is False:
                upstream_payload["chat_template_kwargs"] = {"enable_thinking": False}
            tools = body.get("tools")
            if tools:
                upstream_payload["tools"] = tools

            upstream_url = f"{self.supervisor.upstream}/v1/chat/completions"
            return await self._proxy_openai_chat(request, upstream_url, upstream_payload, spec.tag, stream)
        finally:
            self.supervisor.release()

    async def generate(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        model = body.get("model", "")
        spec = self._resolve_spec(model)
        if not spec:
            return web.json_response({"error": f"model '{model}' not found"}, status=404)
        if _is_unload_request(body):
            return await self._handle_unload(model)
        await self.supervisor.acquire(spec)
        try:
            stream = bool(body.get("stream", True))
            want_json = body.get("format") == "json"
            upstream_payload = {
                "model": spec.served_name,
                "prompt": body.get("prompt", ""),
                "stream": stream,
                **_ollama_options_to_openai(body.get("options") or {}),
            }
            if want_json:
                # /v1/completions doesn't accept response_format; constrain via GBNF.
                upstream_payload["grammar"] = _JSON_GRAMMAR
            upstream_url = f"{self.supervisor.upstream}/v1/completions"
            return await self._proxy_openai_generate(
                request, upstream_url, upstream_payload, spec.tag, stream, strip_thinking=want_json,
            )
        finally:
            self.supervisor.release()

    async def embeddings(self, request: web.Request) -> web.Response:
        body = await request.json()
        model = body.get("model", "")
        spec = self._resolve_spec(model)
        if not spec:
            return web.json_response({"error": f"model '{model}' not found"}, status=404)
        await self.supervisor.acquire(spec)
        try:
            inp = body.get("prompt") or body.get("input") or ""
            upstream_payload = {"model": spec.served_name, "input": inp}
            assert self._client_session is not None
            async with self._client_session.post(f"{self.supervisor.upstream}/v1/embeddings", json=upstream_payload) as resp:
                data = await resp.json()
            # ollama shape: {"embedding": [...]}  (single)  or {"embeddings": [[...]]}
            if isinstance(data.get("data"), list) and data["data"]:
                first = data["data"][0].get("embedding", [])
                return web.json_response({"embedding": first, "embeddings": [d.get("embedding", []) for d in data["data"]]})
            return web.json_response({"embedding": [], "embeddings": []})
        finally:
            self.supervisor.release()

    async def _handle_unload(self, model: str) -> web.Response:
        """Unload the currently-loaded model. Returns ollama's shape."""
        await self.supervisor.stop()
        return web.json_response({
            "model": model,
            "created_at": _ts(),
            "response": "",
            "done": True,
            "done_reason": "unload",
        })

    # ---- OpenAI passthrough (some callers use /v1 directly) -------------------

    async def openai_passthrough(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        model = body.get("model", "")
        spec = self._resolve_spec(model)
        if not spec:
            return web.json_response({"error": f"model '{model}' not found"}, status=404)
        await self.supervisor.acquire(spec)
        try:
            body["model"] = spec.served_name

            upstream_url = f"{self.supervisor.upstream}{request.path}"
            assert self._client_session is not None
            async with self._client_session.post(upstream_url, json=body) as upstream:
                resp = web.StreamResponse(status=upstream.status, headers={"Content-Type": upstream.headers.get("Content-Type", "application/json")})
                await resp.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        finally:
            self.supervisor.release()

    # ---- helpers -------------------------------------------------------------

    async def _proxy_openai_chat(
        self,
        request: web.Request,
        upstream_url: str,
        payload: Dict[str, Any],
        model_tag: str,
        stream: bool,
    ) -> web.StreamResponse:
        assert self._client_session is not None
        if not stream:
            async with self._client_session.post(upstream_url, json=payload) as upstream:
                upstream_data = await upstream.json()
            return web.json_response(_openai_chat_to_ollama(upstream_data, model_tag))

        resp = web.StreamResponse(status=200, headers={"Content-Type": "application/x-ndjson"})
        await resp.prepare(request)
        async with self._client_session.post(upstream_url, json=payload) as upstream:
            async for ollama_line in _sse_chat_to_ollama_lines(upstream.content, model_tag):
                await resp.write(ollama_line)
        await resp.write_eof()
        return resp

    async def _proxy_openai_generate(
        self,
        request: web.Request,
        upstream_url: str,
        payload: Dict[str, Any],
        model_tag: str,
        stream: bool,
        strip_thinking: bool = False,
    ) -> web.StreamResponse:
        assert self._client_session is not None
        if not stream:
            async with self._client_session.post(upstream_url, json=payload) as upstream:
                upstream_data = await upstream.json()
            text = "".join(c.get("text", "") for c in upstream_data.get("choices", []))
            if strip_thinking:
                text = _strip_thinking(text)
            return web.json_response({
                "model": model_tag,
                "created_at": _ts(),
                "response": text,
                "done": True,
                "done_reason": "stop",
            })

        resp = web.StreamResponse(status=200, headers={"Content-Type": "application/x-ndjson"})
        await resp.prepare(request)
        async with self._client_session.post(upstream_url, json=payload) as upstream:
            async for ollama_line in _sse_generate_to_ollama_lines(upstream.content, model_tag):
                await resp.write(ollama_line)
        await resp.write_eof()
        return resp


# Standard GBNF for any valid JSON value — adapted from llama.cpp/grammars/json.gbnf.
_JSON_GRAMMAR = r'''
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [1-9] [0-9]{0,15})? ws

ws ::= | " " | "\n" [ \t]{0,20}
'''.strip()

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks so JSON-format consumers get only data."""
    return _THINK_BLOCK.sub("", text).lstrip()


def _is_unload_request(body: Dict[str, Any]) -> bool:
    """ollama convention: `keep_alive: 0` with empty prompt/messages means
    'unload the model now'. A batch pipeline uses this between stages to hand
    the card to other GPU work."""
    keep_alive = body.get("keep_alive", None)
    if keep_alive in (0, "0", "0s"):
        prompt_empty = not body.get("prompt") and not body.get("messages")
        if prompt_empty:
            return True
    return False


def _ollama_options_to_openai(opts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "temperature" in opts:
        out["temperature"] = opts["temperature"]
    if "top_p" in opts:
        out["top_p"] = opts["top_p"]
    if "num_predict" in opts:
        out["max_tokens"] = opts["num_predict"]
    if "stop" in opts:
        out["stop"] = opts["stop"]
    if "seed" in opts:
        out["seed"] = opts["seed"]
    # llama.cpp accepts these as OpenAI-extension fields; ollama exposed them
    # as Modelfile parameters that bound qwen3 sampling under control. Forward
    # them so callers can restore the per-model discipline ollama gave us.
    if "top_k" in opts:
        out["top_k"] = opts["top_k"]
    if "min_p" in opts:
        out["min_p"] = opts["min_p"]
    if "repeat_penalty" in opts:
        out["repeat_penalty"] = opts["repeat_penalty"]
    return out


def _openai_tool_calls_to_ollama(calls: Any) -> List[Dict[str, Any]]:
    """OpenAI tool_calls -> ollama shape.

    The one real difference: OpenAI serializes `arguments` as a JSON *string*,
    ollama clients expect an already-parsed object. Malformed argument JSON is
    passed through under a `_raw` key rather than dropped, so a caller can see
    what the model actually emitted.
    """
    out: List[Dict[str, Any]] = []
    for call in calls or []:
        fn = call.get("function") or {}
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
        else:
            args = raw_args or {}
        out.append({"function": {"name": fn.get("name", ""), "arguments": args}})
    return out


def _openai_chat_to_ollama(data: Dict[str, Any], model_tag: str) -> Dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    out_msg: Dict[str, Any] = {
        "role": msg.get("role", "assistant"),
        "content": msg.get("content", "") or "",
    }
    # llama-server splits reasoning ("thinking") output into its own field for
    # qwen3/r1-style models — surface it as ollama's "thinking" plus keep it
    # in content so old clients still see something.
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if reasoning:
        out_msg["thinking"] = reasoning
        if not out_msg["content"]:
            out_msg["content"] = reasoning
    tool_calls = _openai_tool_calls_to_ollama(msg.get("tool_calls"))
    if tool_calls:
        out_msg["tool_calls"] = tool_calls
        # A tool-calling turn legitimately has empty content; do not let the
        # reasoning fallback above stuff the thinking text into it, or clients
        # render the model's scratchpad as the assistant's answer.
        if reasoning and not (msg.get("content") or ""):
            out_msg["content"] = ""
    return {
        "model": model_tag,
        "created_at": _ts(),
        "message": out_msg,
        "done": True,
        "done_reason": choice.get("finish_reason", "stop") or "stop",
        "total_duration": 0,
        "prompt_eval_count": (data.get("usage") or {}).get("prompt_tokens", 0),
        "eval_count": (data.get("usage") or {}).get("completion_tokens", 0),
    }


class _ToolCallAccumulator:
    """Reassembles tool calls that arrive split across SSE deltas.

    OpenAI streams a call as fragments keyed by `index`: the first carries the
    name with empty arguments, then the argument JSON dribbles in piece by
    piece (`{`, `"city": "Braga"`, `}`). Only the concatenation is parseable, so
    fragments are buffered and emitted once, at finish — which is also where
    ollama clients expect tool calls to appear.
    """

    def __init__(self) -> None:
        self._calls: Dict[int, Dict[str, str]] = {}

    def feed(self, deltas: Any) -> None:
        for call in deltas or []:
            idx = call.get("index", 0)
            slot = self._calls.setdefault(idx, {"name": "", "arguments": ""})
            fn = call.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            frag = fn.get("arguments")
            if isinstance(frag, str):
                slot["arguments"] += frag

    def drain(self) -> List[Dict[str, Any]]:
        out = _openai_tool_calls_to_ollama([
            {"function": {"name": slot["name"], "arguments": slot["arguments"]}}
            for _, slot in sorted(self._calls.items())
        ])
        self._calls.clear()
        return out


async def _sse_chat_to_ollama_lines(stream: aiohttp.StreamReader, model_tag: str) -> AsyncIterator[bytes]:
    tools = _ToolCallAccumulator()
    emitted_done = False
    async for raw in _iter_sse(stream):
        if raw == "[DONE]":
            # A finish_reason chunk already closed the stream. Emitting a second
            # terminal line here would overwrite it with an empty `stop`, hiding
            # the tool calls from any client that reads the last message.
            if emitted_done:
                return
            done_msg: Dict[str, Any] = {"role": "assistant", "content": ""}
            # Upstreams that end the stream with [DONE] and no finish_reason
            # chunk would otherwise drop the buffered call entirely.
            pending = tools.drain()
            if pending:
                done_msg["tool_calls"] = pending
            done = {
                "model": model_tag,
                "created_at": _ts(),
                "message": done_msg,
                "done": True,
                "done_reason": "tool_calls" if pending else "stop",
            }
            yield (json.dumps(done) + "\n").encode()
            return
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        choice = (obj.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        finish = choice.get("finish_reason")
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        content = delta.get("content") or ""
        message: Dict[str, Any] = {
            "role": delta.get("role") or "assistant",
            "content": content,
        }
        if reasoning:
            message["thinking"] = reasoning
        tools.feed(delta.get("tool_calls"))
        if finish:
            pending = tools.drain()
            if pending:
                message["tool_calls"] = pending
        ollama_msg: Dict[str, Any] = {
            "model": model_tag,
            "created_at": _ts(),
            "message": message,
            "done": bool(finish),
        }
        if finish:
            ollama_msg["done_reason"] = finish
            emitted_done = True
        yield (json.dumps(ollama_msg) + "\n").encode()


async def _sse_generate_to_ollama_lines(stream: aiohttp.StreamReader, model_tag: str) -> AsyncIterator[bytes]:
    async for raw in _iter_sse(stream):
        if raw == "[DONE]":
            yield (json.dumps({
                "model": model_tag,
                "created_at": _ts(),
                "response": "",
                "done": True,
                "done_reason": "stop",
            }) + "\n").encode()
            return
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        choice = (obj.get("choices") or [{}])[0]
        text = choice.get("text", "") or ""
        finish = choice.get("finish_reason")
        out = {
            "model": model_tag,
            "created_at": _ts(),
            "response": text,
            "done": bool(finish),
        }
        if finish:
            out["done_reason"] = finish
        yield (json.dumps(out) + "\n").encode()


async def _iter_sse(stream: aiohttp.StreamReader) -> AsyncIterator[str]:
    async for raw_line in stream:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line or not line.startswith("data:"):
            continue
        yield line[5:].strip()


def make_app(registry: ModelRegistry, supervisor: LlamaServerSupervisor) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    r = Router(registry, supervisor)
    app.on_startup.append(r.startup)
    app.on_cleanup.append(r.cleanup)
    app.router.add_get("/", r.health)
    app.router.add_get("/api/version", r.version)
    app.router.add_get("/api/tags", r.list_tags)
    app.router.add_get("/api/ps", r.ps)
    app.router.add_get("/gpu/state", r.gpu_state)
    app.router.add_post("/api/show", r.show)
    app.router.add_post("/api/chat", r.chat)
    app.router.add_post("/api/generate", r.generate)
    app.router.add_post("/api/embeddings", r.embeddings)
    app.router.add_post("/api/embed", r.embeddings)
    app.router.add_post("/api/pull", r.pull)
    app.router.add_delete("/api/delete", r.delete)
    app.router.add_get("/v1/models", r.models)
    app.router.add_post("/v1/chat/completions", r.openai_passthrough)
    app.router.add_post("/v1/completions", r.openai_passthrough)
    app.router.add_post("/v1/embeddings", r.openai_passthrough)
    return app

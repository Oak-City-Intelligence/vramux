"""Ollama-compatible HTTP router on top of the residency arbiter.

Implements the subset of ollama's REST API that the callers on this machine
actually use: /api/tags, /api/show, /api/chat, /api/generate, /api/embeddings,
plus pass-through for /v1/chat/completions etc.

Routing only. The OpenAI↔ollama wire translation lives in `translate.py`; what
is resident on the card is `residency.py`'s business. This file's job is to
turn a request into (spec, upstream URL, payload) and stream the answer back.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import aiohttp
from aiohttp import web

from . import env
from .lease import (
    DEFAULT_TTL,
    Broker,
    CardUnreadable,
    InvalidRequest,
    LeaseError,
    NoRoom,
    NotGrantable,
    UnknownLease,
)
from .registry import ModelRegistry
from .residency import BackendLoading, ResidencyArbiter
from .translate import (
    _JSON_GRAMMAR,
    _is_unload_request,
    _ollama_options_to_openai,
    _openai_chat_to_ollama,
    _sse_chat_to_ollama_lines,
    _sse_generate_to_ollama_lines,
    _strip_thinking,
    _ts,
)

log = logging.getLogger("vramux.router")


# Longest gap tolerated between bytes from an upstream server. Generous enough
# for a long prefill on a big context, short enough that a wedged backend
# surfaces as an error instead of an unbounded hang.
UPSTREAM_READ_TIMEOUT = env.get_float("UPSTREAM_READ_TIMEOUT", 300.0)


class Router:
    def __init__(
        self,
        registry: ModelRegistry,
        arbiter: ResidencyArbiter,
        broker: Optional[Broker] = None,
    ) -> None:
        self.registry = registry
        self.arbiter = arbiter
        self.broker = broker
        self._client_session: Optional[aiohttp.ClientSession] = None

    async def startup(self, _app: web.Application) -> None:
        # No total timeout — a long generation is legitimate — but a read
        # timeout is essential: a wedged upstream that accepts the connection
        # and then never writes would otherwise hang the caller forever.
        self._client_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=UPSTREAM_READ_TIMEOUT)
        )
        await self.arbiter.reconcile(self.registry.all())
        if self.broker is not None:
            # Expiry only becomes a guarantee once something is sweeping.
            self.broker.start()

    async def cleanup(self, _app: web.Application) -> None:
        if self._client_session:
            await self._client_session.close()
        if self.broker is not None:
            await self.broker.close()
        await self.arbiter.stop()

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
        """Mimic ollama's `/api/ps` — every model currently loaded.

        A list because admission is no longer one: reporting only the hottest
        resident would tell a client the card is free of a model that is very
        much on it.
        """
        models = []
        for spec in self.arbiter.current_specs:
            entry = spec.to_ollama_tag_entry()
            entry["size_vram"] = entry["size"]
            entry["expires_at"] = "0001-01-01T00:00:00Z"
            models.append(entry)
        return web.json_response({"models": models})

    async def gpu_state(self, _request: web.Request) -> web.Response:
        """What is on the card right now, with ownership resolved.

        Read-only, and it is the only place ownership can be resolved: the
        observer that knows which PIDs vramux started lives in this process.
        A CLI asking from outside would see every process as foreign.
        """
        observer = self.arbiter.observer
        if observer is None:
            return web.json_response({"error": "observation disabled"}, status=503)
        snap = await observer.snapshot()
        if snap is None:
            return web.json_response({"error": "no GPU visible"}, status=503)
        leases = []
        budget = None
        if self.broker is not None:
            leases = await self.broker.views(snap)
            budget = (await self.broker.budget(snap)).to_json()
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
            "residents": [r.tag for r in self.arbiter.residents],
            # A load in progress, so a slow cold container reads as a wait
            # rather than as a card that has stopped answering.
            "loading": self.arbiter.loading,
            "costs": observer.cache.all(),
            # Grants outstanding, and the arithmetic behind them. Residents are
            # accounted for by what the device reports them using, not by a
            # lease — nothing requires a grant to serve a model yet.
            "leases": leases,
            "budget": budget,
        })

    # ---- leases ---------------------------------------------------------------

    async def lease_acquire(self, request: web.Request) -> web.Response:
        """Grant memory, or say honestly why not.

        `413` is "never, on this card" and `408` is "not within the time you
        gave me". Conflating them would make a misconfigured request wait two
        minutes before failing the way it was always going to.
        """
        if self.broker is None:
            return web.json_response({"error": "leasing disabled"}, status=503)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "body must be JSON"}, status=400)
        pids = body.get("pids") or []
        if body.get("pid") is not None:
            pids = [body["pid"], *pids]
        try:
            lease = await self.broker.acquire(
                mb=body.get("mb"),
                owner=body.get("owner", ""),
                ttl=body.get("ttl", DEFAULT_TTL),
                priority=int(body.get("priority", 5)),
                pids=pids,
                wait=float(body.get("wait", 0) or 0),
            )
        except LeaseError as exc:
            return _lease_error(exc)
        except (TypeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(await self.broker.view(lease))

    async def lease_release(self, request: web.Request) -> web.Response:
        if self.broker is None:
            return web.json_response({"error": "leasing disabled"}, status=503)
        try:
            lease = await self.broker.release(request.match_info["lease_id"])
        except LeaseError as exc:
            return _lease_error(exc)
        return web.json_response({"lease": lease.id, "released": True})

    async def lease_renew(self, request: web.Request) -> web.Response:
        if self.broker is None:
            return web.json_response({"error": "leasing disabled"}, status=503)
        ttl = None
        if request.can_read_body:
            try:
                ttl = (await request.json()).get("ttl")
            except Exception:
                ttl = None
        try:
            lease = await self.broker.renew(request.match_info["lease_id"], ttl)
        except LeaseError as exc:
            return _lease_error(exc)
        return web.json_response(await self.broker.view(lease))

    async def lease_list(self, _request: web.Request) -> web.Response:
        if self.broker is None:
            return web.json_response({"error": "leasing disabled"}, status=503)
        try:
            return web.json_response({"leases": await self.broker.views()})
        except LeaseError as exc:
            return _lease_error(exc)

    async def evict(self, request: web.Request) -> web.Response:
        """Drop a named resident by hand. Idle models unload themselves; this
        is for the times you want the card back now."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "body must be JSON"}, status=400)
        tag = body.get("tag") or ""
        if not tag:
            return web.json_response({"error": "tag is required"}, status=400)
        evicted = await self.arbiter.evict(tag)
        if not evicted:
            return web.json_response({"error": f"{tag} is not resident"}, status=404)
        return web.json_response({"tag": tag, "evicted": True})

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
        model instead of thrashing the card. When the server is idle (nothing
        loaded) it falls back to the first registered model — which is the
        small/fast one by convention, a sensible default for side tasks."""
        if model == "auto":
            cur = self.arbiter.current_spec
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
            return await self._handle_unload(model, spec)
        try:
            await self.arbiter.acquire(spec)
        except BackendLoading as exc:
            return _still_loading(exc)
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

            upstream_url = f"{self.arbiter.upstream_for(spec.tag)}/v1/chat/completions"
            return await self._proxy_openai_chat(request, upstream_url, upstream_payload, spec.tag, stream)
        finally:
            self.arbiter.release(spec.tag)

    async def generate(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        model = body.get("model", "")
        spec = self._resolve_spec(model)
        if not spec:
            return web.json_response({"error": f"model '{model}' not found"}, status=404)
        if _is_unload_request(body):
            return await self._handle_unload(model, spec)
        try:
            await self.arbiter.acquire(spec)
        except BackendLoading as exc:
            return _still_loading(exc)
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
            upstream_url = f"{self.arbiter.upstream_for(spec.tag)}/v1/completions"
            return await self._proxy_openai_generate(
                request, upstream_url, upstream_payload, spec.tag, stream, strip_thinking=want_json,
            )
        finally:
            self.arbiter.release(spec.tag)

    async def embeddings(self, request: web.Request) -> web.Response:
        body = await request.json()
        model = body.get("model", "")
        spec = self._resolve_spec(model)
        if not spec:
            return web.json_response({"error": f"model '{model}' not found"}, status=404)
        try:
            await self.arbiter.acquire(spec)
        except BackendLoading as exc:
            return _still_loading(exc)
        try:
            inp = body.get("prompt") or body.get("input") or ""
            upstream_payload = {"model": spec.served_name, "input": inp}
            assert self._client_session is not None
            async with self._client_session.post(f"{self.arbiter.upstream_for(spec.tag)}/v1/embeddings", json=upstream_payload) as resp:
                data = await resp.json()
            # ollama shape: {"embedding": [...]}  (single)  or {"embeddings": [[...]]}
            if isinstance(data.get("data"), list) and data["data"]:
                first = data["data"][0].get("embedding", [])
                return web.json_response({"embedding": first, "embeddings": [d.get("embedding", []) for d in data["data"]]})
            return web.json_response({"embedding": [], "embeddings": []})
        finally:
            self.arbiter.release(spec.tag)

    async def _handle_unload(self, model: str, spec=None) -> web.Response:
        """Unload one model — ollama's `keep_alive: 0`. Returns ollama's shape.

        Named, not "everything". With one resident the two were the same
        thing; with two, tearing the card down because a client finished with
        one of them takes the other model out from under whoever was using it.
        A tag that is not resident is a no-op that still answers `done`, which
        is what a client asking for memory it is not holding should hear.
        """
        if spec is None:
            await self.arbiter.stop()
        else:
            await self.arbiter.evict(spec.tag)
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
        try:
            await self.arbiter.acquire(spec)
        except BackendLoading as exc:
            return _still_loading(exc)
        try:
            body["model"] = spec.served_name

            upstream_url = f"{self.arbiter.upstream_for(spec.tag)}{request.path}"
            assert self._client_session is not None
            async with self._client_session.post(upstream_url, json=body) as upstream:
                resp = web.StreamResponse(status=upstream.status, headers={"Content-Type": upstream.headers.get("Content-Type", "application/json")})
                await resp.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        finally:
            self.arbiter.release(spec.tag)

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


def _still_loading(exc: BackendLoading) -> web.Response:
    """503 with a reason, instead of a socket that never answers.

    `Retry-After` is deliberately short: whatever is loading is nearly always
    about to finish, and the caller should come back rather than fail over.
    """
    log.warning("refusing a request: %s", exc)
    return web.json_response(
        {"error": str(exc)}, status=503, headers={"Retry-After": "10"}
    )


_LEASE_STATUS = {
    InvalidRequest: 400,
    UnknownLease: 404,
    NoRoom: 408,
    NotGrantable: 413,
    CardUnreadable: 503,
}


def _lease_error(exc: LeaseError) -> web.Response:
    status = _LEASE_STATUS.get(type(exc), 400)
    if status >= 500 or status == 408:
        log.warning("lease request refused (%d): %s", status, exc)
    return web.json_response({"error": str(exc)}, status=status)


def make_app(
    registry: ModelRegistry,
    arbiter: ResidencyArbiter,
    broker: Optional[Broker] = None,
) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    r = Router(registry, arbiter, broker)
    app.on_startup.append(r.startup)
    app.on_cleanup.append(r.cleanup)
    app.router.add_get("/", r.health)
    app.router.add_get("/api/version", r.version)
    app.router.add_get("/api/tags", r.list_tags)
    app.router.add_get("/api/ps", r.ps)
    app.router.add_get("/gpu/state", r.gpu_state)
    app.router.add_get("/gpu/lease", r.lease_list)
    app.router.add_post("/gpu/lease", r.lease_acquire)
    app.router.add_delete("/gpu/lease/{lease_id}", r.lease_release)
    app.router.add_post("/gpu/lease/{lease_id}/renew", r.lease_renew)
    app.router.add_post("/gpu/evict", r.evict)
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

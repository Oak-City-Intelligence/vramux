"""A resident that is not a language model: loaded on request, reached through
`/m/{tag}/...`, and never idled out when its own traffic bypasses the router.

Motivating case: an image workflow server in a container. It has no chat API,
answers readiness on `/system_stats` rather than `/health`, and its UI talks to
its own port — so vramux cannot see it being used and must not guess.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from vramux.registry import KIND_DOCKER, ModelRegistry, ModelSpec
from vramux.router import Router, make_app
from vramux.translate import _is_load_request

from test_residency import FakeArbiter, packing_arbiter, spec


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


# ---- registry ---------------------------------------------------------------


def test_docker_entry_takes_a_health_path_and_a_zero_idle(isolated_registry, tmp_path):
    cfg = tmp_path / "models.yml"
    cfg.write_text(yaml.safe_dump({"models": {"comfy:0.37": {
        "kind": "docker", "compose_file": "/x/docker-compose.yml",
        "compose_service": "comfy", "port": 8188,
        "health_path": "/system_stats", "idle_timeout": 0, "exclusive": True,
    }}}))
    reg = ModelRegistry(model_dir=tmp_path / "none", config_file=cfg)
    s = reg.get("comfy:0.37")
    assert s is not None and s.kind == KIND_DOCKER
    assert s.health_path == "/system_stats"
    assert s.idle_timeout == 0 and s.exclusive


def test_health_path_defaults_to_health():
    s = ModelSpec(tag="d:1", kind=KIND_DOCKER, compose_file=Path("/x"), compose_service="d", port=1)
    assert s.health_path == "/health"


# ---- load-only ------------------------------------------------------------


def test_empty_prompt_is_a_load_and_keep_alive_zero_is_not():
    assert _is_load_request({"model": "x"})
    assert _is_load_request({"model": "x", "prompt": ""})
    assert not _is_load_request({"model": "x", "prompt": "hi"})
    assert not _is_load_request({"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    assert not _is_load_request({"model": "x", "keep_alive": 0})


async def test_empty_generate_loads_the_model_and_answers_done():
    arbiter = FakeArbiter()
    reg = ModelRegistry()
    s = spec("img:1")
    reg._specs = {"img:1": s}  # noqa: SLF001 — no disk, no scan
    router = Router(reg, arbiter)
    resp = await router.generate(_Req({"model": "img:1"}))
    body = json.loads(resp.text)
    assert body["done"] and body["done_reason"] == "load"
    assert arbiter.current_tag == "img:1"
    assert arbiter.residents[0].inflight == 0, "a warm-up holds nothing in flight"


# ---- never idle -------------------------------------------------------------


async def _one_idle_pass(arbiter, monkeypatch) -> None:
    """Run the watcher body once, in the foreground, against a stale resident.

    The watcher the arbiter spawned for itself is cancelled first, so only one
    task is consuming the patched sleep.
    """
    arbiter._idle_task.cancel()
    monkeypatch.setattr("vramux.residency.asyncio.sleep", _no_sleep_once())
    for r in arbiter.residents:
        r.last_use = time.monotonic() - 10_000
    try:
        await arbiter._idle_watch()
    except _Stop:
        pass  # second pass reached with a resident still up


async def test_zero_idle_timeout_means_never(monkeypatch):
    arbiter = FakeArbiter()
    await arbiter.acquire(spec("ui:1", kind=KIND_DOCKER, idle_timeout=0))
    arbiter.release("ui:1")
    await _one_idle_pass(arbiter, monkeypatch)
    assert arbiter.current_tag == "ui:1"


async def test_positive_idle_timeout_still_expires(monkeypatch):
    arbiter = FakeArbiter()
    await arbiter.acquire(spec("llm:1", idle_timeout=5))
    arbiter.release("llm:1")
    await _one_idle_pass(arbiter, monkeypatch)
    assert arbiter.current_tag is None


def _no_sleep_once():
    """The watcher loops forever; after one pass, make it exit."""
    calls = {"n": 0}

    async def fake_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] > 1:
            raise _Stop()

    return fake_sleep


class _Stop(Exception):
    pass


class _RealUpstreamArbiter(FakeArbiter):
    """Fake residency, real socket: the proxy has to hit something."""

    def __init__(self, upstream: str):
        super().__init__()
        self._real_upstream = upstream

    def upstream_for(self, tag: str) -> str:
        return self._real_upstream


# ---- /m/{tag}/... -----------------------------------------------------------


async def test_model_proxy_admits_then_forwards_verbatim():
    seen = {}

    async def upstream_handler(request: web.Request):
        seen["method"] = request.method
        seen["path"] = request.path_qs
        seen["body"] = await request.text()
        seen["ctype"] = request.headers.get("Content-Type")
        return web.json_response({"prompt_id": "abc"}, status=201)

    up = web.Application()
    up.router.add_route("*", "/{p:.*}", upstream_handler)
    async with TestServer(up) as upstream:
        arbiter = _RealUpstreamArbiter(str(upstream.make_url("")).rstrip("/"))
        reg = ModelRegistry()
        reg._specs = {"img:1": spec("img:1")}  # noqa: SLF001
        app = make_app(reg, arbiter)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/m/img:1/prompt?client_id=x",
                data=json.dumps({"prompt": {}}),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 201
            assert await resp.json() == {"prompt_id": "abc"}
            assert arbiter.current_tag == "img:1"
            assert arbiter.residents[0].inflight == 0
    assert seen == {
        "method": "POST", "path": "/prompt?client_id=x",
        "body": '{"prompt": {}}', "ctype": "application/json",
    }


async def test_model_proxy_unknown_tag_is_404():
    arbiter = FakeArbiter()
    app = make_app(ModelRegistry(), arbiter)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/m/ghost:1/anything")
        assert resp.status == 404

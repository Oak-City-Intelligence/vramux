"""What the HTTP surface says once more than one model can be resident.

Two answers stopped being the same thing when admission opened: "what is
loaded" is now a list, and "unload" is now about one model rather than the
card. Both are tested against the fake arbiter from `test_residency`, so no
backend, no GPU and no subprocess is involved.
"""

from __future__ import annotations

import json

from vramux.registry import ModelRegistry
from vramux.router import Router

from test_residency import FakeArbiter, packing_arbiter, spec


def _body(response) -> dict:
    return json.loads(response.text)


async def test_ps_lists_every_resident_not_just_the_hottest():
    """Reporting only the hottest would tell a client the card is free of a
    model that is very much on it."""
    arbiter = packing_arbiter({"a:9b": 100, "b:9b": 100}, free_mb=20000)
    await arbiter.acquire(spec("a:9b"))
    arbiter.release("a:9b")
    await arbiter.acquire(spec("b:9b"))
    arbiter.release("b:9b")

    router = Router(ModelRegistry(), arbiter)
    listed = _body(await router.ps(None))["models"]
    assert sorted(m["name"] for m in listed) == ["a:9b", "b:9b"]


async def test_unload_takes_out_the_named_model_and_leaves_the_other():
    """`keep_alive: 0` means "I am done with this model", not "tear the card
    down" — which is what it used to mean, back when there was only one."""
    arbiter = packing_arbiter({"a:9b": 100, "b:9b": 100}, free_mb=20000)
    await arbiter.acquire(spec("a:9b"))
    arbiter.release("a:9b")
    b = spec("b:9b")
    await arbiter.acquire(b)
    arbiter.release("b:9b")

    router = Router(ModelRegistry(), arbiter)
    payload = _body(await router._handle_unload("b:9b", b))
    assert payload["done_reason"] == "unload"
    assert [r.tag for r in arbiter.residents] == ["a:9b"]


async def test_unloading_something_that_is_not_resident_still_answers_done():
    arbiter = FakeArbiter()
    router = Router(ModelRegistry(), arbiter)
    payload = _body(await router._handle_unload("ghost:9b", spec("ghost:9b")))
    assert payload["done"] is True

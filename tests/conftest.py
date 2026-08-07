"""Shared helpers. Everything here is GPU-less and subprocess-less by design:
the whole suite must be runnable on a machine with no card and no llama.cpp.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Iterable, List

import pytest


class FakeStream:
    """Minimal stand-in for `aiohttp.StreamReader` as `_iter_sse` uses it:
    an async iterator over raw byte lines."""

    def __init__(self, lines: Iterable[bytes]) -> None:
        self._lines = list(lines)

    def __aiter__(self) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            for line in self._lines:
                yield line

        return gen()


def sse(*events: Any) -> FakeStream:
    """Build an SSE byte stream. A `str` event is sent verbatim after `data: `
    (used for `[DONE]` and for deliberately malformed payloads); anything else
    is JSON-encoded. Blank separator lines are interleaved, as a real upstream
    sends them, to prove they are skipped."""
    lines: List[bytes] = []
    for ev in events:
        payload = ev if isinstance(ev, str) else json.dumps(ev)
        lines.append(f"data: {payload}\n".encode())
        lines.append(b"\n")
    return FakeStream(lines)


def chunk(
    *,
    content: str = "",
    reasoning: str = "",
    tool_calls: Any = None,
    finish: str = None,
    role: str = None,
) -> Dict[str, Any]:
    """One OpenAI streaming chat chunk."""
    delta: Dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


def tc(index: int, *, name: str = None, args: str = None) -> Dict[str, Any]:
    """One tool-call fragment as OpenAI streams them: keyed by `index`, with
    the name on the first fragment and the argument JSON dribbling in after."""
    fn: Dict[str, Any] = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    return {"index": index, "function": fn}


async def collect(agen: AsyncIterator[bytes]) -> List[Dict[str, Any]]:
    """Drain a translation generator into parsed ollama JSON-lines."""
    out: List[Dict[str, Any]] = []
    async for raw in agen:
        assert raw.endswith(b"\n"), "ollama wire format is newline-delimited JSON"
        out.append(json.loads(raw))
    return out


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path):
    """Point the ollama-blob discovery at an empty tree so tests never see
    this machine's real model store."""
    from llama_router import registry as reg

    empty = tmp_path / "no-ollama"
    monkeypatch.setattr(reg, "MANIFESTS_ROOT", empty)
    monkeypatch.setattr(reg, "BLOBS_ROOT", empty)
    return reg

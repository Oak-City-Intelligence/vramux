"""OpenAI ↔ ollama wire-format translation.

llama-server and every container backend speak OpenAI. The clients on this
machine speak ollama. Neither is negotiable, so the difference lives here —
free functions over plain dicts and byte streams, with no knowledge of the
router, the registry or what is resident.

That is what makes this the best-tested part of the project: every case below
regressed at least once in real use, and each one is reachable from a test
without a GPU, a socket or a model.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List

import aiohttp


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
            frag_buf = self._calls.setdefault(idx, {"name": "", "arguments": ""})
            fn = call.get("function") or {}
            if fn.get("name"):
                frag_buf["name"] = fn["name"]
            frag = fn.get("arguments")
            if isinstance(frag, str):
                frag_buf["arguments"] += frag

    def drain(self) -> List[Dict[str, Any]]:
        out = _openai_tool_calls_to_ollama([
            {"function": {"name": frag_buf["name"], "arguments": frag_buf["arguments"]}}
            for _, frag_buf in sorted(self._calls.items())
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

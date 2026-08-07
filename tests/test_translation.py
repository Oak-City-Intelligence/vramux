"""OpenAI -> ollama wire-format translation.

These are the cases that regressed and were verified with throwaway inline
asserts last session. Reverting either fix in `router.py` must turn one of
these red.
"""

from __future__ import annotations

import pytest

from llama_router.router import (
    _is_unload_request,
    _ollama_options_to_openai,
    _openai_chat_to_ollama,
    _openai_tool_calls_to_ollama,
    _sse_chat_to_ollama_lines,
    _sse_generate_to_ollama_lines,
    _strip_thinking,
)

from conftest import chunk, collect, sse, tc


def terminals(lines):
    return [l for l in lines if l.get("done")]


# ---- streaming chat -------------------------------------------------------


async def test_plain_content_streams_and_ends_once():
    lines = await collect(_sse_chat_to_ollama_lines(
        sse(
            chunk(role="assistant", content="Hel"),
            chunk(content="lo"),
            chunk(finish="stop"),
            "[DONE]",
        ),
        "m:1",
    ))
    assert "".join(l["message"]["content"] for l in lines) == "Hello"
    assert len(terminals(lines)) == 1
    assert terminals(lines)[0]["done_reason"] == "stop"
    assert all(l["model"] == "m:1" for l in lines)


async def test_fragmented_tool_call_reassembled_and_emitted_once():
    """Args arrive as `{`, `"city": "Bra`, `ga"}` — only the concatenation
    parses, so fragments must be buffered and emitted a single time."""
    lines = await collect(_sse_chat_to_ollama_lines(
        sse(
            chunk(tool_calls=[tc(0, name="get_weather", args="")]),
            chunk(tool_calls=[tc(0, args='{"city": ')]),
            chunk(tool_calls=[tc(0, args='"Bra')]),
            chunk(tool_calls=[tc(0, args='ga"}')]),
            chunk(finish="tool_calls"),
            "[DONE]",
        ),
        "m:1",
    ))
    with_calls = [l for l in lines if l["message"].get("tool_calls")]
    assert len(with_calls) == 1, "fragments must not be emitted per-delta"
    call = with_calls[0]["message"]["tool_calls"][0]["function"]
    assert call == {"name": "get_weather", "arguments": {"city": "Braga"}}
    assert with_calls[0]["done"] is True


async def test_done_does_not_blank_the_finish_chunk():
    """The regression: `[DONE]` emitted a second terminal line whose empty
    message overwrote the first one's tool_calls for any client reading the
    last message."""
    lines = await collect(_sse_chat_to_ollama_lines(
        sse(
            chunk(tool_calls=[tc(0, name="f", args="{}")]),
            chunk(finish="tool_calls"),
            "[DONE]",
        ),
        "m:1",
    ))
    assert len(terminals(lines)) == 1
    last = lines[-1]
    assert last["done_reason"] == "tool_calls"
    assert last["message"]["tool_calls"][0]["function"]["name"] == "f"


async def test_done_without_finish_chunk_still_emits_buffered_calls():
    """Some upstreams close with `[DONE]` and never send a finish_reason. The
    buffered call must not be dropped on the floor."""
    lines = await collect(_sse_chat_to_ollama_lines(
        sse(
            chunk(tool_calls=[tc(0, name="get_weather", args='{"city":"Braga"}')]),
            "[DONE]",
        ),
        "m:1",
    ))
    assert len(terminals(lines)) == 1
    last = lines[-1]
    assert last["done_reason"] == "tool_calls"
    assert last["message"]["tool_calls"][0]["function"]["arguments"] == {"city": "Braga"}


async def test_done_without_any_tool_calls_reports_stop():
    lines = await collect(_sse_chat_to_ollama_lines(sse("[DONE]"), "m:1"))
    assert len(lines) == 1
    assert lines[0]["done"] is True
    assert lines[0]["done_reason"] == "stop"
    assert "tool_calls" not in lines[0]["message"]


async def test_parallel_tool_calls_keyed_by_index_and_ordered():
    lines = await collect(_sse_chat_to_ollama_lines(
        sse(
            chunk(tool_calls=[tc(0, name="a", args="{"), tc(1, name="b", args="{")]),
            chunk(tool_calls=[tc(1, args='"x":2}')]),
            chunk(tool_calls=[tc(0, args='"x":1}')]),
            chunk(finish="tool_calls"),
            "[DONE]",
        ),
        "m:1",
    ))
    calls = lines[-1]["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert [c["function"]["arguments"] for c in calls] == [{"x": 1}, {"x": 2}]


async def test_malformed_argument_json_is_surfaced_not_dropped():
    lines = await collect(_sse_chat_to_ollama_lines(
        sse(
            chunk(tool_calls=[tc(0, name="f", args="{not json")]),
            chunk(finish="tool_calls"),
            "[DONE]",
        ),
        "m:1",
    ))
    args = lines[-1]["message"]["tool_calls"][0]["function"]["arguments"]
    assert args == {"_raw": "{not json"}


async def test_empty_argument_string_becomes_empty_object():
    lines = await collect(_sse_chat_to_ollama_lines(
        sse(chunk(tool_calls=[tc(0, name="f", args="")]), chunk(finish="tool_calls"), "[DONE]"),
        "m:1",
    ))
    assert lines[-1]["message"]["tool_calls"][0]["function"]["arguments"] == {}


async def test_thinking_is_split_from_content():
    lines = await collect(_sse_chat_to_ollama_lines(
        sse(
            chunk(reasoning="hmm"),
            chunk(content="answer"),
            chunk(finish="stop"),
            "[DONE]",
        ),
        "m:1",
    ))
    assert lines[0]["message"]["thinking"] == "hmm"
    assert lines[0]["message"]["content"] == ""
    assert lines[1]["message"]["content"] == "answer"
    assert "thinking" not in lines[1]["message"]


async def test_unparseable_sse_payload_is_skipped():
    lines = await collect(_sse_chat_to_ollama_lines(
        sse("{not json at all", chunk(content="ok"), chunk(finish="stop"), "[DONE]"),
        "m:1",
    ))
    assert [l["message"]["content"] for l in lines] == ["ok", ""]


async def test_non_data_sse_lines_are_ignored():
    from conftest import FakeStream

    stream = FakeStream([
        b": keep-alive comment\n",
        b"event: message\n",
        b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n',
        b"data: [DONE]\n",
    ])
    lines = await collect(_sse_chat_to_ollama_lines(stream, "m:1"))
    assert [l["message"]["content"] for l in lines] == ["x"]


# ---- streaming generate ---------------------------------------------------


async def test_generate_stream_translates_text_and_terminates():
    lines = await collect(_sse_generate_to_ollama_lines(
        sse(
            {"choices": [{"text": "foo", "finish_reason": None}]},
            {"choices": [{"text": "bar", "finish_reason": "length"}]},
            "[DONE]",
        ),
        "m:1",
    ))
    assert "".join(l["response"] for l in lines) == "foobar"
    assert lines[1]["done"] is True and lines[1]["done_reason"] == "length"
    # `[DONE]` after a finish chunk still yields a terminal line here; unlike
    # chat it carries no message to blank, so it is harmless.
    assert lines[-1]["done"] is True


# ---- non-streaming chat ---------------------------------------------------


def test_nonstream_chat_maps_usage_and_finish_reason():
    out = _openai_chat_to_ollama(
        {
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        },
        "m:1",
    )
    assert out["message"] == {"role": "assistant", "content": "hi"}
    assert out["done"] is True and out["done_reason"] == "stop"
    assert out["prompt_eval_count"] == 7 and out["eval_count"] == 3


def test_nonstream_reasoning_falls_back_into_content_when_empty():
    out = _openai_chat_to_ollama(
        {"choices": [{"message": {"content": "", "reasoning_content": "thought"}}]},
        "m:1",
    )
    assert out["message"]["thinking"] == "thought"
    assert out["message"]["content"] == "thought"


def test_nonstream_tool_call_turn_keeps_content_empty():
    """A tool-calling turn legitimately has no content; the reasoning fallback
    must not stuff the scratchpad in there and have clients render it."""
    out = _openai_chat_to_ollama(
        {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "let me call the tool",
                    "tool_calls": [{"function": {"name": "f", "arguments": '{"a":1}'}}],
                },
                "finish_reason": "tool_calls",
            }]
        },
        "m:1",
    )
    assert out["message"]["content"] == ""
    assert out["message"]["thinking"] == "let me call the tool"
    assert out["message"]["tool_calls"][0]["function"]["arguments"] == {"a": 1}


def test_missing_finish_reason_defaults_to_stop():
    out = _openai_chat_to_ollama({"choices": [{"message": {"content": "x"}}]}, "m:1")
    assert out["done_reason"] == "stop"


def test_tool_call_arguments_already_object_pass_through():
    assert _openai_tool_calls_to_ollama([{"function": {"name": "f", "arguments": {"a": 1}}}]) == [
        {"function": {"name": "f", "arguments": {"a": 1}}}
    ]


# ---- small helpers --------------------------------------------------------


@pytest.mark.parametrize("keep_alive", [0, "0", "0s"])
def test_unload_request_detected(keep_alive):
    assert _is_unload_request({"keep_alive": keep_alive}) is True


def test_unload_not_triggered_when_there_is_work():
    assert _is_unload_request({"keep_alive": 0, "messages": [{"role": "user"}]}) is False
    assert _is_unload_request({"messages": []}) is False


def test_options_mapping_includes_llamacpp_extensions():
    out = _ollama_options_to_openai({
        "temperature": 0.7, "top_p": 0.9, "num_predict": 128, "stop": ["</s>"],
        "seed": 1, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.1, "unknown": 5,
    })
    assert out == {
        "temperature": 0.7, "top_p": 0.9, "max_tokens": 128, "stop": ["</s>"],
        "seed": 1, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.1,
    }


def test_strip_thinking_removes_block_and_leading_space():
    assert _strip_thinking("<think>x</think>\n\n{\"a\":1}") == '{"a":1}'
    assert _strip_thinking('{"a":1}') == '{"a":1}'

"""The deprecated-variable shim."""

from __future__ import annotations

import pytest

from vramux import env


@pytest.fixture(autouse=True)
def clear_warn_cache():
    env._warned.clear()


def test_current_name_wins(monkeypatch):
    monkeypatch.setenv("VRAMUX_PORT", "1")
    monkeypatch.setenv("MYLLAMA_PORT", "2")
    assert env.get("PORT") == "1"


def test_legacy_prefix_is_honoured_with_a_warning(monkeypatch, caplog):
    monkeypatch.delenv("VRAMUX_PORT", raising=False)
    monkeypatch.setenv("MYLLAMA_PORT", "2")
    with caplog.at_level("WARNING"):
        assert env.get("PORT") == "2"
    assert "MYLLAMA_PORT is deprecated" in caplog.text
    assert "VRAMUX_PORT" in caplog.text


def test_deprecation_warns_once_per_variable(monkeypatch, caplog):
    monkeypatch.setenv("MYLLAMA_PORT", "2")
    with caplog.at_level("WARNING"):
        env.get("PORT")
        env.get("PORT")
    assert caplog.text.count("MYLLAMA_PORT is deprecated") == 1


def test_unprefixed_legacy_name_still_works(monkeypatch, caplog):
    """`LLAMA_SERVER_BIN` predates any prefix at all."""
    monkeypatch.delenv("VRAMUX_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.delenv("MYLLAMA_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.setenv("LLAMA_SERVER_BIN", "/opt/llama-server")
    with caplog.at_level("WARNING"):
        assert env.get("LLAMA_SERVER_BIN") == "/opt/llama-server"
    assert "deprecated" in caplog.text


def test_default_returned_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv("VRAMUX_NOPE", raising=False)
    monkeypatch.delenv("MYLLAMA_NOPE", raising=False)
    assert env.get("NOPE", "fallback") == "fallback"
    assert env.get("NOPE") is None


def test_numeric_helpers(monkeypatch):
    monkeypatch.setenv("VRAMUX_IDLE_TIMEOUT", "12.5")
    assert env.get_float("IDLE_TIMEOUT", 900) == 12.5
    assert env.get_int("IDLE_TIMEOUT", 900) == 12


def test_unparseable_number_falls_back_instead_of_crashing(monkeypatch, caplog):
    """A typo in the unit file must not stop the router from starting."""
    monkeypatch.setenv("VRAMUX_IDLE_TIMEOUT", "fifteen minutes")
    with caplog.at_level("WARNING"):
        assert env.get_float("IDLE_TIMEOUT", 900) == 900
    assert "not a number" in caplog.text

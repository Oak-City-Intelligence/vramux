"""A model can name its own llama-server build without moving anybody else."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from vramux import backends as sup
from vramux.observer import cost_key
from vramux.registry import ModelRegistry

from test_registry import gguf, write_yaml


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_an_entry_carries_its_own_binary(isolated_registry, tmp_path):
    blob = gguf(tmp_path / "m", "kev.gguf")
    other = gguf(tmp_path / "m", "plain.gguf")
    build = executable(tmp_path / "new-build" / "llama-server")
    cfg = write_yaml(tmp_path / "models.yml", {
        "kev:4b": {"gguf": str(blob), "server_bin": str(build)},
        "plain:9b": {"gguf": str(other)},
    })
    reg = ModelRegistry(config_file=cfg, model_dir=tmp_path / "nope")
    assert reg.get("kev:4b").server_bin == build
    assert reg.get("plain:9b").server_bin is None


def test_a_bad_binary_is_refused_at_load_naming_the_model(isolated_registry, tmp_path, caplog):
    blob = gguf(tmp_path / "m", "kev.gguf")
    not_exec = tmp_path / "build" / "llama-server"
    not_exec.parent.mkdir()
    not_exec.write_text("")
    cfg = write_yaml(tmp_path / "models.yml", {
        "kev:4b": {"gguf": str(blob), "server_bin": str(tmp_path / "missing")},
        "lev:4b": {"gguf": str(blob), "server_bin": str(not_exec)},
    })
    with caplog.at_level(logging.ERROR, logger="vramux.registry"):
        reg = ModelRegistry(config_file=cfg, model_dir=tmp_path / "nope")
    assert reg.get("kev:4b") is None and reg.get("lev:4b") is None
    assert "kev:4b" in caplog.text and "lev:4b" in caplog.text


def test_a_binary_is_part_of_the_cost_key_only_when_named(isolated_registry, tmp_path):
    blob = gguf(tmp_path / "m", "kev.gguf")
    build = executable(tmp_path / "new-build" / "llama-server")
    cfg = write_yaml(tmp_path / "models.yml", {
        "a:4b": {"gguf": str(blob)},
        "b:4b": {"gguf": str(blob), "server_bin": str(build)},
    })
    reg = ModelRegistry(config_file=cfg, model_dir=tmp_path / "nope")
    plain, own = reg.get("a:4b"), reg.get("b:4b")
    own.tag = plain.tag
    # Same configuration on another build is a different cost...
    assert cost_key(own) != cost_key(plain)
    # ...and an entry that names no build keeps the key it was measured under.
    own.server_bin = None
    assert cost_key(own) == cost_key(plain)


async def test_start_uses_the_entry_binary_over_the_global_one(monkeypatch, tmp_path, caplog):
    build = executable(tmp_path / "new-build" / "llama-server")
    monkeypatch.setattr(sup, "resolve_llama_server_bin", lambda: tmp_path / "global" / "llama-server")
    started = []

    async def fake_exec(*args, **kwargs):
        started.append(args[0])
        raise RuntimeError("stop here")

    monkeypatch.setattr(sup.asyncio, "create_subprocess_exec", fake_exec)
    from vramux.registry import ModelSpec
    spec = ModelSpec(tag="kev:4b", gguf_path=gguf(tmp_path / "m", "kev.gguf"), server_bin=build)
    backend = sup.ProcessBackend("127.0.0.1", 18080)
    with caplog.at_level(logging.INFO, logger="vramux.backends"), pytest.raises(RuntimeError, match="stop here"):
        await backend.start(spec, startup_timeout=1)
    assert started == [str(build)]
    assert str(build) in caplog.text

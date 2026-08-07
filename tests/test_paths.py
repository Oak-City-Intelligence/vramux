"""Nothing resolves to one particular machine.

The whole point of this file is the Stage 1 exit criterion: a clean checkout
has to work somewhere that is not the box it was written on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vramux import registry as reg
from vramux import supervisor as sup
from vramux.registry import KIND_DOCKER, ModelRegistry, ModelSpec


# ---- llama-server binary --------------------------------------------------


def test_binary_comes_from_path_when_not_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("VRAMUX_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.delenv("MYLLAMA_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.delenv("LLAMA_SERVER_BIN", raising=False)
    fake = tmp_path / "bin" / "llama-server"
    fake.parent.mkdir()
    fake.touch(mode=0o755)
    monkeypatch.setenv("PATH", str(fake.parent))
    assert sup.resolve_llama_server_bin() == fake


def test_explicit_binary_wins_over_path(monkeypatch, tmp_path):
    """A llama.cpp build tree is the common case and is rarely on $PATH."""
    monkeypatch.setenv("VRAMUX_LLAMA_SERVER_BIN", str(tmp_path / "build" / "llama-server"))
    monkeypatch.setenv("PATH", "/usr/bin")
    assert sup.resolve_llama_server_bin() == tmp_path / "build" / "llama-server"


def test_missing_binary_resolves_to_none(monkeypatch, tmp_path):
    monkeypatch.delenv("VRAMUX_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.delenv("MYLLAMA_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.delenv("LLAMA_SERVER_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert sup.resolve_llama_server_bin() is None


async def test_start_without_a_binary_says_what_to_set(monkeypatch, tmp_path):
    monkeypatch.setattr(sup, "resolve_llama_server_bin", lambda: None)
    backend = sup.ProcessBackend("127.0.0.1", 18080)
    with pytest.raises(RuntimeError, match="VRAMUX_LLAMA_SERVER_BIN"):
        await backend.start(ModelSpec(tag="a:1b"), 5.0)


# ---- model dir and config file -------------------------------------------


def test_model_dir_defaults_to_a_relative_models_directory(monkeypatch):
    monkeypatch.delenv("VRAMUX_MODEL_DIR", raising=False)
    monkeypatch.delenv("MYLLAMA_MODEL_DIR", raising=False)
    assert reg._default_model_dir() == Path("models")


def test_model_dir_expands_a_tilde(monkeypatch):
    monkeypatch.setenv("VRAMUX_MODEL_DIR", "~/gguf")
    assert "~" not in str(reg._default_model_dir())


def test_config_file_is_taken_from_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("VRAMUX_MODELS_CONFIG", raising=False)
    monkeypatch.delenv("MYLLAMA_MODELS_CONFIG", raising=False)
    (tmp_path / "models.yml").write_text("models: {}\n")
    monkeypatch.chdir(tmp_path)
    assert reg._default_config_file() == tmp_path / "models.yml"


def test_configured_config_file_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("VRAMUX_MODELS_CONFIG", str(tmp_path / "elsewhere.yml"))
    assert reg._default_config_file() == tmp_path / "elsewhere.yml"


# ---- ollama blob store ----------------------------------------------------


def test_ollama_root_only_picks_a_location_that_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.delenv("VRAMUX_OLLAMA_MODELS", raising=False)
    real = tmp_path / "second"
    real.mkdir()
    monkeypatch.setattr(reg, "_OLLAMA_ROOT_CANDIDATES", (str(tmp_path / "absent"), str(real)))
    assert reg._resolve_ollama_root() == real


def test_ollama_root_honours_ollamas_own_variable(monkeypatch, tmp_path):
    (tmp_path / "custom").mkdir()
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "custom"))
    assert reg._resolve_ollama_root() == tmp_path / "custom"


def test_no_ollama_installed_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setattr(reg, "_OLLAMA_ROOT_CANDIDATES", (str(tmp_path / "absent"),))
    root = reg._resolve_ollama_root()
    assert not root.is_dir()
    monkeypatch.setattr(reg, "MANIFESTS_ROOT", root / "manifests")
    assert reg._default_specs() == {}


# ---- ctx_overrides --------------------------------------------------------


def test_ctx_overrides_apply_to_discovered_models(isolated_registry, tmp_path):
    """The per-tag context sizes used to be a dict of this box's model tags
    living in the module. They are configuration, so they live in config."""
    models = tmp_path / "gguf"
    models.mkdir()
    (models / "Qwen2.5-Coder-7B-Q4_K_M.gguf").write_bytes(b"\0")
    cfg = tmp_path / "models.yml"
    cfg.write_text("ctx_overrides:\n  qwen2.5-coder:7b: 16384\n")
    reg_ = ModelRegistry(config_file=cfg, model_dir=models)
    assert reg_.get("qwen2.5-coder:7b").ctx_size == 16384


def test_discovered_models_without_an_override_get_the_default(isolated_registry, tmp_path):
    models = tmp_path / "gguf"
    models.mkdir()
    (models / "Qwen2.5-Coder-7B-Q4_K_M.gguf").write_bytes(b"\0")
    reg_ = ModelRegistry(config_file=tmp_path / "absent.yml", model_dir=models)
    assert reg_.get("qwen2.5-coder:7b").ctx_size == reg.DEFAULT_CTX_SIZE


def test_ctx_override_for_an_unknown_tag_is_ignored(isolated_registry, tmp_path):
    cfg = tmp_path / "models.yml"
    cfg.write_text("ctx_overrides:\n  nothing:here: 4096\n")
    assert ModelRegistry(config_file=cfg, model_dir=tmp_path / "nope").all() == []


def test_a_models_entry_beats_a_ctx_override(isolated_registry, tmp_path):
    models = tmp_path / "gguf"
    models.mkdir()
    blob = models / "Qwen2.5-Coder-7B-Q4_K_M.gguf"
    blob.write_bytes(b"\0")
    cfg = tmp_path / "models.yml"
    cfg.write_text(
        "ctx_overrides:\n  qwen2.5-coder:7b: 16384\n"
        f"models:\n  qwen2.5-coder:7b:\n    gguf: {blob}\n    ctx: 32768\n"
    )
    assert ModelRegistry(config_file=cfg, model_dir=models).get("qwen2.5-coder:7b").ctx_size == 32768


# ---- docker specs are fully described by config ---------------------------


def test_docker_backend_requires_an_explicit_service_name():
    """The service name used to default to this machine's one container."""
    spec = ModelSpec(tag="d:35b", kind=KIND_DOCKER, compose_file=Path("/tmp/c.yml"), port=1)
    with pytest.raises(RuntimeError, match="compose_service"):
        sup.DockerComposeBackend(spec)

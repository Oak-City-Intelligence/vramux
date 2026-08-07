"""Registry loading: both backend kinds, aliasing, dedup, precedence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from vramux.registry import KIND_DOCKER, KIND_LLAMA_SERVER, ModelRegistry, ModelSpec


def write_yaml(path: Path, models: dict) -> Path:
    path.write_text(yaml.safe_dump({"models": models}))
    return path


def gguf(dirpath: Path, name: str, size: int = 16) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_bytes(b"\0" * size)
    return p


# ---- YAML loading ---------------------------------------------------------


def test_loads_both_kinds(isolated_registry, tmp_path):
    blob = gguf(tmp_path / "m", "Qwen3.5-9B-Instruct-Q4_K_M.gguf")
    cfg = write_yaml(tmp_path / "models.yml", {
        "qwen3.5:9b": {"gguf": str(blob), "ctx": 16384, "family": "qwen3.5"},
        "container-model:35b": {
            "kind": "docker",
            "compose_file": str(tmp_path / "docker-compose.yml"),
            "compose_service": "container-model",
            "port": 30000,
            "served_name": "container-model/Qwen3.6-35B",
            "ctx": 131072,
            "idle_timeout": 3600,
        },
    })
    reg = ModelRegistry(config_file=cfg, model_dir=tmp_path / "nope")

    gguf_spec = reg.get("qwen3.5:9b")
    assert gguf_spec.kind == KIND_LLAMA_SERVER
    assert gguf_spec.gguf_path == blob and gguf_spec.ctx_size == 16384
    assert gguf_spec.served_name == "qwen3.5:9b"  # llama-server gets --alias

    doc = reg.get("container-model:35b")
    assert doc.kind == KIND_DOCKER
    assert doc.port == 30000 and doc.compose_service == "container-model"
    assert doc.idle_timeout == 3600.0
    # A docker backend answers to its own baked-in id, not our tag.
    assert doc.served_name == "container-model/Qwen3.6-35B"


def test_docker_entry_without_served_name_falls_back_to_tag(isolated_registry, tmp_path):
    cfg = write_yaml(tmp_path / "models.yml", {
        "x:1b": {"kind": "docker", "compose_file": str(tmp_path / "c.yml"), "port": 1},
    })
    assert ModelRegistry(config_file=cfg, model_dir=tmp_path / "nope").get("x:1b").served_name == "x:1b"


def test_unreadable_config_does_not_crash_the_registry(isolated_registry, tmp_path):
    cfg = tmp_path / "models.yml"
    cfg.write_text("models: [this is: not: valid")
    reg = ModelRegistry(config_file=cfg, model_dir=tmp_path / "nope")
    assert reg.all() == []


def test_missing_config_file_is_fine(isolated_registry, tmp_path):
    reg = ModelRegistry(config_file=tmp_path / "absent.yml", model_dir=tmp_path / "nope")
    assert reg.all() == []


# ---- directory scan and precedence ---------------------------------------


def test_scan_infers_tags_from_filenames(isolated_registry, tmp_path):
    models = tmp_path / "gguf"
    gguf(models, "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
    reg = ModelRegistry(config_file=tmp_path / "absent.yml", model_dir=models)
    assert reg.get("qwen2.5-coder:7b") is not None


def test_scan_flags_embedding_models(isolated_registry, tmp_path):
    models = tmp_path / "gguf"
    gguf(models, "nomic-embed-text-v1.5.gguf")
    reg = ModelRegistry(config_file=tmp_path / "absent.yml", model_dir=models)
    assert any(s.is_embedding for s in reg.all())


def test_yaml_overrides_scanned_entry_for_same_tag(isolated_registry, tmp_path):
    models = tmp_path / "gguf"
    blob = gguf(models, "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
    cfg = write_yaml(tmp_path / "models.yml", {
        "qwen2.5-coder:7b": {"gguf": str(blob), "ctx": 32768, "n_gpu_layers": 999,
                             "extra_args": ["--flash-attn"]},
    })
    reg = ModelRegistry(config_file=cfg, model_dir=models)
    spec = reg.get("qwen2.5-coder:7b")
    assert spec.ctx_size == 32768
    assert spec.n_gpu_layers == 999
    assert spec.extra_args == ["--flash-attn"]


def test_scanned_alias_for_a_yaml_gguf_is_dropped(isolated_registry, tmp_path):
    """The scan derives an ugly filename tag for a blob the YAML already names
    properly. Listing both would show the same model twice."""
    models = tmp_path / "gguf"
    blob = gguf(models, "Some-Weird-Name-9B-Q4.gguf")
    cfg = write_yaml(tmp_path / "models.yml", {"pretty:9b": {"gguf": str(blob), "ctx": 8192}})
    reg = ModelRegistry(config_file=cfg, model_dir=models)
    tags = [s.tag for s in reg.all()]
    assert tags == ["pretty:9b"]


def test_scan_first_match_wins_for_colliding_tags(isolated_registry, tmp_path):
    models = tmp_path / "gguf"
    gguf(models / "a", "Model-7B-Q4_K_M.gguf")
    gguf(models / "b", "Model-7B-Q8_0.gguf")
    reg = ModelRegistry(config_file=tmp_path / "absent.yml", model_dir=models)
    assert len([s for s in reg.all() if s.tag == "model:7b"]) == 1


# ---- ollama blob discovery ------------------------------------------------


def test_ollama_manifests_are_discovered_and_latest_is_aliased(isolated_registry, tmp_path, monkeypatch):
    reg_mod = isolated_registry
    manifests = tmp_path / "manifests" / "library"
    blobs = tmp_path / "blobs"
    blobs.mkdir(parents=True)
    digest = "sha256:" + "ab" * 32
    (blobs / digest.replace("sha256:", "sha256-")).write_bytes(b"\0" * 8)
    (manifests / "gemma3").mkdir(parents=True)
    (manifests / "gemma3" / "latest").write_text(json.dumps({
        "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": digest}]
    }))
    monkeypatch.setattr(reg_mod, "MANIFESTS_ROOT", manifests)
    monkeypatch.setattr(reg_mod, "BLOBS_ROOT", blobs)

    reg = ModelRegistry(config_file=tmp_path / "absent.yml", model_dir=tmp_path / "nope")
    assert reg.get("gemma3:latest") is not None
    # ollama treats bare "<name>" and "<name>:latest" as the same tag...
    assert reg.get("gemma3") is reg.get("gemma3:latest")
    # ...and the alias must not be listed as a second model.
    assert [s.tag for s in reg.all()] == ["gemma3:latest"]


def test_manifest_without_a_model_layer_is_skipped(isolated_registry, tmp_path, monkeypatch):
    manifests = tmp_path / "manifests" / "library"
    (manifests / "broken").mkdir(parents=True)
    (manifests / "broken" / "latest").write_text(json.dumps({"layers": [{"mediaType": "other"}]}))
    monkeypatch.setattr(isolated_registry, "MANIFESTS_ROOT", manifests)
    reg = ModelRegistry(config_file=tmp_path / "absent.yml", model_dir=tmp_path / "nope")
    assert reg.all() == []


# ---- lookup ---------------------------------------------------------------


def test_bare_name_resolves_to_a_variant(isolated_registry, tmp_path):
    cfg = write_yaml(tmp_path / "models.yml", {"qwen3.5:9b": {"gguf": str(gguf(tmp_path / "m", "a.gguf"))}})
    reg = ModelRegistry(config_file=cfg, model_dir=tmp_path / "nope")
    assert reg.get("qwen3.5").tag == "qwen3.5:9b"


def test_unknown_tag_returns_none(isolated_registry, tmp_path):
    reg = ModelRegistry(config_file=tmp_path / "absent.yml", model_dir=tmp_path / "nope")
    assert reg.get("nope:1b") is None


# ---- dedup identity -------------------------------------------------------


def test_same_blob_at_different_ctx_are_distinct_models(isolated_registry, tmp_path):
    blob = gguf(tmp_path / "m", "a.gguf")
    cfg = write_yaml(tmp_path / "models.yml", {
        "q:9b": {"gguf": str(blob), "ctx": 16384},
        "q:9b-96k": {"gguf": str(blob), "ctx": 98304},
    })
    reg = ModelRegistry(config_file=cfg, model_dir=tmp_path / "nope")
    assert len(reg.all()) == 2


def test_docker_identity_does_not_collide_with_gguf_identity():
    a = ModelSpec(tag="a", gguf_path=None, ctx_size=8192)
    b = ModelSpec(tag="b", kind=KIND_DOCKER, compose_file=None, compose_service=None, ctx_size=8192)
    assert a.identity != b.identity


# ---- tag entry shape ------------------------------------------------------


def test_tag_entry_reports_size_and_format_per_kind(isolated_registry, tmp_path):
    blob = gguf(tmp_path / "m", "a.gguf", size=1234)
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"\0" * 99)
    cfg = write_yaml(tmp_path / "models.yml", {
        "q:9b": {"gguf": str(blob), "family": "qwen"},
        "d:35b": {"kind": "docker", "compose_file": str(tmp_path / "c.yml"), "port": 1,
                  "weights_dir": str(weights), "quantization": "vendor-quant-2bit"},
    })
    reg = ModelRegistry(config_file=cfg, model_dir=tmp_path / "nope")
    entries = {s.tag: s.to_ollama_tag_entry() for s in reg.all()}
    assert entries["q:9b"]["size"] == 1234
    assert entries["q:9b"]["details"]["format"] == "gguf"
    # A docker model has no single blob to stat; size comes from the weights dir.
    assert entries["d:35b"]["size"] == 99
    assert entries["d:35b"]["details"]["format"] == "docker"
    assert entries["d:35b"]["details"]["quantization_level"] == "vendor-quant-2bit"


def test_size_is_zero_when_weights_are_absent():
    assert ModelSpec(tag="x", gguf_path=Path("/nonexistent/x.gguf")).size_bytes == 0
    assert ModelSpec(tag="x").size_bytes == 0

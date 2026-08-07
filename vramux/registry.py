"""Model registry: ollama tag → GGUF blob path + serving params.

Loaded from the models config in the console-dir config dir, with a
fallback hard-coded mapping that covers the models present in
/usr/share/ollama/.ollama/models when this module was written.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from . import env


OLLAMA_MODELS_ROOT = Path("/usr/share/ollama/.ollama/models")
MANIFESTS_ROOT = OLLAMA_MODELS_ROOT / "manifests" / "registry.ollama.ai" / "library"
BLOBS_ROOT = OLLAMA_MODELS_ROOT / "blobs"

DEFAULT_MODEL_DIR = Path(env.get("MODEL_DIR", "<model-dir>"))
DEFAULT_CONFIG_FILE = Path(
    env.get("MODELS_CONFIG", str(Path(__file__).resolve().parent.parent / "models.yml"))
)


KIND_LLAMA_SERVER = "llama-server"
KIND_DOCKER = "docker"


@dataclass
class ModelSpec:
    """A single servable model.

    Two backend kinds share this shape:

    * ``llama-server`` (default) — a local GGUF served by a llama-server
      subprocess the supervisor spawns.
    * ``docker`` — an already-built compose service that exposes an
      OpenAI-compatible server on ``port``. Used for models llama.cpp cannot
      load at all (e.g. the the container model ``vendor-quant`` 2-bit quant, which needs its own
      CUDA kernels). The supervisor brings the container up and down exactly as
      it starts and stops a llama-server, so both kinds compete for the single
      GPU slot under the same swap and idle-unload rules.
    """

    tag: str                     # ollama-style "name:variant", e.g. "qwen3.5:27b"
    gguf_path: Optional[Path] = None   # llama-server kind: path to the .gguf / ollama blob
    ctx_size: int = 8192         # context window in tokens
    # None = let llama-server auto-fit layers (`common_fit_params`). Use an
    # explicit int (e.g. 999) only when you want to force-offload regardless
    # of free VRAM — that path bypasses auto-fit and OOMs on large models.
    n_gpu_layers: Optional[int] = None
    extra_args: List[str] = field(default_factory=list)
    is_embedding: bool = False
    family: Optional[str] = None  # display only

    # --- docker kind ---------------------------------------------------------
    kind: str = KIND_LLAMA_SERVER
    compose_file: Optional[Path] = None
    compose_service: Optional[str] = None
    port: Optional[int] = None          # host port the container publishes
    # Model id the upstream server actually answers to. llama-server is told
    # `--alias <tag>` so tag works there; a docker backend has its own served
    # name baked in at container start, so requests must use that.
    served_name_override: Optional[str] = None
    # Per-model idle timeout. A container that takes minutes to become ready
    # should not be evicted on the same schedule as a GGUF that loads in
    # seconds. None = use the supervisor default.
    idle_timeout: Optional[float] = None
    # Directory of weights, for size reporting only (docker kind has no single
    # blob to stat).
    weights_dir: Optional[Path] = None
    quantization: str = "Q4_K_M"        # display only

    @property
    def served_name(self) -> str:
        return self.served_name_override or self.tag

    @property
    def size_bytes(self) -> int:
        if self.gguf_path is not None:
            try:
                return self.gguf_path.stat().st_size
            except OSError:
                return 0
        if self.weights_dir is not None:
            try:
                return sum(f.stat().st_size for f in self.weights_dir.rglob("*") if f.is_file())
            except OSError:
                return 0
        return 0

    @property
    def identity(self) -> tuple:
        """Key that distinguishes real models from registry aliases."""
        if self.kind == KIND_DOCKER:
            return (self.kind, str(self.compose_file), self.compose_service, self.ctx_size)
        return (str(self.gguf_path), self.ctx_size)

    def to_ollama_tag_entry(self) -> Dict:
        """Shape matching ollama's /api/tags entry."""
        return {
            "name": self.tag,
            "model": self.tag,
            "modified_at": "2025-01-01T00:00:00Z",
            "size": self.size_bytes,
            "digest": "",
            "details": {
                "format": "gguf" if self.kind == KIND_LLAMA_SERVER else self.kind,
                "family": self.family or self.tag.split(":")[0],
                "parameter_size": self.tag.split(":")[-1].upper(),
                "quantization_level": self.quantization,
            },
        }


def _resolve_ollama_blob(name: str, variant: str) -> Optional[Path]:
    """Read an ollama manifest and return the GGUF blob path."""
    manifest = MANIFESTS_ROOT / name / variant
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for layer in data.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer["digest"].replace("sha256:", "sha256-")
            return BLOBS_ROOT / digest
    return None


def _default_specs() -> Dict[str, ModelSpec]:
    """Discover every model under the ollama blob store."""
    specs: Dict[str, ModelSpec] = {}
    if not MANIFESTS_ROOT.is_dir():
        return specs

    # Per-tag context-size overrides (ollama variant naming convention)
    ctx_overrides: Dict[str, int] = {
        "qwen3.5:9b-96k": 98304,
        "qwen3.5:9b-64k": 65536,
        "qwen3.5:9b": 16384,
        "qwen3.5:27b": 16384,
        "qwen3.6:27b": 16384,
        "qwen3.6:27b-q4_k_m": 16384,
        "gpt-oss:20b": 16384,
        "command-r:35b": 16384,
        "phi4:14b": 16384,
        "qwen2.5-coder:7b": 16384,
        "llama3.1:8b": 16384,
        "gemma3:latest": 8192,
        "glm-4.7-flash:latest": 16384,
        "a client-agent:latest": 16384,
        "nomic-embed-text:latest": 8192,
    }

    for name_dir in sorted(MANIFESTS_ROOT.iterdir()):
        if not name_dir.is_dir():
            continue
        name = name_dir.name
        for variant_file in sorted(name_dir.iterdir()):
            if not variant_file.is_file():
                continue
            variant = variant_file.name
            blob = _resolve_ollama_blob(name, variant)
            if blob is None or not blob.is_file():
                continue
            tag = f"{name}:{variant}"
            specs[tag] = ModelSpec(
                tag=tag,
                gguf_path=blob,
                ctx_size=ctx_overrides.get(tag, 8192),
                is_embedding=name.startswith("nomic-embed"),
                family=name,
            )
            # ollama treats "<name>:latest" and bare "<name>" as the same tag.
            if variant == "latest":
                specs[name] = specs[tag]
    return specs


_FILENAME_TAG_RE = re.compile(r"^(.+?)[-_.](\d+(?:\.\d+)?[bB])(?:[-_.](.+))?$")


def _infer_tag_from_filename(stem: str) -> str:
    """Best-effort: 'Qwen2.5-Coder-7B-Instruct-Q4_K_M' -> 'qwen2.5-coder:7b'."""
    low = stem.lower()
    m = _FILENAME_TAG_RE.match(low)
    if m:
        name, size, _rest = m.groups()
        return f"{name.rstrip('-_.')}:{size}"
    return low


def _scan_model_dir(root: Path) -> Dict[str, ModelSpec]:
    """Discover .gguf files in `root` (recursively) and turn them into specs."""
    specs: Dict[str, ModelSpec] = {}
    if not root.is_dir():
        return specs
    for gguf in sorted(root.rglob("*.gguf")):
        tag = _infer_tag_from_filename(gguf.stem)
        if tag in specs:
            continue  # first match wins; YAML override takes precedence later
        specs[tag] = ModelSpec(
            tag=tag,
            gguf_path=gguf,
            ctx_size=8192,
            family=tag.split(":")[0],
            is_embedding="embed" in gguf.stem.lower(),
        )
    return specs


class ModelRegistry:
    """Loads model specs from YAML config, with discovery from a model dir
    and a final fallback to ollama-blob manifests."""

    def __init__(
        self,
        config_file: Optional[Path] = None,
        model_dir: Optional[Path] = None,
    ) -> None:
        self.config_file = config_file or DEFAULT_CONFIG_FILE
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self._specs: Dict[str, ModelSpec] = {}
        self.reload()

    def reload(self) -> None:
        # Precedence (later overrides earlier): ollama blobs < /mnt dir scan < YAML.
        specs = _default_specs()
        specs.update(_scan_model_dir(self.model_dir))
        yaml_gguf_paths: set = set()
        if self.config_file.is_file():
            try:
                data = yaml.safe_load(self.config_file.read_text()) or {}
            except (OSError, yaml.YAMLError):
                data = {}
            for tag, entry in (data.get("models") or {}).items():
                kind = entry.get("kind", KIND_LLAMA_SERVER)
                if kind == KIND_DOCKER:
                    specs[tag] = ModelSpec(
                        tag=tag,
                        kind=KIND_DOCKER,
                        compose_file=Path(entry["compose_file"]).expanduser(),
                        compose_service=entry.get("compose_service", "container-model"),
                        port=int(entry["port"]),
                        served_name_override=entry.get("served_name"),
                        ctx_size=int(entry.get("ctx", 8192)),
                        idle_timeout=(float(entry["idle_timeout"]) if entry.get("idle_timeout") is not None else None),
                        weights_dir=(Path(entry["weights_dir"]).expanduser() if entry.get("weights_dir") else None),
                        quantization=entry.get("quantization", "unknown"),
                        family=entry.get("family"),
                    )
                    continue
                gguf = Path(entry["gguf"]).expanduser()
                yaml_gguf_paths.add(str(gguf))
                specs[tag] = ModelSpec(
                    tag=tag,
                    gguf_path=gguf,
                    ctx_size=int(entry.get("ctx", 8192)),
                    n_gpu_layers=(int(entry["n_gpu_layers"]) if entry.get("n_gpu_layers") is not None else None),
                    extra_args=list(entry.get("extra_args", [])),
                    is_embedding=bool(entry.get("embedding", False)),
                    family=entry.get("family"),
                )
        # Drop auto-discovered specs whose GGUF is already covered by YAML —
        # keeps the listed-tags clean and avoids ugly filename-derived names.
        if yaml_gguf_paths:
            specs = {
                tag: spec
                for tag, spec in specs.items()
                if not (tag not in (data.get("models") or {})
                        and str(spec.gguf_path) in yaml_gguf_paths)
            }
        self._specs = specs

    def get(self, tag: str) -> Optional[ModelSpec]:
        if tag in self._specs:
            return self._specs[tag]
        # Tolerate "name" without :variant — pick "name:latest" or first match.
        if ":" not in tag:
            cand = self._specs.get(f"{tag}:latest")
            if cand:
                return cand
            for spec_tag, spec in self._specs.items():
                if spec_tag.startswith(f"{tag}:"):
                    return spec
        return None

    def all(self) -> List[ModelSpec]:
        # Deduplicate by gguf_path+ctx — registry intentionally aliases tags
        # (e.g., "qwen3.5:latest" → same blob as "qwen3.5:9b") and we don't
        # want to list each alias separately.
        seen = set()
        out: List[ModelSpec] = []
        for spec in self._specs.values():
            key = spec.identity
            if key in seen:
                continue
            seen.add(key)
            out.append(spec)
        return out

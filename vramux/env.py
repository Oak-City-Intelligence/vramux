"""Environment configuration, with a shim for the old variable names.

Every setting is read as ``VRAMUX_<NAME>``. The project shipped for a while as
`llama-router` with a ``MYLLAMA_`` prefix, and this machine's systemd unit and
notes still carry that spelling, so the old names keep working — once each,
with a warning, so a stale variable shows up in the journal instead of quietly
becoming permanent.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("vramux.env")

_PREFIX = "VRAMUX_"
_LEGACY_PREFIX = "MYLLAMA_"

# Settings that predate any prefix at all.
_UNPREFIXED_LEGACY = {"LLAMA_SERVER_BIN": "LLAMA_SERVER_BIN"}

_warned: set = set()


def _warn_once(legacy: str, current: str) -> None:
    if legacy in _warned:
        return
    _warned.add(legacy)
    log.warning("%s is deprecated — rename it to %s", legacy, current)


def get(name: str, default: Optional[str] = None) -> Optional[str]:
    """Value of ``VRAMUX_<name>``, falling back to the deprecated spellings."""
    current = _PREFIX + name
    val = os.environ.get(current)
    if val is not None:
        return val
    for legacy in (_LEGACY_PREFIX + name, _UNPREFIXED_LEGACY.get(name)):
        if legacy is None:
            continue
        val = os.environ.get(legacy)
        if val is not None:
            _warn_once(legacy, current)
            return val
    return default


def get_float(name: str, default: float) -> float:
    raw = get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s%s=%r is not a number — using %s", _PREFIX, name, raw, default)
        return default


def get_int(name: str, default: int) -> int:
    return int(get_float(name, float(default)))

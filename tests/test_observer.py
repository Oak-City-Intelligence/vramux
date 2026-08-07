"""The observer: reading the card, attribution, and cost measurement.

No GPU and no `nvidia-smi`: the probe is a plain function, so the device is
whatever the test says it is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vramux import nvml
from vramux.nvml import DeviceState, GpuProcess
from vramux.observer import (
    FOREIGN_DRIFT_TOLERANCE_MB,
    CostCache,
    Observer,
    cost_key,
)
from vramux.registry import KIND_DOCKER, ModelSpec
from vramux.supervisor import parse_compose_top


def device(used=1000, total=24564, procs=()):
    return DeviceState(
        index=0, name="Test GPU", total_mb=total, used_mb=used,
        free_mb=total - used, processes=list(procs),
    )


def fake_probe(*states):
    """A probe that returns each state in turn, repeating the last."""
    seq = list(states)

    async def probe(_index=0):
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return probe


# ---- nvidia-smi output parsing -------------------------------------------


def test_device_line_with_a_comma_in_the_model_name():
    """`nvidia-smi` does not quote fields, and GPU names contain commas."""
    devices = nvml._parse_devices("0, NVIDIA RTX A6000, Ada, 49140, 1200, 47940")
    assert len(devices) == 1
    assert devices[0].name == "NVIDIA RTX A6000, Ada"
    assert (devices[0].total_mb, devices[0].used_mb, devices[0].free_mb) == (49140, 1200, 47940)


def test_process_line_keeps_a_command_line_full_of_commas():
    procs = nvml._parse_processes("283219, 148, /app/browser --enable-features=A,B,C --flag")
    assert len(procs) == 1
    assert procs[0].pid == 283219 and procs[0].used_mb == 148
    assert procs[0].name.endswith("--flag")
    assert procs[0].short_name == "browser"


def test_unavailable_memory_field_does_not_drop_the_process():
    """A process we cannot size still holds memory; hiding it is worse than
    counting it as zero."""
    procs = nvml._parse_processes("4127, [N/A], /usr/bin/thing")
    assert len(procs) == 1 and procs[0].used_mb == 0


def test_unparseable_lines_are_skipped_not_fatal():
    assert nvml._parse_devices("garbage\n\n0, GPU, 100, 10, 90") != []
    assert nvml._parse_processes("nonsense") == []


def test_free_memory_is_derived_when_the_driver_omits_it():
    devices = nvml._parse_devices("0, GPU, 24564, 8000, [N/A]")
    assert devices[0].free_mb == 24564 - 8000


def test_device_without_required_fields_is_dropped():
    assert nvml._parse_devices("0, GPU, [N/A], [N/A], [N/A]") == []


def test_accounted_and_unattributed_memory():
    """Driver and context overhead belongs to no process. A budget built from
    process sums alone runs optimistic by exactly this gap."""
    d = device(used=7761, procs=[GpuProcess(1, 6588, "llama-server")])
    assert d.accounted_mb == 6588
    assert d.unattributed_mb == 7761 - 6588


async def test_probe_returns_none_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(nvml.shutil, "which", lambda _: None)
    assert nvml.probe() is None
    assert await nvml.aprobe() is None


# ---- docker compose top ---------------------------------------------------


def test_compose_top_reads_the_pid_column_from_the_header():
    """Column layout is not stable across compose versions. This layout leads
    with SERVICE and a replica number, so a fixed index reads the replica."""
    out = (
        "SERVICE  #   UID   PID     PPID    C   STIME  TTY  TIME      CMD\n"
        "runtime  1   root  477150  477119  25  07:14  ?    00:00:09  python -m server\n"
        "runtime  1   root  477646  477150  99  07:14  ?    00:00:30  model::scheduler\n"
    )
    assert parse_compose_top(out) == [477150, 477646]


def test_compose_top_handles_the_bare_ps_header():
    out = (
        "UID    PID    PPID   C   STIME  TTY  TIME      CMD\n"
        "root   1234   1200   0   07:14  ?    00:00:01  server\n"
    )
    assert parse_compose_top(out) == [1234]


def test_compose_top_without_a_header_yields_nothing():
    """Better to attribute nothing than to attribute the wrong number."""
    assert parse_compose_top("root 1234 1200 0 07:14 ? 00:00:01 server") == []


def test_compose_top_tolerates_empty_output():
    assert parse_compose_top("") == []


# ---- attribution ----------------------------------------------------------


async def test_processes_vramux_started_are_recognised_the_rest_foreign():
    procs = [
        GpuProcess(100, 6588, "llama-server"),
        GpuProcess(200, 271, "/usr/bin/compositor"),
    ]
    obs = Observer(probe=fake_probe(device(used=7761, procs=procs)), cache=CostCache(Path("/dev/null")))
    obs.claim("qwen:9b", [100])
    snap = await obs.snapshot()
    assert snap.recognised_mb == 6588
    assert snap.foreign_mb == 271
    assert [a.owner for a in snap.attributions] == ["qwen:9b", None]


async def test_a_resident_with_no_visible_process_is_named_not_hidden():
    """A container backend whose PIDs could not be resolved is counted as
    foreign — pessimistic, but it must be visible that it happened."""
    obs = Observer(probe=fake_probe(device(used=20000, procs=[GpuProcess(9, 19000, "x")])),
                   cache=CostCache(Path("/dev/null")))
    obs.claim("container:35b", [])
    snap = await obs.snapshot()
    assert snap.unlocated_owners == ["container:35b"]
    assert snap.foreign_mb == 19000 and snap.recognised_mb == 0


async def test_release_returns_memory_to_foreign():
    obs = Observer(probe=fake_probe(device(procs=[GpuProcess(100, 500, "s")])),
                   cache=CostCache(Path("/dev/null")))
    obs.claim("a:1b", [100])
    assert (await obs.snapshot()).recognised_mb == 500
    obs.release("a:1b")
    snap = await obs.snapshot()
    assert snap.recognised_mb == 0 and snap.foreign_mb == 500
    assert snap.unlocated_owners == []


async def test_snapshot_is_none_when_the_card_cannot_be_read():
    async def blind(_index=0):
        return None

    assert await Observer(probe=blind, cache=CostCache(Path("/dev/null"))).snapshot() is None


async def test_render_marks_foreign_processes_and_totals():
    obs = Observer(probe=fake_probe(device(used=7761, procs=[
        GpuProcess(100, 6588, "llama-server"), GpuProcess(200, 271, "compositor"),
    ])), cache=CostCache(Path("/dev/null")))
    obs.claim("qwen:9b", [100])
    text = (await obs.snapshot()).render()
    assert "foreign" in text and "qwen:9b" in text
    assert "6588" in text and "24564" in text


# ---- cost keys ------------------------------------------------------------


def spec(**kw):
    base = dict(tag="a:1b", ctx_size=8192)
    base.update(kw)
    return ModelSpec(**base)


def test_context_length_changes_the_cost_key():
    """The same weights at 16K and 128K differ by many gigabytes."""
    assert cost_key(spec(ctx_size=16384)) != cost_key(spec(ctx_size=131072))


def test_identical_configuration_gives_a_stable_key():
    assert cost_key(spec(gguf_path=Path("/m/a.gguf"))) == cost_key(spec(gguf_path=Path("/m/a.gguf")))


@pytest.mark.parametrize("field,value", [
    ("n_gpu_layers", 999),
    ("extra_args", ["-fa", "on"]),
    ("quantization", "Q8_0"),
    ("kind", KIND_DOCKER),
])
def test_footprint_affecting_fields_change_the_key(field, value):
    assert cost_key(spec()) != cost_key(spec(**{field: value}))


# ---- cost cache -----------------------------------------------------------


def test_cache_round_trips_and_counts_samples(tmp_path):
    path = tmp_path / "costs.json"
    cache = CostCache(path)
    s = spec(tag="q:9b", ctx_size=16384)
    cache.record(cost_key(s), s, 6591)
    cache.record(cost_key(s), s, 6600)

    entry = CostCache(path).get(cost_key(s))
    assert entry["measured_mb"] == 6600
    assert entry["previous_mb"] == 6591
    assert entry["samples"] == 2
    assert entry["tag"] == "q:9b" and entry["ctx"] == 16384


def test_corrupt_cache_is_ignored_rather_than_fatal(tmp_path):
    path = tmp_path / "costs.json"
    path.write_text("{not json")
    assert CostCache(path).all() == {}


def test_cache_from_a_future_version_is_ignored(tmp_path):
    path = tmp_path / "costs.json"
    path.write_text(json.dumps({"version": 999, "entries": {"k": {"measured_mb": 1}}}))
    assert CostCache(path).all() == {}


def test_unwritable_cache_does_not_raise(tmp_path):
    cache = CostCache(tmp_path / "nope" / "x" / "costs.json")
    cache.path = Path("/proc/cannot/write/costs.json")
    cache.record("k", spec(), 100)  # must not raise


# ---- measurement ----------------------------------------------------------


async def test_a_clean_load_is_measured(tmp_path):
    cache = CostCache(tmp_path / "costs.json")
    obs = Observer(probe=fake_probe(device(used=1000), device(used=7591)), cache=cache)
    s = spec(tag="q:9b")
    async with obs.measuring(s):
        pass
    assert cache.get(cost_key(s))["measured_mb"] == 6591


async def test_measurement_is_discarded_when_foreign_usage_moves(tmp_path):
    """Something else allocated during the window, so the delta is not ours."""
    cache = CostCache(tmp_path / "costs.json")
    before = device(used=1000, procs=[GpuProcess(200, 100, "other")])
    after = device(used=9000, procs=[GpuProcess(200, 100 + FOREIGN_DRIFT_TOLERANCE_MB + 1, "other")])
    obs = Observer(probe=fake_probe(before, after), cache=cache)
    s = spec()
    async with obs.measuring(s):
        pass
    assert cache.get(cost_key(s)) is None


async def test_small_foreign_drift_is_tolerated(tmp_path):
    """A compositor repainting must not void every measurement."""
    cache = CostCache(tmp_path / "costs.json")
    before = device(used=1000, procs=[GpuProcess(200, 100, "compositor")])
    after = device(used=7000, procs=[GpuProcess(200, 110, "compositor")])
    obs = Observer(probe=fake_probe(before, after), cache=cache)
    s = spec()
    async with obs.measuring(s):
        pass
    assert cache.get(cost_key(s))["measured_mb"] == 6000


async def test_nothing_recorded_when_usage_did_not_grow(tmp_path):
    cache = CostCache(tmp_path / "costs.json")
    obs = Observer(probe=fake_probe(device(used=5000), device(used=5000)), cache=cache)
    s = spec()
    async with obs.measuring(s):
        pass
    assert cache.get(cost_key(s)) is None


async def test_a_failed_load_records_nothing_and_still_raises(tmp_path):
    cache = CostCache(tmp_path / "costs.json")
    obs = Observer(probe=fake_probe(device(used=1000), device(used=9000)), cache=cache)
    s = spec()
    with pytest.raises(RuntimeError):
        async with obs.measuring(s):
            raise RuntimeError("backend died")
    assert cache.get(cost_key(s)) is None


async def test_measurement_survives_a_blind_probe(tmp_path):
    """No GPU visible means no data point, never a failed load."""
    async def blind(_index=0):
        return None

    cache = CostCache(tmp_path / "costs.json")
    obs = Observer(probe=blind, cache=cache)
    async with obs.measuring(spec()):
        pass
    assert cache.all() == {}


async def test_a_broken_probe_cannot_break_a_load(tmp_path):
    async def explode(_index=0):
        raise OSError("nvidia-smi went away")

    obs = Observer(probe=explode, cache=CostCache(tmp_path / "costs.json"))
    ran = False
    async with obs.measuring(spec()):
        ran = True
    assert ran

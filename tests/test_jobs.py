"""The job store: states derived from workflows, and a queue that survives."""

from __future__ import annotations

import json
import time

import pytest

from vramux import jobs as jobs_mod
from vramux.jobs import CANCELLED, DONE, FAILED, PENDING, RUNNING, JobStore


def test_a_job_is_as_far_along_as_its_workflows(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    job = store.submit("img:1", [{"1": {}}, {"2": {}}])
    assert job.state == PENDING and store.position(job) == 0
    a, b = job.workflows
    store.mark_running(job, a)
    assert job.state == RUNNING
    store.mark_done(job, a, [{"filename": "x.png"}])
    store.mark_failed(job, b, "boom")
    assert job.state == FAILED and job.finished is not None


def test_cancel_leaves_a_running_prompt_to_finish(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    job = store.submit("img:1", [{"1": {}}, {"2": {}}])
    store.mark_running(job, job.workflows[0])
    assert store.cancel(job) == 1
    assert job.state == RUNNING, "the running workflow is still out"
    store.mark_done(job, job.workflows[0], [])
    assert job.state == CANCELLED


def test_fifo_across_jobs_and_a_running_workflow_first(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    first = store.submit("img:1", [{"a": {}}])
    second = store.submit("img:1", [{"b": {}}])
    assert store.next_workflow("img:1") == (first, first.workflows[0])
    assert store.position(second) == 1
    store.mark_running(second, second.workflows[0])
    assert store.next_workflow("img:1") == (second, second.workflows[0])


def test_the_queue_survives_a_restart(tmp_path):
    path = tmp_path / "jobs.json"
    store = JobStore(path)
    job = store.submit("img:1", [{"a": {}}, {"b": {}}])
    store.mark_running(job, job.workflows[0])
    job.workflows[0].prompt_id = "p-1"
    store.changed()
    store.record_handover({"dir": "to_render", "total_s": 3.0})

    again = JobStore(path)
    restored = again.get(job.id)
    assert [w.state for w in restored.workflows] == [RUNNING, PENDING]
    assert restored.workflows[0].prompt_id == "p-1"
    assert restored.workflows[1].prompt == {"b": {}}
    assert again.handovers == [{"dir": "to_render", "total_s": 3.0}]


def test_finished_jobs_are_kept_for_a_bounded_time_and_count(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod, "FINISHED_KEEP_MAX", 2)
    store = JobStore(tmp_path / "jobs.json")
    done = []
    for _ in range(3):
        job = store.submit("img:1", [{}])
        store.mark_done(job, job.workflows[0], [])
        done.append(job)
    assert store.get(done[0].id) is None and store.get(done[2].id) is not None

    done[2].finished = time.time() - jobs_mod.FINISHED_KEEP_S - 1
    store.changed()
    assert store.get(done[2].id) is None
    on_disk = json.loads((tmp_path / "jobs.json").read_text())
    assert [r["id"] for r in on_disk["jobs"]] == [done[1].id]


def test_the_file_honours_its_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VRAMUX_JOBS_FILE", str(tmp_path / "q.json"))
    assert jobs_mod.default_jobs_file() == tmp_path / "q.json"
    monkeypatch.delenv("VRAMUX_JOBS_FILE")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert jobs_mod.default_jobs_file() == tmp_path / "vramux" / "jobs.json"


# ---- writing ------------------------------------------------------------------


def test_a_finished_workflow_is_persisted_without_its_graph(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    job = store.submit("img:1", [{"big": {"graph": 1}}, {"next": {}}])
    store.mark_done(job, job.workflows[0], [{"filename": "a.png"}])
    rows = json.loads((tmp_path / "jobs.json").read_text())["jobs"][0]["workflows"]
    assert "prompt" not in rows[0] and rows[0]["outputs"] == [{"filename": "a.png"}]
    assert rows[1]["prompt"] == {"next": {}}, "an open workflow keeps what it needs to run"
    again = JobStore(tmp_path / "jobs.json").get(job.id)
    assert again.workflows[0].state == DONE and again.workflows[0].prompt == {}


async def test_writes_leave_the_event_loop_and_bursts_coalesce(tmp_path, monkeypatch):
    import threading

    writes = []
    real = jobs_mod._write_file

    def counting(path, text):
        writes.append(threading.current_thread() is threading.main_thread())
        real(path, text)

    monkeypatch.setattr(jobs_mod, "_write_file", counting)
    store = JobStore(tmp_path / "jobs.json")
    jobs = [store.submit("img:1", [{"n": i}]) for i in range(20)]
    await store.flush()
    assert 1 <= len(writes) <= 2, f"twenty changes, {len(writes)} writes"
    assert not any(writes), "the write ran on the event loop's thread"
    on_disk = json.loads((tmp_path / "jobs.json").read_text())
    assert len(on_disk["jobs"]) == 20 and jobs[-1].id in {r["id"] for r in on_disk["jobs"]}


# ---- reading a bad file ---------------------------------------------------------


@pytest.mark.parametrize("content", [
    "not json at all",
    json.dumps(["a", "list"]),
    json.dumps({"jobs": {"not": "a list"}}),
    json.dumps({"jobs": [{"id": "x", "model": "img:1", "created_wall": "yesterday",
                          "workflows": []}]}),
    json.dumps({"jobs": [{"id": "x", "model": "img:1", "created_wall": 1.0,
                          "workflows": [{"index": "0", "prompt": {}, "state": "pending"}]}]}),
    json.dumps({"jobs": [], "handovers": "none"}),
])
def test_a_bad_file_is_set_aside_and_startup_goes_on(tmp_path, caplog, content):
    path = tmp_path / "jobs.json"
    path.write_text(content)
    with caplog.at_level("ERROR", logger="vramux.jobs"):
        store = JobStore(path)
    assert store.all() == []
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "jobs.json" in errors[0].getMessage()
    assert not path.exists()
    [aside] = list(tmp_path.glob("jobs.json.corrupt-*"))
    assert aside.read_text() == content


def test_an_unknown_state_fails_the_workflow_on_load(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"jobs": [{
        "id": "x", "model": "img:1", "created_wall": 1.0,
        "workflows": [
            {"index": 0, "prompt": {"a": {}}, "state": "exploded"},
            {"index": 1, "prompt": {"b": {}}, "state": "pending"},
        ],
    }]}))
    job = JobStore(path).get("x")
    assert job.workflows[0].state == FAILED
    assert job.workflows[0].error == "unknown state on load"
    assert job.workflows[1].state == PENDING

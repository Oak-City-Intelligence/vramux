"""The job store: states derived from workflows, and a queue that survives."""

from __future__ import annotations

import json
import time

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

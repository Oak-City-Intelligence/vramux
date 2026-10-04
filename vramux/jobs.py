"""Render jobs: what was asked for, how far it got, and what came out.

A job is a *batch*: one submit carries every workflow a caller wants rendered,
and the scheduler runs them all under one handover of the card each way. Each
workflow is one workflow-server prompt with its own state, because a batch of
eight where the fifth fails still has seven results to hand back.

The store is bookkeeping only — it never talks to the card or the workflow
server. `scheduler.py` moves jobs through it, and every transition is written
to disk before anything else happens, for the reason `lease.py` writes its
leases: a restart that forgets a queued job has silently dropped somebody's
work, and a restart that forgets a *running* one cannot tell whether to look
for the result or submit it again.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("vramux.jobs")

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL = (DONE, FAILED, CANCELLED)

# How long a finished job is kept for `GET /gpu/jobs/{id}`, and the most kept
# at once. A caller polls for its result within minutes; a day is generous,
# and the count bounds the file for a queue that runs hundreds a day.
FINISHED_KEEP_S = 24 * 3600.0
FINISHED_KEEP_MAX = 200

# Handovers remembered for the medians `vramux top` prints.
HANDOVER_KEEP = 20

# Submissions of one workflow before it is failed rather than resubmitted. Two:
# the original, and one more after the workflow server restarted and lost it.
MAX_ATTEMPTS = 2


def default_jobs_file() -> Path:
    override = os.environ.get("VRAMUX_JOBS_FILE")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(base) / "vramux" / "jobs.json"


_STATES = (PENDING, RUNNING, DONE, FAILED, CANCELLED)


def _write_file(path: Path, text: str) -> None:
    """tmp + fsync + replace: a job file half-written after a power cut is a
    queue that is gone. Run off the event loop — an fsync can take a while."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class _Unreadable(ValueError):
    """The job file is not a shape this version wrote."""


def _number(value, name: str, optional: bool = False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _Unreadable(f"{name} is not a number: {value!r}")
    return float(value)


def _typed(value, typ, name: str, optional: bool = False):
    if value is None and optional:
        return None
    if isinstance(value, bool) and typ is int or not isinstance(value, typ):
        raise _Unreadable(f"{name} is not a {typ.__name__}: {value!r}")
    return value


def _workflow_from(row) -> "Workflow":
    _typed(row, dict, "a workflow row")
    state = _typed(row.get("state"), str, "workflow state")
    wf = Workflow(
        index=_typed(row.get("index"), int, "workflow index"),
        # Absent for a finished workflow: its graph is not kept.
        prompt=_typed(row.get("prompt", {}), dict, "workflow prompt"),
        state=state,
        prompt_id=_typed(row.get("prompt_id"), str, "prompt_id", optional=True),
        outputs=_typed(row.get("outputs", []), list, "outputs"),
        error=_typed(row.get("error"), str, "error", optional=True),
        attempts=_typed(row.get("attempts", 0), int, "attempts"),
    )
    if state not in _STATES:
        # A state this version does not know is not a reason to lose the
        # queue, and not one to guess about either.
        wf.state, wf.error = FAILED, "unknown state on load"
    return wf


def _job_from(row) -> "Job":
    _typed(row, dict, "a job row")
    return Job(
        id=_typed(row.get("id"), str, "job id"),
        model=_typed(row.get("model"), str, "job model"),
        created=_number(row.get("created_wall"), "created_wall"),
        started=_number(row.get("started_wall"), "started_wall", optional=True),
        finished=_number(row.get("finished_wall"), "finished_wall", optional=True),
        cancelled=_typed(row.get("cancelled", False), bool, "cancelled"),
        workflows=[_workflow_from(w) for w in _typed(row.get("workflows"), list, "workflows")],
    )


def _iso(wall: Optional[float]) -> Optional[str]:
    if wall is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wall))


@dataclass
class Workflow:
    """One prompt of a batch."""

    index: int
    prompt: Dict[str, Any]
    state: str = PENDING
    prompt_id: Optional[str] = None
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    attempts: int = 0

    def to_json(self, with_prompt: bool = True) -> Dict[str, Any]:
        row = {
            "index": self.index, "state": self.state, "prompt_id": self.prompt_id,
            "outputs": self.outputs, "error": self.error, "attempts": self.attempts,
        }
        if with_prompt:
            row["prompt"] = self.prompt
        return row


@dataclass
class Job:
    id: str
    model: str
    workflows: List[Workflow]
    created: float
    started: Optional[float] = None
    finished: Optional[float] = None
    cancelled: bool = False

    @property
    def state(self) -> str:
        """Derived, never stored: a job is exactly as far along as its parts."""
        states = [w.state for w in self.workflows]
        if all(s in TERMINAL for s in states):
            if self.cancelled:
                return CANCELLED
            return FAILED if FAILED in states else DONE
        if any(s != PENDING for s in states):
            return RUNNING
        return PENDING

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    def open_workflows(self) -> List[Workflow]:
        """What still needs the card: pending, or running with a result unseen."""
        return [w for w in self.workflows if w.state in (PENDING, RUNNING)]

    def to_json(self, with_prompt: bool = False) -> Dict[str, Any]:
        return {
            "id": self.id, "model": self.model, "state": self.state,
            "created": _iso(self.created), "started": _iso(self.started),
            "finished": _iso(self.finished),
            "workflows": [w.to_json(with_prompt) for w in self.workflows],
        }


class JobStore:
    """Every job vramux knows about, persisted on each change."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path
        self._jobs: Dict[str, Job] = {}
        self.handovers: List[Dict[str, Any]] = []
        # Woken on every transition. A `?wait=1` caller sleeps on it rather
        # than polling the store.
        self._changed = asyncio.Event()
        # Writes go to a thread, one at a time, and a burst of transitions
        # becomes one write of the latest state.
        self._dirty = False
        self._writer: Optional[asyncio.Task] = None
        self._write_lock = asyncio.Lock()
        if path is not None:
            self._load()

    # ---- reading --------------------------------------------------------------

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def all(self) -> List[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created)

    def open_jobs(self, model: Optional[str] = None) -> List[Job]:
        """Jobs with work left, oldest first."""
        return [
            j for j in self.all()
            if not j.terminal and (model is None or j.model == model)
        ]

    def next_workflow(self, model: str, only: Optional[set] = None) -> Optional[tuple]:
        """The next (job, workflow) to run on `model`: FIFO across jobs.

        A running workflow comes before any pending one — it was submitted and
        its result has not been seen, which after a restart is the first thing
        to settle. `only`, a set of (job id, index), limits the choice to it.
        """
        jobs = self.open_jobs(model)
        for state in (RUNNING, PENDING):
            for job in jobs:
                for wf in job.workflows:
                    if wf.state == state and (only is None or (job.id, wf.index) in only):
                        return job, wf
        return None

    def open_keys(self, model: str) -> set:
        """(job id, index) of every workflow on `model` that still needs the card."""
        return {(j.id, w.index) for j in self.open_jobs(model) for w in j.open_workflows()}

    def position(self, job: Job) -> int:
        """Open jobs ahead of this one on the same model; 0 is next."""
        ahead = 0
        for other in self.open_jobs(job.model):
            if other.id == job.id:
                return ahead
            ahead += 1
        return 0

    # ---- changing -------------------------------------------------------------

    def submit(self, model: str, prompts: List[Dict[str, Any]]) -> Job:
        job = Job(
            id=uuid.uuid4().hex, model=model, created=time.time(),
            workflows=[Workflow(index=i, prompt=p) for i, p in enumerate(prompts)],
        )
        self._jobs[job.id] = job
        log.info("job %s: %d workflow(s) queued for %s", job.id, len(prompts), model)
        self.changed()
        return job

    def cancel(self, job: Job) -> int:
        """Cancel what has not started. A running prompt finishes; its result
        is kept. Returns the number of workflows cancelled."""
        n = 0
        for wf in job.workflows:
            if wf.state == PENDING:
                wf.state = CANCELLED
                n += 1
        if n or not job.terminal:
            job.cancelled = True
        self._finish_if_done(job)
        self.changed()
        return n

    def mark_running(self, job: Job, wf: Workflow) -> None:
        wf.state = RUNNING
        if job.started is None:
            job.started = time.time()
        self.changed()

    def mark_done(self, job: Job, wf: Workflow, outputs: List[Dict[str, Any]]) -> None:
        wf.state = DONE
        wf.outputs = outputs
        wf.error = None
        self._finish_if_done(job)
        self.changed()

    def mark_failed(self, job: Job, wf: Workflow, error: str) -> None:
        wf.state = FAILED
        wf.error = error
        self._finish_if_done(job)
        self.changed()

    def record_handover(self, row: Dict[str, Any]) -> None:
        self.handovers.append(row)
        del self.handovers[:-HANDOVER_KEEP]
        self.changed()

    def changed(self) -> None:
        """Persist, and wake every waiter. Called after each transition."""
        self._prune()
        self._save()
        event, self._changed = self._changed, asyncio.Event()
        event.set()

    async def wait_for_change(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._changed.wait(), timeout=max(0.0, timeout))
        except asyncio.TimeoutError:
            pass

    def _finish_if_done(self, job: Job) -> None:
        if job.terminal and job.finished is None:
            job.finished = time.time()
            log.info("job %s %s", job.id, job.state)

    def _prune(self) -> None:
        now = time.time()
        finished = sorted(
            (j for j in self._jobs.values() if j.terminal),
            key=lambda j: j.finished or j.created,
        )
        drop = [j for j in finished if now - (j.finished or j.created) > FINISHED_KEEP_S]
        excess = len(finished) - len(drop) - FINISHED_KEEP_MAX
        if excess > 0:
            drop += [j for j in finished if j not in drop][:excess]
        for job in drop:
            del self._jobs[job.id]

    # ---- persistence ----------------------------------------------------------

    def _serialise(self) -> str:
        """The file's contents, taken on the event loop so it is one moment's
        state. A finished workflow drops its graph — a batch of large
        workflows would otherwise grow the file with prompts nothing will
        run again — and keeps its outputs and error."""
        rows = []
        for job in self._jobs.values():
            rows.append({
                "id": job.id, "model": job.model,
                "created_wall": job.created, "started_wall": job.started,
                "finished_wall": job.finished, "cancelled": job.cancelled,
                "workflows": [
                    w.to_json(with_prompt=w.state not in TERMINAL) for w in job.workflows
                ],
            })
        return json.dumps({"jobs": rows, "handovers": self.handovers})

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            # No loop to hand the write to (a CLI, a test): write it here.
            self._write(self._serialise())
            return
        self._dirty = True
        if self._writer is None or self._writer.done():
            self._writer = loop.create_task(self._drain_writes())

    def _write(self, text: str) -> None:
        try:
            _write_file(self._path, text)
        except OSError as exc:
            log.warning("could not write job state to %s: %s", self._path, exc)

    async def _drain_writes(self) -> None:
        async with self._write_lock:
            while self._dirty:
                self._dirty = False
                await asyncio.to_thread(self._write, self._serialise())

    async def flush(self) -> None:
        """Wait for every change so far to reach the disk."""
        while self._writer is not None and not self._writer.done():
            await self._writer

    def _load(self) -> None:
        """Take back the queue. Pending stays pending; a running workflow
        keeps its prompt id, for the scheduler to look up once the workflow
        server is up — it cannot be settled from here.

        A file this version cannot read never stops startup: it is set aside
        beside itself, named, and the store starts empty. Silently dropping
        the rows that do not parse would lose jobs without saying which.
        """
        try:
            text = self._path.read_text()
        except FileNotFoundError:
            return
        except OSError as exc:
            log.warning("ignoring unreadable job state %s: %s", self._path, exc)
            return
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise _Unreadable("the top level is not an object")
            jobs = [_job_from(r) for r in _typed(data.get("jobs", []), list, "jobs")]
            handovers = _typed(data.get("handovers", []), list, "handovers")
        except (ValueError, TypeError) as exc:
            aside = self._path.with_name(f"{self._path.name}.corrupt-{int(time.time())}")
            try:
                os.replace(self._path, aside)
            except OSError:
                aside = None
            log.error(
                "%s is not a job file this vramux can read (%s) — %s; starting empty",
                self._path, exc,
                f"set aside as {aside.name}" if aside else "and it could not be set aside",
            )
            return
        for job in jobs:
            self._jobs[job.id] = job
            if not job.terminal:
                log.info("restored job %s (%s), %d workflow(s) open",
                         job.id, job.state, len(job.open_workflows()))
        self.handovers = handovers[-HANDOVER_KEEP:]

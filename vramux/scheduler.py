"""One card, two tenants: language models and render jobs, in time slices.

The card has two kinds of work that cannot share it. Language-model sessions
are many, short and latency-bound; a render takes the whole card for minutes.
Flipping the card per render request — which is what loading the stack on
every job did — paid a handover each way for each job, and a stack brought up
by a load nobody followed with a job sat on the card holding the language
models off it.

So the card has a *mode*, and one task owns it:

* **llm** — jobs queue (submitting never blocks) while the language models
  work. The card flips to render when jobs are pending and one of: the oldest
  has waited `render_start_deadline`; `render_batch_trigger` jobs are pending;
  or no language-model request is in flight or has arrived for
  `QUIET_GRACE`. An open render window also flips.
* **render** — the stack is acquired through the arbiter like any load (the
  language models drain and are evicted), and every pending workflow runs,
  one at a time, oldest job first. A language-model request that arrives
  parks behind the hold. The drain ends when nothing is pending, when a
  parked request has waited `llm_latency_bound` (after the running workflow,
  never in the middle of one), or when a window closes. Then the hold is
  released and the parked requests take the card.

A language-model client therefore waits at most one drain window, not one
handover per render. Both handovers are timed and kept, because whether a
batch trigger pays for its latency is a question only the measured cost of a
flip can answer.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp

from .backends import http_queue_depth
from .jobs import MAX_ATTEMPTS, RUNNING, Job, JobStore, Workflow
from .registry import ModelRegistry, ModelSpec
from .residency import ResidencyArbiter

log = logging.getLogger("vramux.scheduler")

LLM = "llm"
RENDER = "render"

# How long the card must have had no language-model request in flight or
# arriving before pending jobs take it without a typed trigger. Short: this is
# "nobody is using the card", not a bound anybody tunes. It is also what runs
# jobs at all when no bound is typed.
QUIET_GRACE = 10.0

# How often llm mode re-checks its triggers when nothing wakes it.
_TICK = 1.0

# How often a submitted prompt's history is read.
_HISTORY_POLL = 1.0

# How often an open window re-reads the stack's queue and history.
_WINDOW_POLL = 2.0

# A window held for N prompts also closes once the stack's queue has sat
# empty this long: the human it was opened for has walked away, and the
# language models parked behind it are not to wait for them to come back.
WINDOW_IDLE = 600.0

# How often the to_llm handover is sampled, and how long after a flip-back a
# following language-model load still counts as that handover.
_HANDOVER_POLL = 0.2
_HANDOVER_WATCH = 900.0

# After a flip that failed, or a stack that went away mid-drain, how long
# before trying again: doubling from `_FLIP_RETRY` per consecutive failure up
# to `_FLIP_RETRY_MAX`, reset by a flip that works. Without it a broken stack
# is a loop of loads, each one evicting the language models for nothing.
_FLIP_RETRY = 30.0
_FLIP_RETRY_MAX = 900.0

# Consecutive failed starts of the stack before the job at the head of its
# queue is failed instead of retried. A job whose stack cannot come up must
# not hold the card's language models hostage for ever.
_STARTS_PER_JOB = 3


def retry_after(failures: int) -> float:
    """Seconds to wait after `failures` consecutive failed flips."""
    return min(_FLIP_RETRY * 2 ** max(0, failures - 1), _FLIP_RETRY_MAX)

# A prompt absent from both history and queue this many reads in a row is
# gone, not between the two: the server moves a finished prompt from one to
# the other atomically, but two reads are not atomic.
_GONE_AFTER = 2

# Per request to the stack.
_HTTP_TIMEOUT = 60.0

# For a /prompt POST, which validates the whole graph before answering. A
# timeout here does not mean the prompt was not queued — so the marker it
# carries is looked for before anything is submitted again.
_PROMPT_TIMEOUT = 60.0


class _Lost(Exception):
    """The stack stopped answering mid-drain. Running work is settled on the
    next render, from its prompt id."""


def _median_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = [r["total_s"] for r in rows]
    return {
        "last": totals[-1] if totals else None,
        "median": round(statistics.median(totals), 1) if totals else None,
        "n": len(totals),
    }


def outputs_of(entry: Dict[str, Any], tag: str) -> List[Dict[str, Any]]:
    """Every file a finished prompt wrote, with the URL that fetches it.

    The history lists outputs per node, keyed by kind (images, gifs, ...);
    anything with a filename is a file. `view` is the stack's own relative
    URL; `proxy` the same through vramux's `/m/` route.
    """
    rows = []
    for node, out in (entry.get("outputs") or {}).items():
        if not isinstance(out, dict):
            continue
        for kind, items in out.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                query = urlencode({
                    "filename": item["filename"],
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output"),
                })
                rows.append({
                    "node": str(node), "kind": kind,
                    "filename": item["filename"],
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output"),
                    "view": f"/view?{query}",
                    "proxy": f"/m/{tag}/view?{query}",
                })
    return rows


def error_of(entry: Dict[str, Any]) -> Optional[str]:
    """The failure a history entry records, or None if it succeeded."""
    status = entry.get("status") or {}
    if status.get("status_str") != "error":
        return None
    for name, data in status.get("messages") or []:
        if name == "execution_error" and isinstance(data, dict):
            return (
                f"node {data.get('node_id')} ({data.get('node_type')}): "
                f"{data.get('exception_message', '').strip()}"
            )
    return "the workflow failed (no error message recorded)"


class Scheduler:
    """Owns the card's mode. Started and stopped by the router's lifecycle."""

    def __init__(
        self, registry: ModelRegistry, arbiter: ResidencyArbiter, store: JobStore,
    ) -> None:
        self.registry = registry
        self.arbiter = arbiter
        self.store = store
        self.mode = LLM
        self.render_model: Optional[str] = None
        self.window: Optional[Dict[str, Any]] = None
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._watch: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        # Monotonic time before which no trigger flips: after a failed flip,
        # or the slice owed to the language models after a latency-bound
        # flip-back — without it the deadline, already past, flips straight
        # back and the bound buys nothing.
        self._not_before = 0.0
        # Consecutive failed flips (backoff), and of those the failed starts.
        self._failures = 0
        self._failed_starts = 0
        # Every prompt id this process submitted, so a window counting a
        # human's prompts does not count vramux's own.
        self._submitted: set = set()
        # Requests parked when the last render let go, not yet admitted. No
        # flip starts while any is left: a chat request waits at most one
        # drain, never one per render queued behind it.
        self._owed: set = set()
        self._owed_logged = False

    # ---- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT)
            )
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        # The main task first: cancelled mid-render it releases the card and
        # starts a handover watch on the way out, which is then cancelled too.
        for name in ("_task", "_watch"):
            task = getattr(self, name)
            setattr(self, name, None)
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        if self._session is not None:
            await self._session.close()
            self._session = None
        await self.store.flush()

    def wake(self) -> None:
        self._wake.set()

    # ---- what can be scheduled ------------------------------------------------

    def job_spec(self, tag: str) -> Optional[ModelSpec]:
        spec = self.registry.get(tag)
        return spec if spec is not None and spec.jobs else None

    def _job_specs(self) -> List[ModelSpec]:
        return [s for s in self.registry.all() if s.jobs]

    # ---- the window -----------------------------------------------------------

    def open_window(self, spec: ModelSpec, jobs: Optional[int], minutes: Optional[float]) -> Dict:
        self.window = {
            "model": spec.tag, "jobs": jobs, "minutes": minutes,
            "opened": time.time(), "baseline": None, "counted": set(), "done": 0,
            "queue_empty_since": None,
        }
        log.info("render window opened on %s: %s", spec.tag,
                 f"{jobs} prompt(s)" if jobs is not None else f"{minutes:g} idle minute(s)")
        self.wake()
        return self.window_view()

    def close_window(self) -> bool:
        was = self.window is not None
        if was:
            log.info("render window on %s closed", self.window["model"])
        self.window = None
        self.wake()
        return was

    def window_view(self) -> Optional[Dict[str, Any]]:
        w = self.window
        if w is None:
            return None
        return {k: w[k] for k in ("model", "jobs", "minutes", "done")}

    # ---- the decision ---------------------------------------------------------

    def _llm_quiet(self, tag: str, now: float) -> bool:
        inflight = sum(r.inflight for r in self.arbiter.residents if r.tag != tag)
        if inflight or self.arbiter.arriving_except(tag):
            return False
        last = max(
            (t for k, t in self.arbiter.last_arrival.items() if k != tag), default=None,
        )
        return last is None or now - last >= QUIET_GRACE

    def decide(self) -> Optional[Tuple[ModelSpec, str]]:
        """Whether to flip to render now, for which model, and why."""
        now = time.monotonic()
        # First, window or not: a stack that failed to come up must not be
        # retried in a loop because somebody is holding a window open on it.
        if now < self._not_before:
            return None
        if self._owed:
            self._owed &= self.arbiter.unadmitted
            if self._owed:
                if not self._owed_logged:
                    self._owed_logged = True
                    log.info("next render deferred until the %d request(s) parked at "
                             "the last flip-back are admitted", len(self._owed))
                return None
        if self.window is not None:
            spec = self.job_spec(self.window["model"])
            if spec is not None:
                return spec, "window"
        wall = time.time()
        candidates = []
        for spec in self._job_specs():
            open_jobs = self.store.open_jobs(spec.tag)
            if open_jobs:
                candidates.append((open_jobs[0].created, spec, open_jobs))
        for _created, spec, open_jobs in sorted(candidates, key=lambda c: c[0]):
            age = wall - open_jobs[0].created
            if spec.render_start_deadline is not None and age >= spec.render_start_deadline:
                return spec, f"oldest job waited {age:.0f}s"
            if spec.render_batch_trigger is not None and len(open_jobs) >= spec.render_batch_trigger:
                return spec, f"{len(open_jobs)} job(s) pending"
            if self._llm_quiet(spec.tag, now):
                return spec, "language models quiet"
        return None

    async def _run(self) -> None:
        while True:
            try:
                choice = self.decide()
                if choice is None:
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=_TICK)
                    except asyncio.TimeoutError:
                        pass
                    continue
                await self._render(*choice)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("scheduler step failed")
                self._failed()
                await asyncio.sleep(_TICK)

    def _failed(self) -> float:
        """Count a failed flip and back off for it. Returns the wait."""
        self._failures += 1
        wait = retry_after(self._failures)
        self._not_before = time.monotonic() + wait
        return wait

    # ---- render mode ----------------------------------------------------------

    def _pending_workflows(self, tag: Optional[str] = None) -> int:
        return sum(len(j.open_workflows()) for j in self.store.open_jobs(tag))

    async def _render(self, spec: ModelSpec, trigger: str) -> None:
        pending = self._pending_workflows(spec.tag)
        # The drain runs what was pending when the flip was decided, and no
        # more: a steady trickle of submits must not hold the language models
        # off the card for ever. What arrives meanwhile waits for the next flip.
        batch = self.store.open_keys(spec.tag)
        decided = time.monotonic()
        self.mode, self.render_model = RENDER, spec.tag
        self.arbiter.hold_render(spec.tag)
        if self._watch is not None:
            self._watch.cancel()
        timings: Dict[str, float] = {}
        try:
            await self.arbiter.acquire(spec, scheduled=True, timings=timings)
        except BaseException as exc:
            # Cancellation included: a hold left behind parks every chat
            # request until the process exits.
            self.arbiter.release_render()
            self.mode, self.render_model = LLM, None
            if not isinstance(exc, Exception):
                raise
            wait = self._failed()
            log.warning("could not flip to render for %s (failure %d in a row): %s "
                        "— retrying in %.0fs", spec.tag, self._failures, exc, wait)
            # Only the stack failing to come up counts against a job. A lease
            # that outranks the load, or a load still in progress, is the card
            # being busy: back off, but nobody's job is at fault.
            if timings.get("stage") == "start":
                self._failed_starts += 1
                if self._failed_starts % _STARTS_PER_JOB == 0:
                    self._fail_head_job(spec, self._failed_starts)
            return
        self._failures = self._failed_starts = 0
        total = time.monotonic() - decided
        self._handover("to_render", total, timings.get("evict_s", 0.0),
                       timings.get("load_s", 0.0), pending, trigger)
        reason = "drained"
        try:
            reason = await self._drain(spec, batch)
        except _Lost as exc:
            reason = f"lost the stack: {exc}"
            self._failed()
        finally:
            released = time.monotonic()
            parked = len(self.arbiter.parked)
            self._owed, self._owed_logged = self.arbiter.parked_tokens, False
            self.arbiter.release(spec.tag, scheduled=True)
            self.arbiter.release_render()
            self.mode, self.render_model = LLM, None
            log.info("render over on %s (%s); %d request(s) were parked",
                     spec.tag, reason, parked)
            self._watch = asyncio.create_task(self._time_to_llm(spec.tag, released, parked))

    def _fail_head_job(self, spec: ModelSpec, starts: int) -> None:
        open_jobs = self.store.open_jobs(spec.tag)
        if not open_jobs:
            return
        head = open_jobs[0]
        reason = f"stack failed to start {starts} times"
        failed = 0
        for wf in head.open_workflows():
            self.store.mark_failed(head, wf, reason)
            failed += 1
        log.error("job %s: %s failed to start %d times in a row — failing its %d "
                  "open workflow(s)", head.id, spec.tag, starts, failed)

    async def _drain(self, spec: ModelSpec, batch: set) -> str:
        try:
            reason = await self._drain_batch(spec, batch)
        finally:
            deferred = len(self.store.open_keys(spec.tag) - batch)
            if deferred:
                log.info("%d workflow(s) submitted during it wait for the next flip", deferred)
        return reason

    async def _drain_batch(self, spec: ModelSpec, batch: set) -> str:
        base = self.arbiter.upstream_for(spec.tag)
        ran = 0
        while True:
            if spec.tag not in {r.tag for r in self.arbiter.residents}:
                raise _Lost(f"{spec.tag} is no longer resident")
            window = self.window if (self.window and self.window["model"] == spec.tag) else None
            if window is not None:
                # Counted from when the window met the stack, so prompts that
                # finished before it opened are not counted towards it.
                await self._window_baseline(base)
            if window is not None:
                # The card is held for the window anyway: jobs that arrive
                # during it run during it, rather than flipping again after.
                batch |= self.store.open_keys(spec.tag)
            bound = spec.llm_latency_bound
            parked = self.arbiter.parked
            # Not before one workflow has run: a flip that took longer than
            # the bound would otherwise be paid for and thrown away.
            if window is None and bound is not None and parked and ran:
                waited = time.monotonic() - parked[0][1]
                if waited >= bound:
                    # The latency bound is checked between workflows only: a
                    # prompt already on the stack is not abandoned for it.
                    self._not_before = time.monotonic() + bound
                    log.info("%s has waited %.0fs behind render (bound %.0fs); "
                             "flipping back with %d workflow(s) still pending",
                             parked[0][0], waited, bound, self._pending_workflows(spec.tag))
                    return "llm latency bound"
            nxt = self.store.next_workflow(spec.tag, only=batch)
            if nxt is not None:
                await self._run_charged(spec, base, *nxt)
                ran += 1
                continue
            if window is None:
                if not ran:
                    # Flipped for work there turned out to be none of — the
                    # same flip would be chosen again at once, in a loop that
                    # never yields. Back off as for a failure.
                    wait = self._failed()
                    log.warning("render on %s ran nothing; not flipping again for %.0fs",
                                spec.tag, wait)
                    return "nothing to run"
                return "drained"
            if await self._window_done(base, window):
                self.window = None
                log.info("render window on %s closed", spec.tag)
                return "window closed"
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=_WINDOW_POLL)
            except asyncio.TimeoutError:
                pass

    # ---- one workflow ---------------------------------------------------------

    async def _run_charged(self, spec: ModelSpec, base: str, job: Job, wf: Workflow) -> None:
        """Run one workflow; any failure that is not the stack going away
        costs it an attempt, so a workflow that breaks the scheduler every
        time it is looked at ends failed, not retried for ever."""
        before = wf.attempts
        try:
            await self._run_workflow(spec, base, job, wf)
        except (_Lost, asyncio.CancelledError):
            raise
        except Exception as exc:
            if wf.attempts == before:
                wf.attempts += 1
            log.exception("job %s workflow %d raised (attempt %d of %d)",
                          job.id, wf.index, wf.attempts, MAX_ATTEMPTS)
            if wf.attempts >= MAX_ATTEMPTS:
                self.store.mark_failed(job, wf, f"{type(exc).__name__}: {exc}")
            else:
                self.store.changed()

    async def _run_workflow(self, spec: ModelSpec, base: str, job: Job, wf: Workflow) -> None:
        if wf.state == RUNNING and wf.prompt_id:
            # Submitted before, result unseen: a restart, or a stack that went
            # away. Look before submitting again.
            where = await self._locate(base, wf.prompt_id)
            if isinstance(where, dict):
                self._settle(spec, job, wf, where)
                return
            if where == "queued":
                await self._follow(spec, base, job, wf)
                return
            log.info("job %s workflow %d: prompt %s is gone from %s",
                     job.id, wf.index, wf.prompt_id, spec.tag)
        elif wf.state == RUNNING and await self._adopt(base, job, wf):
            # Sent, but the answer never came back (a timeout, or a crash
            # between the POST and its reply): the stack may well have it.
            await self._follow(spec, base, job, wf)
            return
        if wf.attempts >= MAX_ATTEMPTS:
            self.store.mark_failed(
                job, wf, f"lost by {spec.tag} after {wf.attempts} submission(s) "
                         f"(it restarted or dropped the prompt)",
            )
            return
        await self._submit(spec, base, job, wf)
        if wf.state == RUNNING:
            await self._follow(spec, base, job, wf)

    async def _submit(self, spec: ModelSpec, base: str, job: Job, wf: Workflow) -> None:
        wf.attempts += 1
        wf.prompt_id = None
        # Persisted before the POST: a crash between the two reads as running
        # with no prompt id, which is resubmitted, never silently dropped.
        self.store.mark_running(job, wf)
        assert self._session is not None
        try:
            async with self._session.post(
                f"{base}/prompt",
                json={
                    "prompt": wf.prompt, "client_id": f"vramux-{job.id}",
                    # Kept by the stack with the prompt, in its queue and its
                    # history: how a submit whose answer was lost is found.
                    "extra_data": {"vramux": self._marker(job, wf)},
                },
                timeout=aiohttp.ClientTimeout(total=_PROMPT_TIMEOUT),
            ) as resp:
                body = await resp.json(content_type=None)
                status = resp.status
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            if await self._adopt(base, job, wf):
                return
            raise _Lost(f"POST /prompt failed: {exc}") from None
        if status != 200 or not isinstance(body, dict) or not body.get("prompt_id"):
            # Refused at validation: the graph is wrong, and submitting it
            # again would be refused again.
            detail = body.get("error") if isinstance(body, dict) else body
            if isinstance(detail, dict):
                detail = detail.get("message") or detail
            node_errors = body.get("node_errors") if isinstance(body, dict) else None
            text = f"{spec.tag} refused the workflow ({status}): {detail}"
            if node_errors:
                text += f"; node errors: {node_errors}"
            self.store.mark_failed(job, wf, text)
            return
        wf.prompt_id = str(body["prompt_id"])
        self._submitted.add(wf.prompt_id)
        self.store.changed()
        log.info("job %s workflow %d submitted to %s as %s",
                 job.id, wf.index, spec.tag, wf.prompt_id)

    @staticmethod
    def _marker(job: Job, wf: Workflow) -> Dict[str, Any]:
        return {"job": job.id, "workflow": wf.index}

    async def _adopt(self, base: str, job: Job, wf: Workflow) -> bool:
        """Find this workflow's prompt on the stack by its marker and take
        its id. True if found. Rows in both the queue and the history carry
        the prompt as [number, prompt_id, graph, extra_data, outputs]."""
        marker = self._marker(job, wf)

        def mine(row) -> bool:
            return (
                isinstance(row, list) and len(row) > 3 and isinstance(row[3], dict)
                and row[3].get("vramux") == marker
            )

        found = None
        queue = await self._get_json(f"{base}/queue")
        for rows in (queue.values() if isinstance(queue, dict) else []):
            for row in rows if isinstance(rows, list) else []:
                if mine(row):
                    found = str(row[1])
        if found is None:
            history = await self._get_json(f"{base}/history")
            for pid, entry in (history.items() if isinstance(history, dict) else []):
                if isinstance(entry, dict) and mine(entry.get("prompt")):
                    found = str(pid)
        if found is None:
            return False
        wf.prompt_id = found
        self._submitted.add(found)
        self.store.changed()
        log.info("job %s workflow %d: found its prompt on the stack as %s; "
                 "not submitting it again", job.id, wf.index, found)
        return True

    async def _follow(self, spec: ModelSpec, base: str, job: Job, wf: Workflow) -> None:
        """Wait for the prompt's history entry. No cap: a render is as long
        as it is, and a stack that dies is noticed by its health, not a clock."""
        absent = 0
        while True:
            await asyncio.sleep(_HISTORY_POLL)
            where = await self._locate(base, wf.prompt_id)
            if isinstance(where, dict):
                self._settle(spec, job, wf, where)
                return
            if where == "queued":
                absent = 0
                continue
            absent += 1
            if absent >= _GONE_AFTER:
                if wf.attempts >= MAX_ATTEMPTS:
                    self.store.mark_failed(
                        job, wf, f"{spec.tag} dropped the prompt after "
                                 f"{wf.attempts} submission(s)",
                    )
                    return
                log.info("job %s workflow %d: %s dropped prompt %s; resubmitting",
                         job.id, wf.index, spec.tag, wf.prompt_id)
                await self._submit(spec, base, job, wf)
                if wf.state != RUNNING:
                    return
                absent = 0

    def _settle(self, spec: ModelSpec, job: Job, wf: Workflow, entry: Dict) -> None:
        error = error_of(entry)
        if error is not None:
            self.store.mark_failed(job, wf, error)
            log.info("job %s workflow %d failed: %s", job.id, wf.index, error)
            return
        outputs = outputs_of(entry, spec.tag)
        self.store.mark_done(job, wf, outputs)
        log.info("job %s workflow %d done, %d output(s)", job.id, wf.index, len(outputs))

    async def _get_json(self, url: str) -> Any:
        assert self._session is not None
        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    raise _Lost(f"GET {url} answered {resp.status}")
                return await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            raise _Lost(f"GET {url} failed: {exc}") from None

    async def _locate(self, base: str, prompt_id: str):
        """A prompt's history entry (dict), "queued", or "absent"."""
        history = await self._get_json(f"{base}/history/{prompt_id}")
        if isinstance(history, dict) and isinstance(history.get(prompt_id), dict):
            return history[prompt_id]
        queue = await self._get_json(f"{base}/queue")
        if isinstance(queue, dict):
            for items in queue.values():
                for item in items if isinstance(items, list) else []:
                    # Queue rows are [number, prompt_id, prompt, extra, outputs].
                    if isinstance(item, list) and len(item) > 1 and item[1] == prompt_id:
                        return "queued"
        return "absent"

    # ---- the window, while rendering ------------------------------------------

    async def _window_baseline(self, base: str) -> None:
        window = self.window
        if window is None or window["baseline"] is not None:
            return
        history = await self._get_json(f"{base}/history")
        window["baseline"] = set(history) if isinstance(history, dict) else set()

    async def _window_done(self, base: str, window: Dict[str, Any]) -> bool:
        depth = await http_queue_depth(base, "/queue")
        now = time.monotonic()
        if depth:
            window["queue_empty_since"] = None
        elif window["queue_empty_since"] is None:
            window["queue_empty_since"] = now
        idle = 0.0 if depth else now - window["queue_empty_since"]
        if window["jobs"] is not None:
            history = await self._get_json(f"{base}/history")
            present = set(history) if isinstance(history, dict) else set()
            # Rebased on what is still there: a history that shrank (cleared
            # by hand, trimmed by the stack) must not leave the count waiting
            # for entries that no longer exist.
            window["baseline"] &= present
            ours = self._submitted | {
                w.prompt_id for j in self.store.all() for w in j.workflows if w.prompt_id
            }
            window["counted"] |= present - window["baseline"] - ours
            window["done"] = len(window["counted"])
            if window["done"] >= window["jobs"]:
                return True
            if idle >= WINDOW_IDLE:
                log.info("render window on %s: queue empty %.0fs with %d of %d prompt(s) "
                         "done — closing it", window["model"], idle,
                         window["done"], window["jobs"])
                return True
            return False
        return idle >= window["minutes"] * 60.0

    # ---- handovers ------------------------------------------------------------

    def _handover(self, direction: str, total: float, evict: float, load: float,
                  pending: int, trigger: str) -> None:
        arrow = "llm->render" if direction == "to_render" else "render->llm"
        log.info("handover %s %.1fs (evict %.1fs, load %.1fs), %d workflow(s) pending%s",
                 arrow, total, evict, load, pending, f" — {trigger}" if trigger else "")
        self.store.record_handover({
            "dir": direction, "at": time.time(), "total_s": round(total, 1),
            "evict_s": round(evict, 1), "load_s": round(load, 1),
            "pending": pending, "trigger": trigger,
        })

    async def _time_to_llm(self, tag: str, released: float, parked: int) -> None:
        """Time the card going back: release → stack gone → a model ready.

        Starts at the release if somebody was parked, else at the first
        language-model request after it — a card nobody asks for is not a
        handover anybody waited on. Gives up quietly after `_HANDOVER_WATCH`
        or if the card flips to render again first.
        """
        deadline = released + _HANDOVER_WATCH
        start = released if parked else None
        gone = None
        while time.monotonic() < deadline and self.mode == LLM:
            now = time.monotonic()
            if start is None:
                asks = [t for k, t in self.arbiter.last_arrival.items()
                        if k != tag and t >= released]
                if asks:
                    start = min(asks)
            residents = {r.tag: r for r in self.arbiter.residents}
            if gone is None and tag not in residents:
                gone = now
            others = [r for t, r in residents.items() if t != tag and r.backend.alive()]
            if start is not None and gone is not None and others and self.arbiter.loading is None:
                self._handover("to_llm", now - start, max(0.0, gone - start),
                               now - max(gone, start),
                               self._pending_workflows(), "")
                return
            await asyncio.sleep(_HANDOVER_POLL)

    # ---- what the console draws -----------------------------------------------

    def state(self) -> Dict[str, Any]:
        open_jobs = self.store.open_jobs()
        oldest = min((j.created for j in open_jobs), default=None)
        specs = self._job_specs()
        spec = self.job_spec(self.render_model) if self.render_model else (
            specs[0] if specs else None)
        rows = self.store.handovers
        return {
            "mode": self.mode,
            "model": self.render_model,
            "pending_jobs": len(open_jobs),
            "pending_workflows": self._pending_workflows(),
            "oldest_age_s": round(time.time() - oldest) if oldest is not None else None,
            "bounds": {
                "model": spec.tag if spec else None,
                "render_start_deadline_s": spec.render_start_deadline if spec else None,
                "render_batch_trigger": spec.render_batch_trigger if spec else None,
                "llm_latency_bound_s": spec.llm_latency_bound if spec else None,
            },
            "parked": len(self.arbiter.parked),
            "window": self.window_view(),
            "handover": {
                "to_render": _median_row([r for r in rows if r.get("dir") == "to_render"]),
                "to_llm": _median_row([r for r in rows if r.get("dir") == "to_llm"]),
            },
        }

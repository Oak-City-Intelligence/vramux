"""The render scheduler: one card, two tenants, against a fake workflow server.

The arbiter is the fake one the residency tests use, so the event log is the
record of every handover; the workflow server is a real aiohttp server
answering /prompt, /history, /queue and /system_stats, because the scheduler's
job is to talk to one.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from vramux import residency, scheduler as sched_mod
from vramux.jobs import DONE, FAILED, PENDING, RUNNING, JobStore
from vramux.registry import KIND_DOCKER
from vramux.router import ROUTER, make_app
from vramux.scheduler import LLM, RENDER, Scheduler

from test_polite_exclusive import QueueArbiter
from test_residency import spec

STACK = "stack:1"


class FakeStack:
    """A workflow server: prompts queue, run one at a time, land in history."""

    def __init__(self) -> None:
        self.prompts = []          # every prompt body received, in order
        self.queue = []            # [number, prompt_id] rows, running first
        self.history = {}
        self.render_s = 0.0
        self.gate = None           # an Event a test holds a render on
        self.fail = set()          # submission numbers that error
        self.refuse = set()        # submission numbers refused at /prompt
        self.slow_post = 0.0       # seconds /prompt sits on its answer
        self.extra = {}            # prompt_id -> extra_data, as the stack keeps it
        self._n = 0

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/prompt", self.post_prompt)
        app.router.add_get("/history", self.get_history)
        app.router.add_get("/history/{pid}", self.get_one)
        app.router.add_get("/queue", self.get_queue)
        app.router.add_get("/system_stats", self.get_stats)
        return app

    async def get_stats(self, _request):
        return web.json_response({})

    async def post_prompt(self, request):
        body = await request.json()
        n = self._n
        self._n += 1
        self.prompts.append(body["prompt"])
        if n in self.refuse:
            return web.json_response(
                {"error": {"message": "Prompt outputs failed validation"},
                 "node_errors": {"3": "bad"}}, status=400,
            )
        pid = f"p{n}"
        extra = dict(body.get("extra_data") or {})
        if "client_id" in body:
            extra["client_id"] = body["client_id"]
        self.extra[pid] = extra
        self.queue.append([n, pid])
        asyncio.get_running_loop().create_task(self._render(n, pid))
        if self.slow_post:
            await asyncio.sleep(self.slow_post)   # queued, but the answer is late
        return web.json_response({"prompt_id": pid, "number": n})

    async def _render(self, n, pid):
        if self.gate is not None:
            await self.gate.wait()
        await asyncio.sleep(self.render_s)
        self.complete(pid, error=n in self.fail)

    def complete(self, pid, error=False):
        self.queue = [row for row in self.queue if row[1] != pid]
        row = [0, pid, {}, self.extra.get(pid, {}), []]
        if error:
            self.history[pid] = {"prompt": row, "outputs": {}, "status": {
                "status_str": "error", "completed": False,
                "messages": [["execution_error", {
                    "node_id": "7", "node_type": "KSampler",
                    "exception_message": "out of memory\n"}]],
            }}
        else:
            self.history[pid] = {
                "prompt": row,
                "outputs": {"9": {"images": [
                    {"filename": f"{pid}.png", "subfolder": "", "type": "output"}]}},
                "status": {"status_str": "success", "completed": True, "messages": []},
            }

    async def get_history(self, _request):
        return web.json_response(self.history)

    async def get_one(self, request):
        pid = request.match_info["pid"]
        return web.json_response({pid: self.history[pid]} if pid in self.history else {})

    async def get_queue(self, _request):
        return web.json_response({
            "queue_running": [r + [{}, self.extra.get(r[1], {}), []] for r in self.queue[:1]],
            "queue_pending": [r + [{}, self.extra.get(r[1], {}), []] for r in self.queue[1:]],
        })


class StackArbiter(QueueArbiter):
    def __init__(self, upstream: str, **kw):
        super().__init__(**kw)
        self._stack_url = upstream

    def upstream_for(self, tag: str) -> str:
        return self._stack_url if tag == STACK else super().upstream_for(tag)


class Reg:
    def __init__(self, *specs):
        self._specs = {s.tag: s for s in specs}

    def get(self, tag):
        return self._specs.get(tag)

    def all(self):
        return list(self._specs.values())


def stack_spec(**kw):
    return spec(STACK, kind=KIND_DOCKER, exclusive=True, idle_timeout=0,
                busy_path="/queue", jobs="comfy", **kw)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(residency, "_BUSY_POLL", 0.01)
    monkeypatch.setattr(sched_mod, "_TICK", 0.02)
    monkeypatch.setattr(sched_mod, "_HISTORY_POLL", 0.01)
    monkeypatch.setattr(sched_mod, "_WINDOW_POLL", 0.02)
    monkeypatch.setattr(sched_mod, "_HANDOVER_POLL", 0.01)
    monkeypatch.setattr(sched_mod, "_FLIP_RETRY", 0.1)
    # Off unless a test turns it down: most tests want a typed trigger to be
    # the only thing that flips the card.
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 3600.0)


@pytest.fixture
async def rig(tmp_path):
    """A fake stack on a socket, an arbiter, a store and a scheduler."""
    fake = FakeStack()
    server = TestServer(fake.app())
    await server.start_server()
    made = []

    def build(stack=None, store=None, llm_inflight=False):
        arbiter = StackArbiter(
            str(server.make_url("")).rstrip("/"),
            drain_timeout=0.2, startup_timeout=5.0, busy_wait=5.0,
        )
        s = stack or stack_spec()
        st = store or JobStore(tmp_path / "jobs.json")
        sch = Scheduler(Reg(s, spec("llm:27b")), arbiter, st)
        made.append(sch)
        return arbiter, s, st, sch

    yield fake, build
    for sch in made:
        await sch.close()
    await server.close()


async def until(cond, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


async def llm_resident(arbiter, hold=False):
    llm = spec("llm:27b")
    await arbiter.acquire(llm)
    if not hold:
        arbiter.release("llm:27b")
    return llm


# ---- submitting -------------------------------------------------------------


async def test_submit_while_an_llm_works_answers_at_once(rig):
    fake, build = rig
    arbiter, stack, store, sch = build()
    await llm_resident(arbiter, hold=True)      # a stream in flight
    app = make_app(Reg(stack, spec("llm:27b")), arbiter, scheduler=sch)
    async with TestClient(TestServer(app)) as client:
        started = time.monotonic()
        resp = await client.post("/gpu/jobs", json={"model": STACK, "workflow": {"1": {}}})
        assert resp.status == 202
        body = await resp.json()
        assert time.monotonic() - started < 1.0
        assert body["state"] == PENDING and body["position"] == 0
        await asyncio.sleep(0.2)
        assert sch.mode == LLM and arbiter.current_tag == "llm:27b"
        assert fake.prompts == []


@pytest.mark.parametrize("body, needle", [
    ({"model": "ghost:1", "workflows": [{"1": {}}]}, "not found"),
    ({"model": "llm:27b", "workflows": [{"1": {}}]}, "does not take jobs"),
    ({"model": STACK, "workflows": []}, "non-empty list"),
    ({"model": STACK, "workflows": ["not an object"]}, "workflow 0"),
])
async def test_a_bad_submit_is_400(rig, body, needle):
    _fake, build = rig
    arbiter, stack, _store, sch = build()
    app = make_app(Reg(stack, spec("llm:27b")), arbiter, scheduler=sch)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/gpu/jobs", json=body)
        assert resp.status == 400
        assert needle in (await resp.json())["error"]


# ---- one flip each way ------------------------------------------------------


async def test_a_batch_runs_under_one_flip_each_way(rig, monkeypatch):
    fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.05)
    arbiter, stack, store, sch = build()
    await llm_resident(arbiter)
    fake.render_s = 0.05
    job = store.submit(STACK, [{"a": {}}, {"b": {}}, {"c": {}}])
    sch.start()
    await until(lambda: sch.mode == RENDER)
    # A chat request arriving mid-render parks behind it, uncapped.
    chat = asyncio.create_task(arbiter.acquire(spec("llm:27b")))
    await asyncio.wait_for(chat, 5)
    arbiter.release("llm:27b")

    assert job.state == DONE
    assert fake.prompts == [{"a": {}}, {"b": {}}, {"c": {}}]
    assert arbiter.events == [
        "start:llm:27b", "stop:llm:27b", "start:stack:1",   # one flip to render
        "stop:stack:1", "start:llm:27b",                    # one flip back
    ]
    await until(lambda: sch.state()["handover"]["to_llm"]["n"] == 1)
    state = sch.state()
    assert state["handover"]["to_render"]["n"] == 1
    assert state["mode"] == LLM and state["pending_workflows"] == 0


async def test_the_deadline_flips_a_busy_card(rig):
    fake, build = rig
    arbiter, _stack, store, sch = build(stack=stack_spec(render_start_deadline=0.3))
    await llm_resident(arbiter, hold=True)       # never quiet
    store.submit(STACK, [{"a": {}}])
    sch.start()
    await asyncio.sleep(0.15)
    assert sch.mode == LLM and fake.prompts == [], "held while the LLM works"
    await until(lambda: fake.prompts)
    assert store.handovers[-1]["trigger"].startswith("oldest job waited")


async def test_the_batch_trigger_flips_on_the_nth_job(rig):
    fake, build = rig
    arbiter, _stack, store, sch = build(stack=stack_spec(render_batch_trigger=2))
    await llm_resident(arbiter, hold=True)
    store.submit(STACK, [{"a": {}}])
    sch.start()
    await asyncio.sleep(0.15)
    assert fake.prompts == []
    store.submit(STACK, [{"b": {}}])
    sch.wake()
    await until(lambda: len(fake.prompts) == 2)
    assert store.handovers[-1]["trigger"] == "2 job(s) pending"


async def test_the_latency_bound_stops_after_the_running_workflow(rig):
    fake, build = rig
    arbiter, _stack, store, sch = build(stack=stack_spec(
        render_batch_trigger=1, llm_latency_bound=0.1))
    fake.gate = asyncio.Event()
    job = store.submit(STACK, [{"a": {}}, {"b": {}}, {"c": {}}])
    sch.start()
    await until(lambda: fake.prompts)
    chat = asyncio.create_task(arbiter.acquire(spec("llm:27b")))
    await asyncio.sleep(0.3)                     # past the bound, mid-workflow
    assert not chat.done(), "the running workflow is not abandoned"
    fake.gate.set()
    await asyncio.wait_for(chat, 5)
    assert len(fake.prompts) == 1, "nothing more was submitted once the bound tripped"
    assert [w.state for w in job.workflows] == [DONE, PENDING, PENDING]
    assert sch.mode == LLM


# ---- results ----------------------------------------------------------------


async def test_wait_returns_the_outputs(rig, monkeypatch):
    fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)
    arbiter, stack, store, sch = build()
    fake.fail = {1}
    app = make_app(Reg(stack, spec("llm:27b")), arbiter, scheduler=sch)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/gpu/jobs", json={"model": STACK, "workflows": [{"a": {}}, {"b": {}}]})
        job_id = (await resp.json())["id"]
        resp = await client.get(f"/gpu/jobs/{job_id}?wait=1&timeout=5")
        body = await resp.json()
    assert body["state"] == FAILED
    ok, bad = body["workflows"]
    assert ok["state"] == DONE and ok["outputs"] == [{
        "node": "9", "kind": "images", "filename": "p0.png", "subfolder": "",
        "type": "output", "view": "/view?filename=p0.png&subfolder=&type=output",
        "proxy": f"/m/{STACK}/view?filename=p0.png&subfolder=&type=output",
    }]
    assert bad["state"] == FAILED and "KSampler" in bad["error"] and "out of memory" in bad["error"]


async def test_wait_times_out_with_the_same_body(rig):
    _fake, build = rig
    arbiter, stack, _store, sch = build()
    await llm_resident(arbiter, hold=True)       # the card is busy: nothing flips
    app = make_app(Reg(stack, spec("llm:27b")), arbiter, scheduler=sch)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/gpu/jobs", json={"model": STACK, "workflow": {"a": {}}})
        job_id = (await resp.json())["id"]
        resp = await client.get(f"/gpu/jobs/{job_id}?wait=1&timeout=0.1")
        assert resp.status == 200 and (await resp.json())["state"] == PENDING
        resp = await client.delete(f"/gpu/jobs/{job_id}")
        assert (await resp.json())["state"] == "cancelled"


async def test_a_refused_workflow_fails_with_the_reason(rig, monkeypatch):
    fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)
    _arbiter, _stack, store, sch = build()
    fake.refuse = {0}
    job = store.submit(STACK, [{"a": {}}])
    sch.start()
    await until(lambda: job.terminal)
    assert "refused the workflow (400)" in job.workflows[0].error
    assert job.workflows[0].attempts == 1, "a graph refused once is refused again"


# ---- restart ----------------------------------------------------------------


async def test_a_restart_keeps_pending_and_settles_running(rig, tmp_path, monkeypatch):
    fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)
    path = tmp_path / "jobs.json"
    first = JobStore(path)
    job = first.submit(STACK, [{"a": {}}, {"b": {}}, {"c": {}}, {"d": {}}])
    a, b, c, d = job.workflows
    for wf, pid, attempts in ((a, "old-done", 1), (b, "old-gone", 1), (d, "old-lost", 2)):
        first.mark_running(job, wf)
        wf.prompt_id, wf.attempts = pid, attempts
    first.changed()
    await first.flush()
    # The stack finished `a` before the restart; `b` and `d` it never kept.
    fake.history["old-done"] = {
        "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}},
        "status": {"status_str": "success", "completed": True, "messages": []},
    }

    store = JobStore(path)
    restored = store.get(job.id)
    assert [w.state for w in restored.workflows] == [RUNNING, RUNNING, PENDING, RUNNING]
    _arbiter, _stack, _st, sch = build(store=store)
    sch.start()
    await until(lambda: restored.terminal)
    a, b, c, d = restored.workflows
    assert a.state == DONE and a.outputs[0]["filename"] == "a.png" and a.attempts == 1
    assert b.state == DONE and b.attempts == 2, "gone once: resubmitted once"
    assert c.state == DONE and c.attempts == 1
    assert d.state == FAILED and "after 2 submission(s)" in d.error
    assert fake.prompts == [{"b": {}}, {"c": {}}]


# ---- lend_after -------------------------------------------------------------


async def test_lend_after_is_waived_for_the_scheduler_and_kept_for_a_plain_load(rig, monkeypatch):
    fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)
    arbiter, stack, store, sch = build(stack=stack_spec(lend_after=60, restore_after=60))
    job = store.submit(STACK, [{"a": {}}])
    sch.start()
    await until(lambda: job.terminal and sch.mode == LLM)
    await asyncio.wait_for(arbiter.acquire(spec("llm:27b")), 1.0)
    arbiter.release("llm:27b")
    assert arbiter.current_tag == "llm:27b" and not arbiter._displaced

    await sch.close()                             # nothing scheduled from here
    await arbiter.acquire(stack)                  # an /api/generate warm-up
    arbiter.release(STACK)
    chat = asyncio.create_task(arbiter.acquire(spec("llm:27b")))
    await asyncio.sleep(0.2)
    assert not chat.done(), "a load nobody scheduled keeps its lend_after"
    chat.cancel()


# ---- windows ----------------------------------------------------------------


async def test_a_window_by_count_holds_until_n_prompts_complete(rig):
    fake, build = rig
    arbiter, stack, _store, sch = build()
    await llm_resident(arbiter, hold=False)
    fake.history["before"] = {"outputs": {}, "status": {}}
    sch.start()
    sch.open_window(stack, jobs=2, minutes=None)
    await until(lambda: sch.mode == RENDER and sch.window["baseline"] is not None)
    fake.complete("human-1")
    await asyncio.sleep(0.1)
    assert sch.mode == RENDER, "one of two"
    fake.complete("human-2")
    await until(lambda: sch.mode == LLM)
    assert sch.window is None


async def test_a_window_by_minutes_closes_after_an_idle_queue(rig):
    fake, build = rig
    arbiter, stack, _store, sch = build()
    sch.start()
    fake.queue = [[0, "human"]]                   # somebody's prompt is running
    sch.open_window(stack, jobs=None, minutes=0.3 / 60)
    await until(lambda: sch.mode == RENDER)
    await asyncio.sleep(0.5)
    assert sch.mode == RENDER, "a busy queue keeps the window open"
    fake.queue = []
    await until(lambda: sch.mode == LLM)


async def test_closing_a_window_by_hand_flips_back(rig):
    _fake, build = rig
    arbiter, stack, _store, sch = build()
    app = make_app(Reg(stack, spec("llm:27b")), arbiter, scheduler=sch)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/gpu/jobs/window", json={"model": STACK, "minutes": 10})
        assert resp.status == 200
        await until(lambda: sch.mode == RENDER)
        resp = await client.delete("/gpu/jobs/window")
        assert (await resp.json()) == {"closed": True}
        await until(lambda: sch.mode == LLM)
        resp = await client.post("/gpu/jobs/window", json={"model": STACK, "jobs": 1, "minutes": 1})
        assert resp.status == 400


async def test_a_stack_that_will_not_start_releases_the_hold(rig, monkeypatch):
    _fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)
    arbiter, _stack, store, sch = build()
    arbiter.fail_next_start = True
    store.submit(STACK, [{"a": {}}])
    sch.start()
    await until(lambda: "start-failed:stack:1" in arbiter.events)
    arbiter.fail_next_start = False     # QueueArbiter does not reset it itself
    await until(lambda: sch.mode == LLM and arbiter.render_hold is None)
    await asyncio.wait_for(arbiter.acquire(spec("llm:27b")), 1.0)
    arbiter.release("llm:27b")


async def test_closing_mid_render_leaves_no_hold(rig, monkeypatch):
    fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)
    arbiter, _stack, store, sch = build()
    fake.gate = asyncio.Event()
    store.submit(STACK, [{"a": {}}])
    sch.start()
    await until(lambda: fake.prompts)
    await sch.close()
    assert arbiter.render_hold is None and sch._watch is None


async def test_a_waiter_whose_client_left_is_dropped(rig):
    _fake, build = rig
    arbiter, stack, store, sch = build()
    job = store.submit(STACK, [{"a": {}}])
    router = make_app(Reg(stack, spec("llm:27b")), arbiter, scheduler=sch)[ROUTER]

    class Closed:
        def is_closing(self):
            return True

    request = SimpleNamespace(
        match_info={"job_id": job.id}, query={"wait": "1", "timeout": "5"},
        transport=Closed(),
    )
    started = time.monotonic()
    resp = await router.job_get(request)
    assert resp.status == 499 and time.monotonic() - started < 2.0


@pytest.mark.parametrize("path, body", [
    ("/gpu/jobs/window", ["stack:1"]),
    ("/gpu/jobs/window", {"model": 7, "jobs": 1}),
    ("/gpu/jobs", {"model": 7, "workflow": {"a": {}}}),
    ("/gpu/jobs", {"model": ["stack:1"], "workflow": {"a": {}}}),
])
async def test_a_malformed_body_is_400_not_500(rig, path, body):
    _fake, build = rig
    arbiter, stack, _store, sch = build()
    app = make_app(Reg(stack, spec("llm:27b")), arbiter, scheduler=sch)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(path, json=body)
        assert resp.status == 400


# ---- failures ---------------------------------------------------------------


def test_failed_flips_back_off_exponentially_to_a_cap(monkeypatch):
    monkeypatch.setattr(sched_mod, "_FLIP_RETRY", 30.0)   # the shipped base
    assert [sched_mod.retry_after(n) for n in range(1, 8)] == [
        30.0, 60.0, 120.0, 240.0, 480.0, 900.0, 900.0]


async def test_a_stack_that_keeps_failing_fails_the_head_job(rig, monkeypatch):
    _fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)
    monkeypatch.setattr(sched_mod, "_FLIP_RETRY", 0.01)
    arbiter, _stack, store, sch = build()
    arbiter.fail_next_start = True               # sticks: every start fails
    head = store.submit(STACK, [{"a": {}}, {"b": {}}])
    second = store.submit(STACK, [{"c": {}}])
    sch.start()
    await until(lambda: head.terminal)
    assert arbiter.events.count("start-failed:stack:1") == 3
    assert {w.error for w in head.workflows} == {"stack failed to start 3 times"}
    assert second.state == PENDING, "only the head job is failed"
    gap = sch._not_before - time.monotonic()
    assert 0.03 < gap <= 0.04 + 0.01, "the third failure backs off 4x the base"

    arbiter.fail_next_start = False
    await until(lambda: second.terminal)
    assert sch._failures == 0, "a successful flip resets the backoff"


async def test_an_exception_in_a_workflow_charges_an_attempt(rig, monkeypatch):
    _fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)

    def broken(entry, tag):
        raise RuntimeError("cannot read these outputs")

    monkeypatch.setattr(sched_mod, "outputs_of", broken)
    _arbiter, _stack, store, sch = build()
    job = store.submit(STACK, [{"a": {}}])
    sch.start()
    await until(lambda: job.terminal)
    wf = job.workflows[0]
    assert wf.state == FAILED and wf.attempts == 2
    assert "cannot read these outputs" in wf.error


# ---- a bounded drain ----------------------------------------------------------


async def test_a_job_submitted_mid_render_waits_for_the_next_flip(rig, monkeypatch, caplog):
    fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)
    _arbiter, _stack, store, sch = build()
    fake.gate = asyncio.Event()
    first = store.submit(STACK, [{"a": {}}])
    sch.start()
    await until(lambda: fake.prompts)
    late = store.submit(STACK, [{"b": {}}])
    with caplog.at_level("INFO", logger="vramux.scheduler"):
        fake.gate.set()
        await until(lambda: late.terminal)
    assert first.state == DONE and late.state == DONE
    assert [h["dir"] for h in store.handovers].count("to_render") == 2, \
        "the late job ran under a flip of its own"
    assert "1 workflow(s) submitted during it wait for the next flip" in caplog.text


async def test_a_count_window_closes_when_its_human_walks_away(rig, monkeypatch):
    _fake, build = rig
    monkeypatch.setattr(sched_mod, "WINDOW_IDLE", 0.2)
    arbiter, stack, _store, sch = build()
    sch.start()
    sch.open_window(stack, jobs=5, minutes=None)
    await until(lambda: sch.mode == RENDER)
    await until(lambda: sch.mode == LLM, timeout=2.0)
    assert sch.window is None


async def test_a_drain_that_runs_nothing_backs_off(rig, monkeypatch):
    _fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)
    monkeypatch.setattr(sched_mod, "_FLIP_RETRY", 5.0)
    _arbiter, _stack, store, sch = build()
    store.submit(STACK, [{"a": {}}])
    monkeypatch.setattr(store, "next_workflow", lambda *a, **k: None)
    sch.start()
    await asyncio.sleep(0.3)
    flips = [h["dir"] for h in store.handovers].count("to_render")
    assert flips == 1, f"{flips} flips in 0.3s: a hot loop"


async def test_a_count_window_counts_only_the_humans_prompts(rig):
    fake, build = rig
    arbiter, stack, store, sch = build()
    await llm_resident(arbiter, hold=True)       # only the window flips
    sch.start()
    sch.open_window(stack, jobs=2, minutes=None)
    await until(lambda: sch.mode == RENDER and sch.window["baseline"] is not None)
    ours = store.submit(STACK, [{"a": {}}, {"b": {}}])
    sch.wake()
    await until(lambda: ours.terminal)
    fake.complete("human-1")
    await asyncio.sleep(0.1)
    assert sch.mode == RENDER, "vramux's own prompts are not the human's"
    fake.history.clear()                          # the stack's history was cleared
    await asyncio.sleep(0.1)
    fake.complete("human-2")
    await until(lambda: sch.mode == LLM)


# ---- a late answer from /prompt -------------------------------------------------


async def test_a_prompt_whose_answer_timed_out_is_found_not_resubmitted(rig, monkeypatch):
    fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)
    monkeypatch.setattr(sched_mod, "_PROMPT_TIMEOUT", 0.1)
    fake.slow_post = 0.5
    fake.render_s = 0.05
    _arbiter, _stack, store, sch = build()
    job = store.submit(STACK, [{"a": {}}])
    sch.start()
    await until(lambda: job.terminal)
    assert job.state == DONE
    assert fake.prompts == [{"a": {}}], "the stack had it; it was not sent twice"
    assert job.workflows[0].prompt_id == "p0" and job.workflows[0].attempts == 1


async def test_a_restored_workflow_with_no_prompt_id_is_looked_for_first(rig, tmp_path, monkeypatch):
    fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)
    store = JobStore(tmp_path / "jobs.json")
    job = store.submit(STACK, [{"a": {}}])
    wf = job.workflows[0]
    store.mark_running(job, wf)
    wf.attempts = 1                    # crashed after the POST, before the answer
    fake.extra["p-early"] = {"vramux": {"job": job.id, "workflow": 0}}
    fake.complete("p-early")
    _arbiter, _stack, _st, sch = build(store=store)
    sch.start()
    await until(lambda: job.terminal)
    assert job.state == DONE and wf.prompt_id == "p-early"
    assert fake.prompts == []


# ---- what a flip-back owes the language models -----------------------------------


async def test_a_request_parked_at_flip_back_is_admitted_before_the_next_flip(rig):
    fake, build = rig
    arbiter, _stack, store, sch = build(stack=stack_spec(render_start_deadline=0))
    order = []
    real_hold = arbiter.hold_render

    def hold(tag):
        order.append("hold")
        real_hold(tag)

    arbiter.hold_render = hold
    fake.gate = asyncio.Event()
    store.submit(STACK, [{"a": {}}])
    sch.start()
    await until(lambda: fake.prompts)

    async def chat():
        await arbiter.acquire(spec("llm:27b"))
        order.append("chat admitted")
        arbiter.release("llm:27b")

    parked = asyncio.create_task(chat())
    await until(lambda: arbiter.parked)
    late = store.submit(STACK, [{"b": {}}])        # deferred to the next flip
    fake.gate.set()
    await asyncio.wait_for(parked, 3)
    await until(lambda: late.terminal)
    assert order[:3] == ["hold", "chat admitted", "hold"], order


async def test_a_refusal_to_make_room_backs_off_but_fails_no_job(rig, monkeypatch):
    _fake, build = rig
    monkeypatch.setattr(sched_mod, "QUIET_GRACE", 0.0)
    monkeypatch.setattr(sched_mod, "_FLIP_RETRY", 0.01)
    arbiter, _stack, store, sch = build()

    async def outranked(spec):
        raise residency.InsufficientVRAM("lease L1 (batch, priority 9) outranks serving")

    monkeypatch.setattr(arbiter, "_make_room_for", outranked)
    job = store.submit(STACK, [{"a": {}}])
    sch.start()
    await until(lambda: sch._failures >= 4)
    assert job.state == PENDING, "the stack never failed to start; nothing is failed"
    assert sch._failed_starts == 0

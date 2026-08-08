"""The client side: talking to a running broker from a shell.

This is the whole adoption story. A consumer migrates only if the correct
version is shorter than the hack it replaces, so the wrapper has to turn a
poll-unload-hope loop into one line:

```bash
vramux lease --mb 18000 --owner batch-pipeline -- ./stage2.sh
```

Acquire, run, renew in the background, release on the way out — including on a
crash, and including the wrapper itself being killed. Release-on-exit is the
fast path and nothing more: a wrapper killed with `SIGKILL` runs no cleanup by
definition, which is why the broker's TTL sweep is what makes the guarantee
real. This side is an optimisation on top of that.

Deliberately stdlib-only. A client that needs a virtualenv to release a lease
is a client that will not be installed on the machine that needs it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

# Renew this many times per TTL. Three gives two chances to miss before a lease
# a live holder still wants expires under it.
_RENEWALS_PER_TTL = 3


def _url(args, path: str) -> str:
    return f"http://{args.host}:{args.port}{path}"


def _request(url: str, method: str = "GET", body: Optional[dict] = None,
             timeout: float = 10.0) -> Tuple[int, dict]:
    """One HTTP call. Returns (status, decoded body); never raises for HTTP
    status, because the status *is* the answer for most of this API."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read() or b"{}"
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"error": raw.decode(errors="replace")}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return 0, {"error": f"cannot reach vramux at {url}: {exc}"}


def _fail(payload: dict, prefix: str = "") -> int:
    print(f"{prefix}{payload.get('error', 'unknown error')}", file=sys.stderr)
    return 1


_YIELD_SIGNALS = {
    "term": signal.SIGTERM,
    "int": signal.SIGINT,
    "hup": signal.SIGHUP,
}


class _Renewer:
    """Heartbeat for a held lease, on a background thread.

    A thread rather than a task: the wrapper's foreground job is waiting on a
    child process, and mixing that with an event loop buys nothing here.

    The heartbeat is also how a yield request arrives. That is the whole
    transport: no callback URL for the broker to reach back on, no second
    connection, nothing to open a port for — the holder is already talking
    three times per TTL, and a holder that has stopped talking is about to
    expire anyway.
    """

    def __init__(self, url: str, ttl: float, on_yield=None) -> None:
        self.url = url
        self.interval = max(1.0, ttl / _RENEWALS_PER_TTL)
        self._on_yield = on_yield
        self._yielded = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            status, payload = _request(self.url, method="POST", body={})
            if status == 200:
                self._check_yield(payload)
                continue
            # Losing the lease mid-run is worth saying out loud: the memory is
            # no longer reserved, and the run is now racing anybody who asks.
            print(
                f"vramux: lease renewal failed ({status}): "
                f"{payload.get('error', 'no answer')}",
                file=sys.stderr,
            )
            if status in (404, 0):
                return

    def _check_yield(self, payload: dict) -> None:
        """Say it once, and act only if the caller asked to act.

        Defaulting to a signal would be wrong: this wrapper does not know what
        the command it is running is in the middle of, and killing a job that
        is nine minutes into a ten-minute stage to free memory for a chat
        request is worse than the contention it solves. The operator decides
        with `--on-yield`, and the default tells a human instead.
        """
        request = payload.get("yield")
        if not request or self._yielded:
            return
        self._yielded = True
        print(
            f"vramux: {request.get('by', 'something')} is waiting for "
            f"{request.get('wanted_mb', '?')} MiB of this lease "
            f"(by {request.get('deadline', 'soon')}) — release it if you can",
            file=sys.stderr,
        )
        if self._on_yield is not None:
            self._on_yield(request)


def lease(args) -> int:
    """Hold a lease for the lifetime of a command."""
    # argparse keeps the `--` separator with REMAINDER often enough to matter,
    # and running a command called `--` is not a thing anybody wants.
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        print("vramux lease: nothing to run — put the command after `--`", file=sys.stderr)
        return 2
    status, payload = _request(
        _url(args, "/gpu/lease"),
        method="POST",
        body={
            "mb": args.mb,
            "owner": args.owner,
            "ttl": args.ttl,
            "priority": args.priority,
            "wait": args.wait,
            # Our own pid, because the command runs inside our process tree and
            # the broker attributes by ancestry. That is what keeps the grant
            # from being charged a second time once the child allocates.
            "pid": os.getpid(),
        },
        timeout=args.wait + 30,
    )
    if status != 200:
        return _fail(payload, "vramux lease: ")

    lease_id = payload["lease"]
    print(
        f"vramux: {lease_id} — {payload['granted_mb']} MiB for {args.owner}, "
        f"expires {payload['expires_at']}",
        file=sys.stderr,
    )
    child: Optional[subprocess.Popen] = None

    def forward(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    def on_yield(_request) -> None:
        # Opt-in, and it signals the command rather than releasing the lease:
        # releasing while the child still holds the memory would hand the card
        # to somebody else on the strength of a promise nobody kept.
        sig = _YIELD_SIGNALS.get(getattr(args, "on_yield", "warn"))
        if sig is None or child is None or child.poll() is not None:
            return
        print(f"vramux: forwarding {sig.name} to the command", file=sys.stderr)
        child.send_signal(sig)

    renewer = _Renewer(_url(args, f"/gpu/lease/{lease_id}/renew"), args.ttl,
                       on_yield=on_yield)
    renewer.start()

    previous = {
        sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        child = subprocess.Popen(args.command)
        return child.wait()
    finally:
        renewer.stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        status, payload = _request(_url(args, f"/gpu/lease/{lease_id}"), method="DELETE")
        if status not in (200, 404):
            print(f"vramux: release failed ({status}) — the lease will expire on its "
                  f"own in {args.ttl:.0f}s", file=sys.stderr)


def free(args) -> int:
    """Block until `--mb` is available to grant, then exit.

    Deliberately not a lease: this reports a fact about the card and reserves
    nothing. A caller that needs the memory kept for it wants `vramux lease`.
    """
    deadline = time.monotonic() + args.wait
    reported = False
    while True:
        status, payload = _request(_url(args, "/gpu/state"))
        if status != 200:
            return _fail(payload, "vramux free: ")
        budget = payload.get("budget")
        if not budget:
            print("vramux free: this router is not accounting for the card", file=sys.stderr)
            return 1
        if budget["free_mb"] >= args.mb:
            print(budget["free_mb"])
            return 0
        if args.mb > budget["ceiling_mb"]:
            print(f"vramux free: {args.mb} MiB exceeds the {budget['ceiling_mb']} MiB "
                  f"this card can ever provide", file=sys.stderr)
            return 1
        if time.monotonic() >= deadline:
            print(f"vramux free: {budget['free_mb']} MiB free, {args.mb} MiB wanted",
                  file=sys.stderr)
            return 1
        if not reported:
            reported = True
            print(f"vramux: waiting for {args.mb} MiB ({budget['free_mb']} MiB free)",
                  file=sys.stderr)
        time.sleep(min(2.0, max(0.1, deadline - time.monotonic())))


def evict(args) -> int:
    status, payload = _request(
        _url(args, "/gpu/evict"), method="POST", body={"tag": args.tag}, timeout=180,
    )
    if status != 200:
        return _fail(payload, "vramux evict: ")
    print(f"evicted {args.tag}")
    return 0


def leases(args) -> int:
    status, payload = _request(_url(args, "/gpu/lease"))
    if status != 200:
        return _fail(payload, "vramux leases: ")
    rows = payload.get("leases", [])
    if not rows:
        print("no leases held")
        return 0
    # HELD is what the holder actually has on the card right now. A grant with
    # HELD far below MiB is a holder that has not allocated yet — which is the
    # correct order to do it in, so it is normal early and suspicious late.
    print(f"  {'MiB':>7}  {'HELD':>7}  {'PRI':>3}  {'EXPIRES':<26} OWNER")
    for row in sorted(rows, key=lambda r: -r["granted_mb"]):
        print(f"  {row['granted_mb']:>7}  {row.get('observed_mb', 0):>7}  "
              f"{row['priority']:>3}  "
              f"{row['expires_at']:<26} {row['owner']}  ({row['lease']})")
        # A holder that has been asked for its memory is the only thing here
        # worth a second line: it is the state where somebody else is waiting.
        asked = row.get("yield")
        if asked:
            print(f"      ↳ asked to yield {asked.get('wanted_mb', '?')} MiB for "
                  f"{asked.get('by', '?')} by {asked.get('deadline', '?')}")
    return 0

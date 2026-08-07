"""`vramux top` — the live view of the card.

The thing that was wanted at the start, and deliberately the last thing built:
a console is only worth having once every number under it is real, and until
Stage 6 half of them were "the one resident" and the other half were a grant
nobody could see the holder of.

Two halves, kept apart on purpose:

* **`render()` is pure.** State in, lines out. Every layout decision — what
  fits, what truncates, what turns red — is testable without a terminal, a
  card, or a running router, which is the same rule the rest of this suite
  runs under.
* **`Feed` is the transport.** It streams `/gpu/events` and falls back to
  polling `/gpu/state` when the router is older than this file or the stream
  breaks. A console that goes blank because an endpoint moved is worse than
  one that quietly polls.

Stdlib only, for the same reason `cli.py` is: a view of the card that needs a
virtualenv is a view nobody opens from the machine that is on fire.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

# How long a dropped stream waits before reconnecting. Short enough that
# restarting the service looks like a blink, long enough that a router that is
# down does not become a reconnect loop hammering a dead socket.
RECONNECT_DELAY = 2.0

# Cadence for the polling fallback. Matches the feed's own default interval,
# so the fallback view is not visibly slower than the streamed one.
POLL_INTERVAL = 1.0


@dataclass
class Line:
    """One rendered row and what it means, so the terminal can colour it
    without the renderer knowing what a terminal is."""

    text: str
    style: str = ""  # "", "head", "warn", "hot", "dim"


# ---- reading the router ---------------------------------------------------


class Feed:
    """The newest state the router has told us about.

    A thread rather than an event loop: the console's foreground job is a
    blocking `getch()`, and an asyncio console would be an asyncio console
    wrapped around one blocking read.
    """

    def __init__(self, host: str, port: int) -> None:
        self.base = f"http://{host}:{port}"
        self._latest: Optional[dict] = None
        self._error: Optional[str] = None
        self._streaming = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True)

    # ---- what the drawing side sees ---------------------------------------

    @property
    def state(self) -> Optional[dict]:
        with self._lock:
            return self._latest

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    @property
    def streaming(self) -> bool:
        with self._lock:
            return self._streaming

    def wait(self, timeout: float) -> None:
        """Block until something new arrived, or `timeout` passes."""
        try:
            self._wake.get(timeout=timeout)
        except queue.Empty:
            pass

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ---- the reading thread ------------------------------------------------

    def _publish(self, payload: Optional[dict], error: Optional[str],
                 streaming: Optional[bool] = None) -> None:
        with self._lock:
            if payload is not None:
                self._latest = payload
            self._error = error
            if streaming is not None:
                self._streaming = streaming
        try:
            self._wake.put_nowait(True)
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._stream()
            except Exception as exc:
                self._publish(None, f"{exc}", streaming=False)
            if self._stop.is_set():
                break
            # The stream ended. Either the router restarted or it predates
            # `/gpu/events`; poll once so the view keeps moving either way.
            self.poll_once()
            self._stop.wait(RECONNECT_DELAY)

    def _stream(self) -> None:
        req = urllib.request.Request(
            self.base + "/gpu/events", headers={"Accept": "text/event-stream"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            self._publish(None, None, streaming=True)
            for payload in _iter_sse(resp, self._stop):
                self._publish(payload, payload.get("error"), streaming=True)
                if self._stop.is_set():
                    return
        self._publish(None, None, streaming=False)

    def poll_once(self) -> None:
        try:
            with urllib.request.urlopen(self.base + "/gpu/state", timeout=5) as resp:
                self._publish(json.load(resp), None, streaming=False)
        except urllib.error.HTTPError as exc:
            self._publish(None, f"router answered {exc.code}", streaming=False)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self._publish(None, f"cannot reach vramux at {self.base}: {exc}",
                          streaming=False)


def _iter_sse(stream, stop: Optional[threading.Event] = None):
    """Yield the JSON payload of each `data:` line.

    Comment lines (`: keepalive`) and the blank separators are skipped, which
    is the whole of the SSE framing this endpoint uses.
    """
    for raw in stream:
        if stop is not None and stop.is_set():
            return
        line = raw.decode("utf-8", "replace").strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            continue
        if not line.startswith("data:"):
            continue
        try:
            yield json.loads(line[5:].strip())
        except ValueError:
            continue


# ---- rendering ------------------------------------------------------------


def _mb(value) -> str:
    try:
        return f"{int(value):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def _ago(iso: Optional[str], now: Optional[float] = None) -> str:
    """"12s", "4m", "2h" — how long since an absolute timestamp.

    The router sends absolute times precisely so this arithmetic happens here:
    an age computed server-side would change every second and turn the event
    stream into a per-second broadcast of nothing.
    """
    if not iso:
        return "—"
    try:
        when = datetime.fromisoformat(iso)
    except ValueError:
        return "—"
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    seconds = (now if now is not None else time.time()) - when.timestamp()
    return _duration(seconds)


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"


def _bar(used: int, total: int, reserve: int, width: int) -> str:
    """Usage as a bar, with the reserve drawn as the part that is not for you.

    `:` rather than `#` for the reserve because it is not free memory and it is
    not somebody's allocation either — drawing it as either one makes the
    arithmetic in the line above it look wrong.
    """
    width = max(10, width)
    if total <= 0:
        return "[" + " " * width + "]"
    filled = min(width, round(width * used / total))
    held = min(width - filled, round(width * reserve / total))
    return "[" + "#" * filled + ":" * held + " " * (width - filled - held) + "]"


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def render(state: Optional[dict], width: int = 80, height: int = 24,
           now: Optional[float] = None, note: str = "") -> List[Line]:
    """The whole view, as lines. No terminal involved.

    `height` bounds the process list rather than the whole render: residents,
    leases and the budget are the point, and a card with forty processes on it
    must not push them off the top.
    """
    width = max(40, width)
    lines: List[Line] = []
    if not state:
        # Before the first frame, and after a router that went away. The note
        # is transport status ("streaming", or why not) and never stands in for
        # the message: "polling" alone on a blank screen reads as a card with
        # nothing on it.
        lines.append(Line("vramux top", "head"))
        lines.append(Line(""))
        lines.append(Line("waiting for the router…", "warn"))
        if note:
            lines.append(Line(_clip(f" {note}", width), "dim"))
        return lines
    if state.get("error"):
        lines.append(Line("vramux top", "head"))
        lines.append(Line(""))
        lines.append(Line(str(state["error"]), "hot"))
        return lines

    device = state.get("device") or {}
    budget = state.get("budget") or {}
    total = int(device.get("total_mb") or 0)
    used = int(device.get("used_mb") or 0)
    reserve = int(budget.get("reserve_mb") or 0)
    free = int(budget.get("free_mb", device.get("free_mb") or 0))

    header = f"vramux  {device.get('name', 'gpu')}  (gpu{device.get('index', 0)})"
    lines.append(Line(_clip(f"{header}{note and '   ' + note}", width), "head"))

    pct = (used * 100 // total) if total else 0
    lines.append(Line(_bar(used, total, reserve, width - 2)))
    grantable = f"grantable {_mb(free)}" if budget else f"free {_mb(device.get('free_mb'))}"
    lines.append(Line(_clip(
        f" {_mb(used)} / {_mb(total)} MiB used ({pct}%)   {grantable}"
        f"   reserve {_mb(reserve)}   foreign {_mb(state.get('foreign_mb'))}",
        width,
    ), "hot" if free <= reserve else ""))

    loading = state.get("loading")
    if loading:
        lines.append(Line(_clip(
            f" loading {loading.get('tag', '?')} — {loading.get('elapsed_s', 0)}s"
            f" of {loading.get('budget_s', 0)}s", width), "warn"))

    # ---- residents
    lines.append(Line(""))
    residents = state.get("resident_detail")
    if residents is None:
        # An older router: tags are all it reports. Say what is known rather
        # than drawing an empty table over a card with models on it.
        residents = [{"tag": tag} for tag in state.get("residents") or []]
    lines.append(Line(_clip(
        f" {'RESIDENT':<28} {'COST':>8} {'PORT':>6} {'REQ':>4} {'IDLE':>6}",
        width), "head"))
    if not residents:
        lines.append(Line(" nothing resident", "dim"))
    for row in residents:
        lines.append(Line(_clip(
            f" {row.get('tag', '?'):<28} {_mb(row.get('cost_mb')):>8}"
            f" {row.get('port') or '—':>6} {row.get('inflight', 0):>4}"
            f" {_ago(row.get('last_use'), now):>6}", width)))

    # ---- leases
    lines.append(Line(""))
    leases = state.get("leases") or []
    lines.append(Line(_clip(
        f" {'LEASE OWNER':<28} {'GRANT':>8} {'HELD':>8} {'OUT':>7} {'TTL':>5}",
        width), "head"))
    if not leases:
        lines.append(Line(" no leases held", "dim"))
    for row in sorted(leases, key=lambda r: -int(r.get("granted_mb") or 0)):
        outstanding = int(row.get("outstanding_mb") or 0)
        lines.append(Line(_clip(
            f" {row.get('owner', '?'):<28} {_mb(row.get('granted_mb')):>8}"
            f" {_mb(row.get('observed_mb')):>8} {_mb(outstanding):>7}"
            f" {_expires_in(row.get('expires_at'), now):>5}", width),
            # A grant nobody has allocated against is normal right after it is
            # taken and worth looking at an hour later. The console cannot tell
            # the two apart, so it marks the fact, not a verdict.
            "warn" if outstanding else ""))

    # ---- everything else on the card
    #
    # "Everything else" means exactly that: not a resident's backend, and not a
    # leaseholder either. A holder's memory is already on screen as HELD, and
    # showing the same 2 386 MiB again under FOREIGN reads as two allocations.
    leased_pids = {pid for row in leases for pid in (row.get("pids") or [])}
    processes = [
        p for p in (state.get("processes") or [])
        if not p.get("owner") and p.get("pid") not in leased_pids
    ]
    if processes:
        lines.append(Line(""))
        lines.append(Line(_clip(f" {'FOREIGN':<28} {'MiB':>8}   PID", width), "head"))
        room = max(3, height - len(lines) - 2)
        for proc in sorted(processes, key=lambda p: -int(p.get("used_mb") or 0))[:room]:
            name = (proc.get("name") or "?").split()[0].split("/")[-1]
            lines.append(Line(_clip(
                f" {name:<28} {_mb(proc.get('used_mb')):>8}   {proc.get('pid')}", width),
                "dim"))
        hidden = len(processes) - room
        if hidden > 0:
            lines.append(Line(f" …and {hidden} more", "dim"))

    unlocated = state.get("unlocated_owners") or []
    if unlocated:
        # Something vramux started is holding memory NVML will not attribute to
        # it. Worth a line: it is the case where the budget is right and the
        # attribution is not.
        lines.append(Line(""))
        lines.append(Line(_clip(
            f" unlocated: {', '.join(unlocated)}", width), "warn"))
    return lines


def _expires_in(iso: Optional[str], now: Optional[float] = None) -> str:
    if not iso:
        return "—"
    try:
        when = datetime.fromisoformat(iso)
    except ValueError:
        return "—"
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return _duration(when.timestamp() - (now if now is not None else time.time()))


# ---- the terminal --------------------------------------------------------


_STYLES = {"head": 1, "warn": 2, "hot": 3, "dim": 4}


def _draw(screen, lines: List[Line], colours: bool) -> None:
    import curses

    screen.erase()
    height, width = screen.getmaxyx()
    for row, line in enumerate(lines[: height - 1]):
        attr = 0
        if colours and line.style in _STYLES:
            attr = curses.color_pair(_STYLES[line.style])
            if line.style == "head":
                attr |= curses.A_BOLD
        try:
            screen.addnstr(row, 0, line.text, width - 1, attr)
        except Exception:
            # A terminal that refuses a write at the last cell is not a reason
            # to take the console down.
            pass
    screen.noutrefresh()
    curses.doupdate()


def top(args) -> int:
    """Entry point for `vramux top`."""
    feed = Feed(args.host, args.port)
    if getattr(args, "once", False):
        # One frame to stdout: what a pipe, a log or a test wants, and the
        # reason none of the rendering above needs a terminal to be exercised.
        feed.poll_once()
        state = feed.state
        for line in render(state, width=args.width, height=args.height,
                           note=feed.error or ""):
            print(line.text.rstrip())
        return 0 if state else 1

    import curses

    feed.start()

    def loop(screen) -> int:
        curses.curs_set(0)
        screen.nodelay(True)
        colours = False
        try:
            curses.start_color()
            curses.use_default_colors()
            for pair, colour in ((1, curses.COLOR_CYAN), (2, curses.COLOR_YELLOW),
                                 (3, curses.COLOR_RED), (4, curses.COLOR_BLUE)):
                curses.init_pair(pair, colour, -1)
            colours = True
        except Exception:
            pass  # a terminal without colour still draws every number
        while True:
            height, width = screen.getmaxyx()
            note = "streaming" if feed.streaming else "polling"
            if feed.error:
                note = feed.error
            _draw(screen, render(feed.state, width=width, height=height, note=note),
                  colours)
            # Wake on a new frame, but never sit longer than a second: the ages
            # and TTLs on screen are computed here and have to keep counting on
            # a card where nothing is happening.
            feed.wait(min(1.0, POLL_INTERVAL))
            key = screen.getch()
            if key in (ord("q"), ord("Q"), 27):
                return 0

    try:
        return curses.wrapper(loop)
    except KeyboardInterrupt:
        return 0
    finally:
        feed.stop()

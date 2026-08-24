"""Entry point.

`python -m vramux` runs the ollama-compatible router — no subcommand, because
that is what the service unit invokes and what everything on the machine
expects. Every other subcommand is a *client*: it talks to a running router
over HTTP and exits. `state` is the exception that also works without one, and
says so when it does.
"""

import argparse
import asyncio
import logging
import sys

from aiohttp import web

from . import cli, env
from .budget import DEFAULT_RESERVE_MB
from .lease import (
    DEFAULT_TTL,
    PRIORITY_DEFAULT,
    SAMPLE_INTERVAL,
    Broker,
    clamped_sample_interval,
)
from .observer import CostCache, Observer
from .registry import ModelRegistry
from .router import make_app
from .residency import (
    DEFAULT_MAX_RESIDENTS,
    DEFAULT_QUEUE_WAIT,
    DEFAULT_SERVING_PRIORITY,
    DEFAULT_YIELD_WAIT,
    ResidencyArbiter,
)


def _serve(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    registry = ModelRegistry()
    observer = Observer(device_index=args.device)
    arbiter = ResidencyArbiter(
        port=args.upstream_port,
        idle_timeout=args.idle_timeout,
        observer=observer,
        max_residents=args.max_residents,
        yield_wait=args.yield_wait,
        serving_priority=args.serving_priority,
        queue_wait=args.queue_wait,
    )
    # `loading` rather than the arbiter itself: the broker needs to know a load
    # is in flight — it has not allocated yet, so the card reads freer than it
    # is about to be — and needs nothing else from residency.
    broker = Broker(
        observer,
        reserve_mb=args.reserve_mb,
        loading=lambda: arbiter.loading,
        sample_interval=clamped_sample_interval(args.sample_interval),
    )
    # The other half of the pair: residency decides room from the same budget
    # the broker grants leases from. Injected after construction because each
    # side needs one callable from the other and neither should import the
    # other's module.
    arbiter.use_budget(broker.budget)
    # The other half of tier 3: residency can stop what it started, and asking
    # a leaseholder to give memory back is the only move it has left.
    arbiter.use_yield(broker.request_yield)
    # And tier 1 read from the lease's side: a lease that outranks a resident
    # may have it evicted — its own children are the one thing vramux can stop.
    broker.use_make_room(arbiter.make_room_for_lease)
    # The last pairing: a refused load's one remaining question — is the
    # memory held by leases that outrank me, worth parking behind — is the
    # broker's to answer.
    arbiter.use_outrankers(broker.outrankers)
    app = make_app(registry, arbiter, broker)
    web.run_app(app, host=args.host, port=args.port, print=None)


def _remote_state(host: str, port: int) -> dict:
    """Ask a running router, which is the only process that knows which PIDs
    it started. Returns {} when there is nothing listening."""
    import json
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/gpu/state"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return json.load(resp)
    except (urllib.error.URLError, OSError, ValueError):
        return {}


def _state(args: argparse.Namespace) -> int:
    """Print the true picture of the card, including what vramux never started."""
    from .nvml import DeviceState, GpuProcess
    from .observer import Attribution, Snapshot

    remote = _remote_state(args.host, args.port)
    if remote.get("device"):
        d = remote["device"]
        snap = Snapshot(
            device=DeviceState(
                index=d["index"], name=d["name"], total_mb=d["total_mb"],
                used_mb=d["used_mb"], free_mb=d["free_mb"],
                processes=[GpuProcess(pid=p["pid"], used_mb=p["used_mb"], name=p["name"])
                           for p in remote.get("processes", [])],
            ),
            attributions=[
                Attribution(
                    process=GpuProcess(pid=p["pid"], used_mb=p["used_mb"], name=p["name"]),
                    owner=p.get("owner"),
                )
                for p in remote.get("processes", [])
            ],
            unlocated_owners=remote.get("unlocated_owners", []),
        )
        entries = remote.get("costs", {})
    else:
        snap = asyncio.run(Observer(device_index=args.device).snapshot())
        if snap is None:
            print("no GPU visible (nvidia-smi missing or the device cannot be read)",
                  file=sys.stderr)
            return 1
        # Nothing is running to tell us what it owns, so every process below is
        # reported as foreign. Say so rather than implying the card is idle.
        print("(no router running — ownership unknown, everything reads as foreign)")
        entries = CostCache().all()

    print(snap.render())

    if entries:
        print()
        print("measured costs")
        print(f"  {'MiB':>7}  {'CTX':>7}  {'N':>3}  MODEL")
        for entry in sorted(entries.values(), key=lambda e: -int(e.get("measured_mb") or 0)):
            print(f"  {entry.get('measured_mb', 0):>7}  {entry.get('ctx', 0):>7}"
                  f"  {entry.get('samples', 0):>3}  {entry.get('tag', '?')}")
    else:
        print()
        print("no measured costs recorded yet")
    return 0


def _top(args: argparse.Namespace) -> int:
    """Imported here rather than at module scope: `console` pulls in `curses`,
    and the router — which is what this module exists to start — has no
    business failing on a box without a terminal library."""
    from . import console

    return console.top(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vramux", description="vramux router (ollama-API compatible)"
    )
    parser.add_argument("--device", type=int, default=env.get_int("DEVICE", 0),
                        help="GPU index to observe")
    parser.add_argument("--host", default=env.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=env.get_int("PORT", 11434))
    parser.add_argument("--upstream-port", type=int, default=env.get_int("UPSTREAM_PORT", 18080))
    parser.add_argument("--idle-timeout", type=float, default=env.get_float("IDLE_TIMEOUT", 900))
    parser.add_argument("--log-level", default=env.get("LOG_LEVEL", "INFO"))
    parser.add_argument("--reserve-mb", type=int,
                        default=env.get_int("RESERVE_MB", DEFAULT_RESERVE_MB),
                        help="headroom held back from every grant")
    parser.add_argument("--max-residents", type=int,
                        default=env.get_int("MAX_RESIDENTS", DEFAULT_MAX_RESIDENTS),
                        help="most models allowed resident at once")
    parser.add_argument("--yield-wait", type=float,
                        default=env.get_float("YIELD_WAIT", DEFAULT_YIELD_WAIT),
                        help="seconds admission waits for a leaseholder to yield "
                             "before refusing the load; 0 disables asking")
    parser.add_argument("--serving-priority", type=int,
                        default=env.get_int("SERVING_PRIORITY", DEFAULT_SERVING_PRIORITY),
                        help="what serving outranks; higher wins, leases default to 5")
    parser.add_argument("--queue-wait", type=float,
                        default=env.get_float("QUEUE_WAIT", DEFAULT_QUEUE_WAIT),
                        help="seconds an outranked load request parks behind a "
                             "higher-priority lease before refusing; 0 fails fast")
    parser.add_argument("--sample-interval", type=float,
                        default=env.get_float("SAMPLE_INTERVAL", SAMPLE_INTERVAL),
                        help="seconds between usage-history samples")
    sub = parser.add_subparsers(dest="subcommand")
    sub.add_parser("serve", help="run the router (the default)")
    sub.add_parser("state", help="print what is on the GPU and exit")

    p_lease = sub.add_parser(
        "lease", help="hold VRAM for the lifetime of a command",
        description="vramux lease --mb 18000 --owner batch -- ./stage2.sh",
    )
    p_lease.add_argument("--mb", type=int, required=True, help="megabytes to reserve")
    p_lease.add_argument("--owner", required=True, help="who is holding it, for the log")
    p_lease.add_argument("--ttl", type=float, default=DEFAULT_TTL,
                         help="seconds before the lease expires unrenewed")
    p_lease.add_argument("--wait", type=float, default=0.0,
                         help="seconds to wait for room before giving up")
    p_lease.add_argument("--priority", type=int, default=PRIORITY_DEFAULT,
                         help="higher wins; a holder below the asker's priority "
                              "is the one asked to yield")
    p_lease.add_argument("--on-yield", choices=("warn", "term", "int", "hup"),
                         default="warn",
                         help="what to do when something asks for this memory: "
                              "warn (default), or forward that signal to the command")
    p_lease.add_argument("command", nargs=argparse.REMAINDER,
                         help="the command to run, after `--`")

    p_free = sub.add_parser("free", help="wait until N MiB could be granted")
    p_free.add_argument("--mb", type=int, required=True)
    p_free.add_argument("--wait", type=float, default=0.0)

    p_evict = sub.add_parser("evict", help="unload a resident model by name")
    p_evict.add_argument("tag")

    sub.add_parser("leases", help="list the leases currently held")

    p_top = sub.add_parser("top", help="live view of the card")
    p_top.add_argument("--once", action="store_true",
                       help="print one frame and exit — for pipes and logs")
    p_top.add_argument("--width", type=int, default=100,
                       help="columns to render at, with --once")
    p_top.add_argument("--height", type=int, default=40,
                       help="rows to render at, with --once")

    args = parser.parse_args()

    clients = {
        "state": _state,
        "lease": cli.lease,
        "free": cli.free,
        "evict": cli.evict,
        "leases": cli.leases,
        "top": _top,
    }
    handler = clients.get(args.subcommand)
    if handler is not None:
        raise SystemExit(handler(args))
    _serve(args)


if __name__ == "__main__":
    main()

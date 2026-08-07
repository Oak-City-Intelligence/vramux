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
from .lease import DEFAULT_TTL, Broker
from .observer import CostCache, Observer
from .registry import ModelRegistry
from .router import make_app
from .residency import ResidencyArbiter


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
    )
    # `loading` rather than the arbiter itself: the broker needs to know a load
    # is in flight — it has not allocated yet, so the card reads freer than it
    # is about to be — and needs nothing else from residency.
    broker = Broker(
        observer,
        reserve_mb=args.reserve_mb,
        loading=lambda: arbiter.loading,
    )
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
    p_lease.add_argument("--priority", type=int, default=5)
    p_lease.add_argument("command", nargs=argparse.REMAINDER,
                         help="the command to run, after `--`")

    p_free = sub.add_parser("free", help="wait until N MiB could be granted")
    p_free.add_argument("--mb", type=int, required=True)
    p_free.add_argument("--wait", type=float, default=0.0)

    p_evict = sub.add_parser("evict", help="unload a resident model by name")
    p_evict.add_argument("tag")

    sub.add_parser("leases", help="list the leases currently held")

    args = parser.parse_args()

    clients = {
        "state": _state,
        "lease": cli.lease,
        "free": cli.free,
        "evict": cli.evict,
        "leases": cli.leases,
    }
    handler = clients.get(args.subcommand)
    if handler is not None:
        raise SystemExit(handler(args))
    _serve(args)


if __name__ == "__main__":
    main()

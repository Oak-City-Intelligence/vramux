"""Entry point: `python -m vramux` runs the ollama-compatible router."""

import argparse
import logging

from aiohttp import web

from . import env
from .registry import ModelRegistry
from .router import make_app
from .supervisor import LlamaServerSupervisor


def main() -> None:
    parser = argparse.ArgumentParser(description="vramux router (ollama-API compatible)")
    parser.add_argument("--host", default=env.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=env.get_int("PORT", 11434))
    parser.add_argument("--upstream-port", type=int, default=env.get_int("UPSTREAM_PORT", 18080))
    parser.add_argument("--idle-timeout", type=float, default=env.get_float("IDLE_TIMEOUT", 900))
    parser.add_argument("--log-level", default=env.get("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    registry = ModelRegistry()
    supervisor = LlamaServerSupervisor(port=args.upstream_port, idle_timeout=args.idle_timeout)
    app = make_app(registry, supervisor)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()

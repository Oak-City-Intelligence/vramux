"""Entry point: `python -m llamacpp` runs the ollama-compatible router."""

import argparse
import logging
import os

from aiohttp import web

from .registry import ModelRegistry
from .router import make_app
from .supervisor import LlamaServerSupervisor


def main() -> None:
    parser = argparse.ArgumentParser(description="vramux llama.cpp router (ollama-API compatible)")
    parser.add_argument("--host", default=os.environ.get("MYLLAMA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MYLLAMA_PORT", "11434")))
    parser.add_argument("--upstream-port", type=int, default=int(os.environ.get("MYLLAMA_UPSTREAM_PORT", "18080")))
    parser.add_argument("--idle-timeout", type=float, default=float(os.environ.get("MYLLAMA_IDLE_TIMEOUT", "900")))
    parser.add_argument("--log-level", default=os.environ.get("MYLLAMA_LOG_LEVEL", "INFO"))
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

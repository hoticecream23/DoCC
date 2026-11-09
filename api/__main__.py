"""Run the API: python -m api --config config/pipeline.yaml --host 127.0.0.1 --port 8000."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from api.app import create_app
from api.jobs import DEFAULT_WORKERS
from baseline.logging_setup import setup_logging

DEFAULT_PORT = 8000


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and serve until Ctrl+C.

    Args:
        argv: Arguments; None for sys.argv.
    """
    ap = argparse.ArgumentParser(prog="python -m api", description="DoCC HTTP API")
    ap.add_argument("--config", type=Path, default=None, help="config directory or a file in it")
    ap.add_argument("--host", default="127.0.0.1", help="bind address; loopback by default")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="cascade jobs run at once")
    args = ap.parse_args(argv)
    config_dir = args.config.parent if args.config and args.config.is_file() else args.config

    setup_logging()
    uvicorn.run(create_app(config_dir, workers=args.workers), host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()

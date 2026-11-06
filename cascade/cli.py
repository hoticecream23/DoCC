"""Command line entry point.

  cascade run --input DIR --output results.jsonl [--model NAME] [--base-url URL]

Writes results.jsonl, results.jsonl.provenance and results.jsonl.bboxes/.
Score with `baseline eval` and gate with `baseline compare`, as for any run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baseline.config import load_config_cached
from baseline.logging_setup import get_logger, setup_logging
from baseline.pipeline import Models

from .config import load_cascade_config
from .runner import run_batch
from .vlm import VLMClient

log = get_logger(__name__)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config_cached(args.config)
    ccfg = load_cascade_config(args.config)
    overrides = {k: v for k, v in (("model", args.model), ("base_url", args.base_url)) if v}
    settings = ccfg.vlm.model_copy(update=overrides)
    backend = VLMClient(settings)  # refuses a missing model or a non local host
    log.info("cascade server", extra={"base_url": settings.base_url, "model": settings.model})
    stats = run_batch(
        args.input,
        args.output,
        cfg,
        ccfg,
        backend,
        models=Models.load(cfg, Path(args.root)),
        bbox_dir=args.bbox_dir,
        resume=not args.no_resume,
    )
    log.info("run complete", extra=stats)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade", description="Cascade v2: VLM extraction with grounding")
    p.add_argument("--config", default=None, help="config directory, defaults to ./config")
    p.add_argument("--root", default=".", help="root for relative model paths")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="process a directory into JSONL")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--model", default=None, help="overrides vlm.model in cascade.yaml")
    r.add_argument("--base-url", default=None, help="overrides vlm.base_url in cascade.yaml")
    r.add_argument("--bbox-dir", default=None, help="defaults to <output>.bboxes")
    r.add_argument("--no-resume", action="store_true", help="reprocess everything")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except Exception:
        log.exception("command failed", extra={"command": args.command})
        return 1


if __name__ == "__main__":
    sys.exit(main())

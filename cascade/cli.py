"""Command line entry point.

  cascade run --input DIR --output results.jsonl [--model NAME] [--base-url URL]
              [--no-resume] [--dry-run] [--cabinet ID]

Writes results.jsonl, results.jsonl.provenance and results.jsonl.bboxes/.
Score with `baseline eval` and gate with `baseline compare`, as for any run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baseline.config import load_config_cached
from baseline.io import report_dry_run
from baseline.logging_setup import get_logger, setup_logging
from baseline.pipeline import Models
from baseline.scan import scan_documents

from .config import load_cascade_config
from .runner import model_root, run_batch
from .vlm import VLMClient

log = get_logger(__name__)

# The shell convention for a process stopped by SIGINT: 128 + 2.
EXIT_INTERRUPTED = 130


def cmd_run(args: argparse.Namespace) -> int:
    """Run the cascade over a directory, or report what a run would write.

    Args:
        args: Parsed `run` arguments.

    Returns:
        The process exit code, 0 on success.

    Raises:
        ValueError: The VLM model is not set or the host is not local.
    """
    if args.dry_run:
        out = Path(args.output)
        paths, _ = scan_documents(args.input)
        outputs = [out, f"{out}.provenance", args.bbox_dir or f"{out}.bboxes"]
        return report_dry_run(outputs, len(paths))
    cfg = load_config_cached(args.config)
    ccfg = load_cascade_config(args.config)
    overrides = {k: v for k, v in (("model", args.model), ("base_url", args.base_url)) if v}
    settings = ccfg.vlm.model_copy(update=overrides)
    backend = VLMClient(settings)  # refuses a missing model or a non local host
    log.info("cascade server", extra={"base_url": settings.base_url, "model": settings.model})
    root = model_root(cfg, args.root)
    stats = run_batch(
        args.input,
        args.output,
        cfg,
        ccfg,
        backend,
        models=Models.load(cfg, root),
        bbox_dir=args.bbox_dir,
        resume=args.resume,
        root=root,
        cabinet_id=args.cabinet,
    )
    log.info("run complete", extra=stats)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the `cascade` argument parser.

    Returns:
        The parser with its `run` sub command.
    """
    p = argparse.ArgumentParser(prog="cascade", description="Cascade v2: VLM extraction with grounding")
    p.add_argument("--config", default=None, help="config directory, defaults to ./config")
    p.add_argument("--root", default=None, help="root for relative model paths; defaults to the config's parent")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="process a directory into JSONL")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--model", default=None, help="overrides vlm.model in cascade.yaml")
    r.add_argument("--base-url", default=None, help="overrides vlm.base_url in cascade.yaml")
    r.add_argument("--bbox-dir", default=None, help="defaults to <output>.bboxes")
    r.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True, help="skip documents already written"
    )
    r.add_argument("--dry-run", action="store_true", help="log what would be written and stop")
    r.add_argument("--cabinet", default=None, help="cabinet_id for every record, overrides the config")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the command and turn failures into exit codes.

    Args:
        argv: Arguments without the program name; None reads sys.argv.

    Returns:
        0 on success, 1 on failure, EXIT_INTERRUPTED on Ctrl+C.
    """
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        log.warning("interrupted")
        return EXIT_INTERRUPTED
    except Exception:  # noqa: BLE001 - CLI top level: log any failure and exit 1, not a traceback
        log.exception("command failed", extra={"command": args.command})
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Command line entry points.

  baseline run             --input DIR --output results.jsonl [--workers N]
  baseline train           --annotations FILE --output-model DIR
  baseline tune-thresholds --annotations FILE --predictions FILE
  baseline profile         --input DIR --output profile.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import classify as classify_mod
from . import tagging as tagging_mod
from .config import load_config_cached, save_yaml
from .extract import extract
from .logging_setup import get_logger, setup_logging
from .pipeline import run_batch
from .profile import run_profile
from .schema import build_canonical_text

log = get_logger(__name__)


def _read_annotations(path: str | Path) -> list[dict]:
    """JSONL of {path|text, label?, tags?}. Sorted so training is reproducible."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} is not valid JSON: {exc}") from exc
    rows.sort(key=lambda r: str(r.get("doc_id") or r.get("path") or r.get("text", ""))[:200])
    return rows


def _text_for(row: dict, cfg) -> str:
    """Annotation may carry text inline, otherwise we extract it."""
    if row.get("text"):
        return str(row["text"])
    p = row.get("path")
    if not p:
        raise ValueError("annotation row needs either text or path")
    res = extract(p, cfg)
    return build_canonical_text(res.pages)


# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    stats = run_batch(
        input_dir=args.input,
        output=args.output,
        config_dir=args.config,
        workers=args.workers,
        bbox_dir=args.bbox_dir,
        resume=not args.no_resume,
        root=args.root,
    )
    log.info("run complete", extra=stats)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config_cached(args.config)
    rows = _read_annotations(args.annotations)
    if not rows:
        log.error("no annotations found", extra={"file": args.annotations})
        return 1

    out = Path(args.output_model)
    texts = [_text_for(r, cfg) for r in rows]
    report: dict = {}

    labelled = [(t, r["label"]) for t, r in zip(texts, rows) if r.get("label")]
    if labelled and args.kind in ("classifier", "both"):
        report["classifier"] = classify_mod.train(
            [t for t, _ in labelled], [l for _, l in labelled], out / "classifier"
        )
    elif args.kind in ("classifier", "both"):
        log.warning("no label field in annotations, skipping classifier")

    tagged = [(t, r["tags"]) for t, r in zip(texts, rows) if r.get("tags")]
    if tagged and args.kind in ("tagger", "both"):
        report["tagger"] = tagging_mod.train(
            [t for t, _ in tagged], [list(g) for _, g in tagged], out / "tagger"
        )
    elif args.kind in ("tagger", "both"):
        log.warning("no tags field in annotations, skipping tagger")

    if not report:
        log.error("nothing was trained")
        return 1
    log.info("training complete", extra={"models": sorted(report)})
    return 0


def cmd_tune_thresholds(args: argparse.Namespace) -> int:
    """Re-score predictions with the tagger, fit F1 optimal cutoffs, save them."""
    cfg = load_config_cached(args.config)
    model_dir = Path(args.root) / str(cfg.tagging_opts().get("model_dir", "models/tagger"))
    model = tagging_mod.load_model(args.model_dir or model_dir)
    if model is None:
        log.error("no tagger model, train one first", extra={"dir": str(model_dir)})
        return 1

    truth_rows = _read_annotations(args.annotations)
    truth: dict[str, list[str]] = {}
    for r in truth_rows:
        key = r.get("doc_id") or r.get("path")
        if key and r.get("tags") is not None:
            truth[str(key)] = list(r["tags"])
    if not truth:
        log.error("annotations carry no tags")
        return 1

    # Predictions file supplies the canonical text, keyed by doc_id or path.
    texts: dict[str, str] = {}
    with open(args.predictions, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            body = rec.get("text", {}).get("full", "")
            for key in (rec.get("doc_id"), rec.get("source_path")):
                if key:
                    texts[str(key)] = body

    # Annotations that carry their own text can be tuned without a join.
    inline = {
        str(r.get("doc_id") or r.get("path")): r["text"]
        for r in truth_rows
        if r.get("text") and (r.get("doc_id") or r.get("path"))
    }

    y_true: dict[str, list[int]] = {t: [] for t in model.tags}
    y_prob: dict[str, list[float]] = {t: [] for t in model.tags}
    matched = 0
    for key, tags in sorted(truth.items()):
        body = texts.get(key) or inline.get(key)
        if body is None:
            continue
        matched += 1
        scores = model.predict_scores(body)
        have = set(tags)
        for t in model.tags:
            y_true[t].append(1 if t in have else 0)
            y_prob[t].append(scores.get(t, 0.0))

    if not matched:
        log.error(
            "no annotation matched a prediction and none carried inline text, "
            "check that doc_id or path keys line up"
        )
        return 1

    tuned = tagging_mod.tune_thresholds(y_true, y_prob)
    log.info("thresholds fitted", extra={"matched": matched, "tags": len(tuned)})

    for tag, info in sorted(tuned.items()):
        log.info(
            "tag threshold",
            extra={"tag": tag, "threshold": info["threshold"], "f1": info["f1"],
                   "positives": info["positives"]},
        )

    if args.dry_run:
        log.info("dry run, tags.yaml not modified")
        return 0

    # Write the tuned values straight back into the config.
    data = dict(cfg.tags)
    for entry in data.get("tags", []):
        info = tuned.get(entry.get("name"))
        if info:
            entry["threshold"] = info["threshold"]
    save_yaml(Path(cfg.config_dir) / "tags.yaml", data)
    log.info("tags.yaml updated", extra={"path": str(Path(cfg.config_dir) / "tags.yaml")})
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    stats = run_profile(args.input, args.output, args.config)
    log.info("profile complete", extra={"documents": stats["n_documents"]})
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baseline", description="Baseline document pipeline")
    p.add_argument("--config", default=None, help="config directory, defaults to ./config")
    p.add_argument("--root", default=".", help="root for relative model paths")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="process a directory into JSONL")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--bbox-dir", default=None, help="defaults to <output>.bboxes")
    r.add_argument("--no-resume", action="store_true", help="reprocess everything")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("train", help="fit and persist the classifier and tagger")
    t.add_argument("--annotations", required=True)
    t.add_argument("--output-model", required=True)
    t.add_argument("--kind", choices=["classifier", "tagger", "both"], default="both")
    t.set_defaults(func=cmd_train)

    th = sub.add_parser("tune-thresholds", help="fit per tag thresholds and save to config")
    th.add_argument("--annotations", required=True)
    th.add_argument("--predictions", required=True)
    th.add_argument("--model-dir", default=None)
    th.add_argument("--dry-run", action="store_true")
    th.set_defaults(func=cmd_tune_thresholds)

    pr = sub.add_parser("profile", help="characterise a corpus, no models involved")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", default="profile.md")
    pr.set_defaults(func=cmd_profile)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except Exception as exc:
        log.exception("command failed", extra={"command": args.command})
        return 1


if __name__ == "__main__":
    sys.exit(main())

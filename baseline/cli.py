"""Command line entry points.

  baseline run             --input DIR --output results.jsonl [--workers N]
  baseline train           --annotations FILE --output-model DIR
  baseline tune-thresholds --annotations FILE --predictions FILE
  baseline profile         --input DIR --output profile.md
  baseline import-labels   --csv FILE --root DIR --output gold.jsonl
  baseline eval            --gold FILE --pred FILE [--output report.md]
  baseline compare         --gold FILE --a FILE --b FILE [--output cmp.md]
  baseline graph           --input results.jsonl --output graph.json
  baseline tables          --input results.jsonl --output tables.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import classify as classify_mod
from . import tagging as tagging_mod
from .config import load_config_cached, save_yaml
from .evaluate import compare, evaluate, render_comparison, render_report
from .extract import extract
from .graph import build as build_graph
from .labels import import_labels, render_summary
from .logging_setup import get_logger, setup_logging
from .pipeline import run_batch
from .profile import run_profile
from .schema import build_canonical_text
from .tables import build as build_tables

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


def cmd_import_labels(args: argparse.Namespace) -> int:
    cfg = load_config_cached(args.config)
    res = import_labels(args.csv, args.root, cfg, args.name_column, args.label_column)
    if res.unmapped_labels:
        return 1
    if not res.rows:
        log.error("no rows resolved, check --root points at the labelled set")
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for row in res.rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    log.info("gold written", extra={"output": str(out), "rows": len(res.rows)})

    summary = render_summary(res, str(args.csv), str(args.root))
    if args.report:
        Path(args.report).write_text(summary, encoding="utf-8")
    else:
        print(summary)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = load_config_cached(args.config)
    unknown = str(cfg.classify_opts().get("unknown_label", "unknown"))
    res = evaluate(args.gold, args.pred, unknown, args.pred_tables)
    if not res["matched"]:
        log.error("nothing matched, check that doc_id or path keys line up")
        return 1

    report = render_report(res, str(args.gold), str(args.pred))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        log.info("report written", extra={"output": str(out)})
    if args.json:
        Path(args.json).write_text(
            json.dumps(res, indent=2, sort_keys=True), encoding="utf-8"
        )

    log.info(
        "scores",
        extra={
            "matched": res["matched"],
            "classification_macro_f1": res["classification"]["macro_f1"],
            "classification_coverage": res["classification"]["coverage"],
            "metadata_micro_f1": res["metadata"]["value"]["micro"]["f1"],
            "tagging_micro_f1": res["tagging"]["micro"]["f1"],
            "validated_precision": res["metadata"]["validated_precision"],
        },
    )
    tb = res.get("tables") or {}
    if tb.get("scored"):
        log.info(
            "table scores",
            extra={"gold_tables": tb["gold_tables"], "pred_tables": tb["pred_tables"],
                   "detection_recall": tb["detection_recall"],
                   "cell_f1": tb["cells"]["f1"], "cell_precision": tb["cells"]["precision"],
                   "adjacency_f1": tb["adjacency"]["f1"],
                   "adjacency_precision": tb["adjacency"]["precision"]},
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    cfg = load_config_cached(args.config)
    unknown = str(cfg.classify_opts().get("unknown_label", "unknown"))
    cmp = compare(args.gold, args.a, args.b, unknown, args.a_tables, args.b_tables)
    report = render_comparison(cmp, str(args.a), str(args.b))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        log.info("comparison written", extra={"output": str(out)})
    for k, v in cmp["headline"].items():
        log.info("delta", extra={"metric": k, "a": v["a"], "b": v["b"], "delta": v["delta"]})
    if cmp["regressions"]:
        log.warning("regressions found", extra={"count": len(cmp["regressions"])})
        for r in cmp["regressions"]:
            log.warning("regression", extra={"detail": r})
    # Non zero exit on regression so this can gate CI.
    return 2 if (cmp["regressions"] and args.fail_on_regression) else 0


def cmd_graph(args: argparse.Namespace) -> int:
    cfg = load_config_cached(args.config)
    if not cfg.graph:
        log.error("no graph.yaml in the config directory")
        return 1
    res = build_graph(
        args.input, args.output, cfg.graph,
        dot_path=args.dot, report_path=args.report,
    )
    st = res["stats"]
    log.info(
        "graph summary",
        extra={"documents": st["documents"], "organizations": st["organizations"],
               "accounts": st["accounts"], "edges": st["edges"],
               "chains": len(res["chains"]),
               "duplicate_groups": len(st["duplicate_groups"]),
               "orphans": len(res["orphans"])},
    )
    return 0


def cmd_tables(args: argparse.Namespace) -> int:
    cfg = load_config_cached(args.config)
    if not cfg.tables:
        log.error("no tables.yaml in the config directory")
        return 1
    res = build_tables(
        args.input, args.output, cfg.tables,
        bbox_dir=args.bbox_dir, report_path=args.report,
        deterministic_timings=bool(
            cfg.pipeline.get("runtime", {}).get("deterministic_timings", False)
        ),
    )
    st = res["stats"]
    log.info(
        "tables summary",
        extra={"documents": st["documents"],
               "documents_with_tables": st["documents_with_tables"],
               "tables": st["tables"], "cells": st["cells"],
               "placed_cells": st["placed_cells"],
               "abstentions": st["abstentions"], "errors": st["errors"],
               "by_method": st["by_method"]},
    )
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

    il = sub.add_parser("import-labels", help="convert a client label CSV into a gold file")
    il.add_argument("--csv", required=True, help="CSV with a file name and a label column")
    il.add_argument("--root", required=True, help="directory the CSV's files live under")
    il.add_argument("--output", required=True, help="gold JSONL to write")
    il.add_argument("--report", help="write the import summary here instead of stdout")
    il.add_argument("--name-column", default="File_Name")
    il.add_argument("--label-column", default="Document_Type")
    il.set_defaults(func=cmd_import_labels)

    ev = sub.add_parser("eval", help="score predictions against gold")
    ev.add_argument("--gold", required=True)
    ev.add_argument("--pred", required=True)
    ev.add_argument("--output", default="report.md")
    ev.add_argument("--pred-tables", default=None,
                    help="tables JSONL, scored when gold carries tables")
    ev.add_argument("--json", default=None, help="also dump raw scores as JSON")
    ev.set_defaults(func=cmd_eval)

    cp = sub.add_parser("compare", help="score two prediction files and diff them")
    cp.add_argument("--gold", required=True)
    cp.add_argument("--a", required=True, help="baseline run")
    cp.add_argument("--b", required=True, help="the challenger")
    cp.add_argument("--a-tables", default=None, help="tables JSONL for the baseline")
    cp.add_argument("--b-tables", default=None, help="tables JSONL for the challenger")
    cp.add_argument("--output", default="comparison.md")
    cp.add_argument("--fail-on-regression", action="store_true",
                    help="exit 2 if anything got worse, for CI")
    cp.set_defaults(func=cmd_compare)

    gr = sub.add_parser("graph", help="build a knowledge graph over a results file")
    gr.add_argument("--input", required=True, help="results JSONL")
    gr.add_argument("--output", default="graph.json")
    gr.add_argument("--dot", default=None, help="also write graphviz source")
    gr.add_argument("--report", default=None, help="also write a markdown summary")
    gr.set_defaults(func=cmd_graph)

    tb = sub.add_parser("tables", help="extract tables over a results file")
    tb.add_argument("--input", required=True, help="results JSONL")
    tb.add_argument("--output", default="tables.jsonl")
    tb.add_argument("--bbox-dir", default=None, help="defaults to <input>.bboxes")
    tb.add_argument("--report", default=None, help="also write a markdown summary")
    tb.set_defaults(func=cmd_tables)

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

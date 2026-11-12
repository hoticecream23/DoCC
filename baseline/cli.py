"""Command line entry points.

Over 400 lines: every CLI subparser and handler lives here so the whole command surface is in one place.

baseline run             --input DIR --output results.jsonl [--workers N] [--no-resume] [--cabinet ID]
                         [--client-pack DIR --cabinet NAME]
baseline train           --annotations FILE --output-model DIR
baseline tune-thresholds --annotations FILE --predictions FILE
baseline profile         --input DIR --output profile.md
baseline import-labels   --csv FILE --root DIR --output gold.jsonl
baseline export-workbook --input results.jsonl --output workbook.xlsx
baseline import-workbook --xlsx FILE --root DIR --output gold.jsonl
baseline eval            --gold FILE --pred FILE [--output report.md]
baseline compare         --gold FILE --a FILE --b FILE [--output cmp.md]
baseline graph           --input results.jsonl --output graph.json
baseline tables          --input results.jsonl --output tables.jsonl
baseline db-import       results.jsonl [--db PATH] [--tables FILE] [--provenance FILE]
baseline db-export       out.jsonl [--db PATH] [--cabinet ID]
baseline client-pack build       inventory.csv schema.csv --out DIR [--all-types]
baseline client-pack materialize --pack DIR --cabinet NAME --share ROOT --work DIR
baseline client-eval     --pack DIR --cabinet NAME --results FILE [--out report.md]
baseline client-run      --pack DIR --cabinet NAME --share ROOT --work DIR [--no-cascade]

Every command that writes takes --dry-run: it logs the paths and document count, then stops.

This module only parses arguments, calls the module that owns each command,
and logs the outcome. Command logic lives in pipeline, training, labels,
workbook, evaluate, graph, tables, profile, db and clientrun.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from . import db
from . import tagging as tagging_mod  # noqa: F401 - re-exported: tests patch cli.tagging_mod.load_model
from .clientpack import PACK_FILE, load_pack, materialize
from .clientpack_build import build_pack
from .clientrun import client_run
from .config import Config, load_config_cached
from .evaluate import compare, evaluate, render_comparison, render_report
from .evaluate.client_eval import render_client_report, score_cabinet
from .graph import build as build_graph
from .io import atomic_open, atomic_write_text, report_dry_run
from .labels import import_labels, render_summary
from .logging_setup import get_logger, setup_logging
from .pipeline import run_batch
from .profile import run_profile
from .scan import scan_documents
from .schema import read_jsonl
from .tables import build as build_tables
from .training import fit_tag_thresholds, read_annotations, save_tag_thresholds, train_models
from .workbook import export_workbook, import_workbook
from .workbook import render_summary as render_workbook_summary

log = get_logger(__name__)

EXIT_INTERRUPTED = 130  # shell convention: 128 + SIGINT
EXIT_REGRESSION = 2  # compare --fail-on-regression found something worse
# These log their own dry run: the generic one cannot tell what they would read or write.
OWN_DRY_RUN = {"tune-thresholds", "client-pack", "client-run"}


def _cfg(args: argparse.Namespace) -> Config:
    """The Config main() loaded; tests that call cmd_* directly may not set it."""
    cfg: Config | None = getattr(args, "cfg", None)
    return cfg or load_config_cached(args.config)


def _write_gold(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write gold rows as sorted-key JSONL, all or nothing; return the row count."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with atomic_open(out, newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    log.info("gold written", extra={"output": str(out), "rows": count})
    return count


def _emit(path: str | None, text: str) -> None:
    """Write text to path, or print it when no path was given."""
    if path:
        atomic_write_text(path, text)
    else:
        print(text)


# --------------------------------------------------------------------------
# command handlers: each returns the process exit code
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    """Process a directory into results JSONL."""
    client_pack = (args.client_pack, args.cabinet) if args.client_pack else None
    stats = run_batch(
        input_dir=args.input,
        output=args.output,
        config_dir=args.config,
        workers=args.workers,
        bbox_dir=args.bbox_dir,
        resume=args.resume,
        root=args.root,
        cabinet_id=None if client_pack else args.cabinet,
        client_pack=client_pack,
    )
    log.info("run complete", extra=stats)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Fit and save the classifier and/or tagger from annotations."""
    rows = read_annotations(args.annotations)
    if not rows:
        log.error("no annotations found", extra={"file": args.annotations})
        return 1
    report = train_models(rows, _cfg(args), Path(args.output_model), args.kind)
    if not report:
        log.error("nothing was trained")
        return 1
    log.info("training complete", extra={"models": sorted(report)})
    return 0


def cmd_tune_thresholds(args: argparse.Namespace) -> int:
    """Re-score predictions with the tagger, fit F1 optimal cutoffs, save them."""
    cfg = _cfg(args)
    model_dir = Path(args.root) / str(cfg.tagging_opts().get("model_dir", "models/tagger"))
    tuned = fit_tag_thresholds(args.model_dir or model_dir, args.annotations, args.predictions)
    if tuned is None:
        return 1
    if args.dry_run:
        log.info("dry run, tags.yaml not modified")
        return 0
    path = save_tag_thresholds(cfg, tuned)
    log.info("tags.yaml updated", extra={"path": str(path)})
    return 0


def cmd_import_labels(args: argparse.Namespace) -> int:
    """Convert a client label CSV into a gold file."""
    res = import_labels(args.csv, args.root, _cfg(args), args.name_column, args.label_column)
    if res.unmapped_labels:
        return 1
    if not res.rows:
        log.error("no rows resolved, check --root points at the labelled set")
        return 1
    _write_gold(args.output, res.rows)
    _emit(args.report, render_summary(res, str(args.csv), str(args.root)))
    return 0


def cmd_export_workbook(args: argparse.Namespace) -> int:
    """Write a results file out as the review workbook."""
    cfg = _cfg(args)
    records = read_jsonl(args.input)
    tables = read_jsonl(args.tables) if args.tables else None
    try:
        counts = export_workbook(
            records,
            cfg,
            args.output,
            tables=tables,
            run_id=args.run_id or "",
            overwrite=args.overwrite,
            merge_from=args.merge_from,
        )
    except FileExistsError as exc:
        log.error("refusing to overwrite", extra={"error": str(exc)})
        return 1
    log.info("workbook written", extra={"output": str(args.output), **counts})
    print(f"Wrote {args.output}")
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n}")
    if args.merge_from:
        print(
            f"Carried the review from {args.merge_from}. A confirmation of a "
            "prediction that has since changed was cleared for re-review; every "
            "correction was kept."
        )
    else:
        print(
            "Review columns are blank on purpose. A row becomes gold only once " "IS_GOLD and a verdict are filled in."
        )
    return 0


def cmd_import_workbook(args: argparse.Namespace) -> int:
    """Convert the reviewed index workbook into a gold file."""
    res = import_workbook(args.xlsx, args.root, _cfg(args))
    # A name the config does not declare is never guessed at, so the report is
    # still written and the exit code says the file is not fit to score with.
    _emit(args.report, render_workbook_summary(res, str(args.xlsx), str(args.root)))
    if res.problems:
        return 1
    if not res.rows:
        log.error("no gold rows produced, check the workbook has reviewed rows")
        return 1
    _write_gold(args.output, res.rows)
    _record_review(args, res.events)
    return 0


def _record_review(args: argparse.Namespace, events: list[db.ReviewEvent]) -> None:
    """Store the workbook's corrections as review events when a store exists for it."""
    path = _db_path(args, args.xlsx)
    if path is None or not path.exists():
        return
    with closing(db.connect(path)) as conn:
        written = db.record_review_events(conn, events)
    log.info("review events stored", extra={"db": str(path), "events": len(events), "written": written})


def _log_scores(res: dict[str, Any]) -> None:
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
            extra={
                "gold_tables": tb["gold_tables"],
                "pred_tables": tb["pred_tables"],
                "detection_recall": tb["detection_recall"],
                "cell_f1": tb["cells"]["f1"],
                "cell_precision": tb["cells"]["precision"],
                "adjacency_f1": tb["adjacency"]["f1"],
                "adjacency_precision": tb["adjacency"]["precision"],
            },
        )


def _write_report(path: str | None, text: str, message: str) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, text)
    log.info(message, extra={"output": str(out)})


def cmd_eval(args: argparse.Namespace) -> int:
    """Score predictions against gold."""
    unknown = str(_cfg(args).classify_opts().get("unknown_label", "unknown"))
    res = evaluate(args.gold, args.pred, unknown, args.pred_tables)
    if not res["matched"]:
        log.error("nothing matched, check that doc_id or path keys line up")
        return 1
    _write_report(args.output, render_report(res, str(args.gold), str(args.pred)), "report written")
    if args.json:
        atomic_write_text(args.json, json.dumps(res, indent=2, sort_keys=True))
    _log_scores(res)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Score two prediction files and diff them; exit 2 on regression when asked."""
    unknown = str(_cfg(args).classify_opts().get("unknown_label", "unknown"))
    cmp = compare(args.gold, args.a, args.b, unknown, args.a_tables, args.b_tables)
    _write_report(args.output, render_comparison(cmp, str(args.a), str(args.b)), "comparison written")
    for metric, values in cmp["headline"].items():
        log.info("delta", extra={"metric": metric, "a": values["a"], "b": values["b"], "delta": values["delta"]})
    if cmp["regressions"]:
        log.warning("regressions found", extra={"count": len(cmp["regressions"])})
        for regression in cmp["regressions"]:
            log.warning("regression", extra={"detail": regression})
    # Non zero exit on regression so this can gate CI.
    return EXIT_REGRESSION if (cmp["regressions"] and args.fail_on_regression) else 0


def cmd_graph(args: argparse.Namespace) -> int:
    """Build a knowledge graph over a results file."""
    cfg = _cfg(args)
    if not cfg.graph:
        log.error("no graph.yaml in the config directory")
        return 1
    res = build_graph(args.input, args.output, cfg.graph, dot_path=args.dot, report_path=args.report)
    stats = res["stats"]
    log.info(
        "graph summary",
        extra={
            "documents": stats["documents"],
            "organizations": stats["organizations"],
            "accounts": stats["accounts"],
            "edges": stats["edges"],
            "chains": len(res["chains"]),
            "duplicate_groups": len(stats["duplicate_groups"]),
            "orphans": len(res["orphans"]),
        },
    )
    return 0


def cmd_tables(args: argparse.Namespace) -> int:
    """Extract tables over a results file."""
    cfg = _cfg(args)
    if not cfg.tables:
        log.error("no tables.yaml in the config directory")
        return 1
    res = build_tables(
        args.input,
        args.output,
        cfg.tables,
        bbox_dir=args.bbox_dir,
        report_path=args.report,
        deterministic_timings=bool(cfg.pipeline.get("runtime", {}).get("deterministic_timings", False)),
    )
    keys = ("documents", "documents_with_tables", "tables", "cells", "placed_cells", "abstentions", "errors")
    log.info("tables summary", extra={**{k: res["stats"][k] for k in keys}, "by_method": res["stats"]["by_method"]})
    return 0


def _db_path(args: argparse.Namespace, beside: str) -> Path | None:
    """--db when given, else db.path from the config resolved beside the given file."""
    return Path(args.db) if args.db else _cfg(args).db_path(beside)


def cmd_db_import(args: argparse.Namespace) -> int:
    """Load a results file, and optional sidecars, into the SQLite store."""
    path = _db_path(args, args.input)
    if path is None:
        log.error("db.path is null in the config and no --db was given")
        return 1
    with closing(db.connect(path)) as conn:
        stats = db.import_jsonl(conn, args.input, provenance_path=args.provenance, tables_path=args.tables)
    log.info("db import complete", extra={"db": str(path), **stats})
    return 0


def cmd_db_export(args: argparse.Namespace) -> int:
    """Write the SQLite store back out as results JSONL."""
    path = _db_path(args, args.output)
    if path is None or not path.exists():
        log.error("no database to export", extra={"db": str(path)})
        return 1
    with closing(db.connect(path)) as conn:
        n = db.export_jsonl(conn, args.output, cabinet_id=args.cabinet)
    log.info("db export complete", extra={"db": str(path), "documents": n})
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Characterise a corpus, no models involved."""
    stats = run_profile(args.input, args.output, args.config)
    log.info("profile complete", extra={"documents": stats["n_documents"]})
    return 0


def cmd_client_pack_build(args: argparse.Namespace) -> int:
    """Build or refresh a client pack from a share inventory."""
    if args.dry_run:
        with open(args.inventory, encoding="utf-8-sig") as fh:
            rows = max(sum(1 for _ in fh) - 1, 0)
        return report_dry_run([Path(args.out) / PACK_FILE, Path(args.out)], rows)
    summary = build_pack(args.inventory, args.schema, args.out, all_types=args.all_types, share_root=args.share or "")
    log.info("pack written", extra={"out": summary.out_dir, "cabinets": len(summary.cabinets)})
    return 0


def cmd_client_pack_materialize(args: argparse.Namespace) -> int:
    """Copy one cabinet's documents out of the share into a work directory."""
    if args.dry_run:
        pack = load_pack(args.pack, args.cabinet)
        return report_dry_run([Path(args.work) / pack.cabinet.slug], len(pack.manifest))
    paths = materialize(args.pack, args.cabinet, args.share, args.work)
    log.info("documents ready", extra={"documents": len(paths)})
    return 0


def cmd_client_eval(args: argparse.Namespace) -> int:
    """Score a run over one cabinet against its client pack."""
    score = score_cabinet(args.pack, args.cabinet, args.results)
    _emit(args.output, render_client_report(score, Path(args.results).name))
    log.info("client scores", extra={"accuracy": round(score.accuracy, 4), "scored": score.scored})
    return 0


def cmd_client_run(args: argparse.Namespace) -> int:
    """Materialize, run and score one cabinet of a client pack."""
    client_run(
        args.pack,
        args.cabinet,
        args.share,
        args.work,
        config_dir=args.config,
        root=args.root,
        cascade=args.cascade,
        dry_run=args.dry_run,
    )
    return 0


# --------------------------------------------------------------------------
# argument parsing: one _add_<command>_parser per subcommand
# --------------------------------------------------------------------------


def _add_run_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    r = sub.add_parser("run", help="process a directory into JSONL")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--bbox-dir", default=None, help="defaults to <output>.bboxes")
    r.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True, help="skip documents already written"
    )
    r.add_argument("--cabinet", default=None, help="cabinet_id for every record; with --client-pack, the pack cabinet")
    r.add_argument("--client-pack", default=None, help="client pack directory: classify and extract by its cabinet")
    r.set_defaults(func=cmd_run)


def _add_train_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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


def _add_profile_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    pr = sub.add_parser("profile", help="characterise a corpus, no models involved")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", default="profile.md")
    pr.set_defaults(func=cmd_profile)


def _add_label_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    il = sub.add_parser("import-labels", help="convert a client label CSV into a gold file")
    il.add_argument("--csv", required=True, help="CSV with a file name and a label column")
    il.add_argument("--root", required=True, help="directory the CSV's files live under")
    il.add_argument("--output", required=True, help="gold JSONL to write")
    il.add_argument("--report", help="write the import summary here instead of stdout")
    il.add_argument("--name-column", default="File_Name")
    il.add_argument("--label-column", default="Document_Type")
    il.set_defaults(func=cmd_import_labels)


def _add_workbook_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ew = sub.add_parser("export-workbook", help="write a results file out as the review workbook")
    ew.add_argument("--input", required=True, help="results JSONL")
    ew.add_argument("--output", required=True, help="workbook to write")
    ew.add_argument("--tables", help="tables JSONL, fills the Line items tab")
    ew.add_argument("--run-id", help="defaults to a timestamp")
    ew.add_argument(
        "--merge-from",
        help="an earlier reviewed workbook, whose review is carried " "onto this run wherever it still applies",
    )
    ew.add_argument("--overwrite", action="store_true", help="replace an existing file, discarding any review in it")
    ew.set_defaults(func=cmd_export_workbook)

    iw = sub.add_parser("import-workbook", help="convert the reviewed index workbook into a gold file")
    iw.add_argument("--xlsx", required=True, help="the reviewed workbook")
    iw.add_argument(
        "--root",
        required=True,
        help="directory the reviewed documents live under, used to " "resolve each doc_id back to a file",
    )
    iw.add_argument("--output", required=True, help="gold JSONL to write")
    iw.add_argument("--report", help="write the import summary here instead of stdout")
    iw.add_argument("--db", default=None, help="store to record corrections in; defaults to db.path beside the xlsx")
    iw.set_defaults(func=cmd_import_workbook)


def _add_eval_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ev = sub.add_parser("eval", help="score predictions against gold")
    ev.add_argument("--gold", required=True)
    ev.add_argument("--pred", required=True)
    ev.add_argument("--output", default="report.md")
    ev.add_argument("--pred-tables", default=None, help="tables JSONL, scored when gold carries tables")
    ev.add_argument("--json", default=None, help="also dump raw scores as JSON")
    ev.set_defaults(func=cmd_eval)

    cp = sub.add_parser("compare", help="score two prediction files and diff them")
    cp.add_argument("--gold", required=True)
    cp.add_argument("--a", required=True, help="baseline run")
    cp.add_argument("--b", required=True, help="the challenger")
    cp.add_argument("--a-tables", default=None, help="tables JSONL for the baseline")
    cp.add_argument("--b-tables", default=None, help="tables JSONL for the challenger")
    cp.add_argument("--output", default="comparison.md")
    cp.add_argument("--fail-on-regression", action="store_true", help="exit 2 if anything got worse, for CI")
    cp.set_defaults(func=cmd_compare)


def _add_results_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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


def _add_db_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    di = sub.add_parser("db-import", help="load a results file into the SQLite store, idempotently")
    di.add_argument("input", metavar="results.jsonl")
    di.add_argument("--db", default=None, help="defaults to db.path beside the results file")
    di.add_argument("--tables", default=None, help="tables JSONL to load as well")
    di.add_argument("--provenance", default=None, help="cascade provenance JSONL to load as well")
    di.set_defaults(func=cmd_db_import)

    de = sub.add_parser("db-export", help="write the SQLite store back out as results JSONL")
    de.add_argument("output", metavar="out.jsonl")
    de.add_argument("--db", default=None, help="defaults to db.path beside the output file")
    de.add_argument("--cabinet", default=None, help="only this cabinet")
    de.set_defaults(func=cmd_db_export)


def _add_client_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    cp = sub.add_parser("client-pack", help="build a client pack, or copy its documents out of the share")
    actions = cp.add_subparsers(dest="action", required=True)
    b = actions.add_parser("build", help="build or refresh a pack from inventory.csv and schema.csv")
    b.add_argument("inventory")
    b.add_argument("schema")
    b.add_argument("--out", required=True, help="pack directory, outside the repo")
    b.add_argument("--all-types", action="store_true", help="every schema type in taxonomy.yaml")
    b.add_argument("--share", default=None, help="share root, recorded in pack.yaml")
    b.set_defaults(func=cmd_client_pack_build)
    m = actions.add_parser("materialize", help="copy one cabinet's documents into a work directory")
    m.add_argument("--pack", required=True)
    m.add_argument("--cabinet", required=True)
    m.add_argument("--share", required=True, help="share root the inventory was taken of")
    m.add_argument("--work", required=True, help="work directory, outside the share")
    m.set_defaults(func=cmd_client_pack_materialize)
    for action in (b, m):
        action.add_argument("--dry-run", action="store_true", help="log what would be written and stop")

    ce = sub.add_parser("client-eval", help="score a run over one cabinet against its client pack")
    ce.add_argument("--pack", required=True)
    ce.add_argument("--cabinet", required=True)
    ce.add_argument("--results", required=True)
    ce.add_argument("--out", dest="output", default=None, help="report file; stdout when omitted")
    ce.set_defaults(func=cmd_client_eval)

    cr = sub.add_parser("client-run", help="materialize, run and score one cabinet of a client pack")
    cr.add_argument("--pack", required=True)
    cr.add_argument("--cabinet", required=True)
    cr.add_argument("--share", required=True, help="share root the inventory was taken of")
    cr.add_argument("--work", required=True, help="work directory, outside the share")
    cr.add_argument("--cascade", action=argparse.BooleanOptionalAction, default=True, help="run the model stage")
    cr.add_argument("--dry-run", action="store_true", help="log what would be written and stop")
    cr.set_defaults(func=cmd_client_run)


def build_parser() -> argparse.ArgumentParser:
    """The `baseline` argument parser with every subcommand.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(prog="baseline", description="Baseline document pipeline")
    parser.add_argument("--config", default=None, help="config directory, defaults to ./config")
    parser.add_argument("--root", default=".", help="root for relative model paths")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(sub)
    _add_train_parsers(sub)
    _add_profile_parser(sub)
    _add_label_parsers(sub)
    _add_workbook_parsers(sub)
    _add_eval_parsers(sub)
    _add_results_parsers(sub)
    _add_db_parsers(sub)
    _add_client_parsers(sub)
    for name, command in sub.choices.items():
        if name not in OWN_DRY_RUN:  # these add their own --dry-run
            command.add_argument("--dry-run", action="store_true", help="log what would be written and stop")
    return parser


def _dry_run(args: argparse.Namespace) -> int:
    """Log the files a command would write, and how many documents it would read."""
    outputs = [getattr(args, k, None) for k in ("output", "output_model", "report", "json", "dot")]
    if args.command == "run":
        outputs.append(args.bbox_dir or f"{args.output}.bboxes")
    if args.command == "db-import":
        outputs.append(args.db or _cfg(args).db_path(args.input))
    source = Path(getattr(args, "input", None) or getattr(args, "annotations", None) or "")
    documents = None
    if source.is_dir():
        documents = len(scan_documents(source)[0])
    elif source.is_file():
        with open(source, encoding="utf-8") as fh:
            documents = sum(1 for line in fh if line.strip())
    return report_dry_run(outputs, documents)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run one command.

    Args:
        argv: Arguments without the program name; None reads sys.argv.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        if getattr(args, "dry_run", False) and args.command not in OWN_DRY_RUN:
            return _dry_run(args)
        args.cfg = load_config_cached(args.config)
        return int(args.func(args))
    except KeyboardInterrupt:
        log.warning("interrupted")
        return EXIT_INTERRUPTED
    except Exception:  # noqa: BLE001 - CLI top level: log any failure and exit 1, not a traceback
        log.exception("command failed", extra={"command": args.command})
        return 1


if __name__ == "__main__":
    sys.exit(main())

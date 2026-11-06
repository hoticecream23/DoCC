"""One document in, one record out, through the model and the grounding step.

The baseline runs first and its record is the fallback. If the model cannot
be reached, or its answer is not the shape asked for, the document gets the
baseline's answer and an error saying so, never an empty record.

Batch output is appended one document at a time so a run can resume, so it
is not written to a temp file and renamed. Instead each document writes its
provenance line, then its results line; the results line is the commit
marker. On resume a trailing partial line is cut from both files and any
provenance line without a results line is dropped, so the pair never
diverges. A run with resume off truncates both files up front.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TypedDict, TypeVar

from baseline import pipeline as baseline_pipeline
from baseline.classify import rule_scores
from baseline.config import Config
from baseline.io import atomic_open, cut_partial_tail
from baseline.logging_setup import get_logger
from baseline.scan import scan_documents
from baseline.schema import (
    BBoxSidecar,
    Classification,
    Diagnostics,
    DocumentRecord,
    MetadataField,
    Tag,
    compute_doc_id,
)
from baseline.tagging import apply_tags

from .config import CascadeConfig
from .merge import ClassProvenance, Decision, FieldProvenance, decide
from .prompt import VLMAnswer, build_messages, check_descriptions, parse_answer, response_schema
from .render import render_pages
from .vlm import Backend, VLMBadRequest, VLMReply

log = get_logger(__name__)

# Errors that mean a bug here, not a bad document, so the traceback is worth logging.
_BUGS = (TypeError, AttributeError, NameError, AssertionError)
# Exception messages can quote the document, so logs keep only this many characters.
ERROR_MESSAGE_CHARS = 200

T = TypeVar("T")

_ProvenanceBase = TypedDict(
    "_ProvenanceBase",
    {
        "doc_id": str,
        "filename": str,
        "model": "str | None",
        "finish_reason": "str | None",
        "pages_sent": int,
        "images_sent": int,
        "fallback": bool,
        "errors": "list[str]",
        "raw": "str | None",
        "class": "ClassProvenance | None",
        "fields": "list[FieldProvenance]",
        "timings_ms": "dict[str, float]",
    },
)


class Provenance(_ProvenanceBase, total=False):
    """One line of the provenance sidecar: how the model's answer became the record."""

    # Set only when a context overflow forced a retry with fewer pages.
    reduced_pages: int


class BatchStats(TypedDict):
    """Counts run_batch logs and returns."""

    found: int
    skipped: int
    written: int
    failed: int
    fallback: int
    excluded: dict[str, int]


def _error_extra(exc: Exception) -> dict[str, str]:
    """Exception type and a short message; the full text can quote the document."""
    return {"error_type": type(exc).__name__, "error": str(exc)[:ERROR_MESSAGE_CHARS]}


@dataclass
class Result:
    """One document's record, word boxes and provenance."""

    record: DocumentRecord
    sidecar: BBoxSidecar
    provenance: Provenance


class _Timer:
    """Times steps into `timings` as cascade_<name> in ms; zero when deterministic."""

    def __init__(self, deterministic: bool) -> None:
        self.deterministic = deterministic
        self.timings: dict[str, float] = {}

    def __call__(self, name: str, fn: Callable[[], T]) -> T:
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            elapsed = 0.0 if self.deterministic else round((time.perf_counter() - t0) * 1000.0, 3)
            self.timings[f"cascade_{name}"] = elapsed


def _diagnostics(base: DocumentRecord, timings: dict[str, float], errors: list[str]) -> Diagnostics:
    d = base.diagnostics
    return Diagnostics(
        page_count=d.page_count,
        is_scanned=d.is_scanned,
        has_native_text=d.has_native_text,
        timings_ms={**d.timings_ms, **timings},
        errors=[*d.errors, *errors],
    )


def _record(
    base: DocumentRecord,
    classification: Classification,
    metadata: list[MetadataField],
    tags: list[Tag],
    diagnostics: Diagnostics,
) -> DocumentRecord:
    return DocumentRecord(
        doc_id=base.doc_id,
        source_path=base.source_path,
        filename=base.filename,
        text=base.text,
        classification=classification,
        metadata=metadata,
        tags=tags,
        diagnostics=diagnostics,
    )


def _drop_bad_offsets(meta: list[MetadataField], full: str) -> tuple[list[MetadataField], list[str]]:
    """Belt and braces. Grounding guarantees this, and the write checks again."""
    keep, problems = [], []
    for m in meta:
        if m.char_end <= len(full) and full[m.char_start : m.char_end] == m.value:
            keep.append(m)
        else:
            problems.append(f"{m.field}: text[{m.char_start}:{m.char_end}] != {m.value!r}, dropped")
    return keep, problems


def _new_provenance(base: DocumentRecord, timings: dict[str, float]) -> Provenance:
    return {
        "doc_id": base.doc_id,
        "filename": base.filename,
        "model": None,
        "finish_reason": None,
        "pages_sent": 0,
        "images_sent": 0,
        "fallback": False,
        "errors": [],
        "raw": None,
        "class": None,
        "fields": [],
        "timings_ms": timings,
    }


def _ask_model(
    path: Path,
    base: DocumentRecord,
    cfg: Config,
    ccfg: CascadeConfig,
    backend: Backend,
    prov: Provenance,
    timed: _Timer,
) -> VLMReply:
    """Render the pages and ask the model, once more with half the pages on a 400."""
    r = ccfg.render
    images = timed("render", lambda: render_pages(path, r.max_pages, r.dpi, r.max_long_edge))
    sent = base.text.pages[: r.max_pages]
    prov["pages_sent"], prov["images_sent"] = len(sent), len(images)
    if not images and not base.text.full.strip():
        raise ValueError("nothing to send: no page image and no text")
    messages = build_messages(cfg, sent, images, r.max_text_chars)
    try:
        return timed("vlm", lambda: backend.complete(messages, response_schema(cfg)))
    except VLMBadRequest:
        # ponytail: one retry with half the page budget for a context overflow;
        # a one page document is resent as is, downscale images if that keeps failing.
        n = max(1, r.max_pages // 2)
        sent, images = sent[:n], images[:n]
        prov["reduced_pages"], prov["pages_sent"], prov["images_sent"] = n, len(sent), len(images)
        retry = build_messages(cfg, sent, images, r.max_text_chars)
        return timed("vlm", lambda: backend.complete(retry, response_schema(cfg)))


def _decide(
    base: DocumentRecord, sidecar: BBoxSidecar, cfg: Config, ccfg: CascadeConfig, answer: VLMAnswer
) -> Decision:
    full, pages = base.text.full, base.text.pages
    rules = rule_scores(full, pages, cfg)
    return decide(full, pages, cfg, ccfg.confidence, rules, answer, base.metadata, sidecar.words, sidecar.page_sizes)


def _fallback(
    base: DocumentRecord, sidecar: BBoxSidecar, prov: Provenance, timings: dict[str, float], exc: Exception
) -> Result:
    """The baseline's record, with the cascade error in its diagnostics."""
    msg = f"cascade: {type(exc).__name__}: {exc}"
    log.error(
        "cascade fell back to the baseline",
        extra={"doc_id": base.doc_id, **_error_extra(exc)},
        exc_info=isinstance(exc, _BUGS),
    )
    prov["fallback"] = True
    prov["errors"].append(msg)
    meta, problems = _drop_bad_offsets(base.metadata, base.text.full)
    prov["errors"].extend(problems)
    record = _record(base, base.classification, meta, base.tags, _diagnostics(base, timings, [msg, *problems]))
    return Result(record, sidecar, prov)


def process_document(
    path: str | Path,
    cfg: Config,
    ccfg: CascadeConfig,
    backend: Backend,
    models: baseline_pipeline.Models | None = None,
) -> Result:
    """Run the baseline, then the model, and merge them into one record.

    Any failure after the baseline gives the baseline's record and an error.

    Args:
        path: The document.
        cfg: The loaded config.
        ccfg: The cascade config.
        backend: The model client.
        models: Baseline models; None runs rules only.

    Returns:
        The record, word boxes and provenance.

    Raises:
        Exception: Only if the baseline cannot read the file at all.
    """
    p = Path(path)
    base, sidecar = baseline_pipeline.process_document(p, cfg, models)
    timed = _Timer(bool(cfg.pipeline.get("runtime", {}).get("deterministic_timings", False)))
    prov = _new_provenance(base, timed.timings)
    try:
        reply = _ask_model(p, base, cfg, ccfg, backend, prov, timed)
        prov["model"], prov["finish_reason"], prov["raw"] = reply.model, reply.finish_reason, reply.content
        answer = parse_answer(reply.content, cfg)
        decision = timed("decide", lambda: _decide(base, sidecar, cfg, ccfg, answer))
        meta, problems = _drop_bad_offsets(decision.metadata, base.text.full)
        facts = baseline_pipeline.tag_facts(base.text, base.diagnostics.page_count, base.diagnostics.is_scanned)
        tags = apply_tags(base.text.full, base.text.pages, cfg, meta, facts, models.tagger if models else None)
    except Exception as exc:  # noqa: BLE001 - any cascade failure falls back to the baseline; _fallback logs it
        return _fallback(base, sidecar, prov, timed.timings, exc)

    prov["class"], prov["fields"] = decision.class_provenance, decision.field_provenance
    prov["errors"].extend(problems)
    record = _record(base, decision.classification, meta, tags, _diagnostics(base, timed.timings, problems))
    return Result(record, sidecar, prov)


def _prepare_resume(out_path: Path, prov_path: Path) -> set[str]:
    """Repair the output pair after a crash and return the doc_ids already done."""
    cut_partial_tail(out_path)
    cut_partial_tail(prov_path)
    done = baseline_pipeline.existing_doc_ids(out_path)
    if prov_path.exists():
        lines = prov_path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if _prov_doc_id(line) in done]
        if len(kept) != len(lines):
            log.warning("dropping provenance lines with no result", extra={"count": len(lines) - len(kept)})
            with atomic_open(prov_path, newline="\n") as prov_fh:
                prov_fh.writelines(kept)
    return done


def _prov_doc_id(line: str) -> str | None:
    try:
        doc_id: str | None = json.loads(line)["doc_id"]
        return doc_id
    except (ValueError, KeyError, TypeError):
        return None


def _process_or_count(
    path: Path,
    skip: set[str],
    stats: BatchStats,
    cfg: Config,
    ccfg: CascadeConfig,
    backend: Backend,
    models: baseline_pipeline.Models | None,
) -> Result | None:
    """Process one document; None when it is skipped or fails, counted in stats."""
    try:
        if compute_doc_id(path) in skip:
            stats["skipped"] += 1
            return None
        return process_document(path, cfg, ccfg, backend, models)
    except Exception as exc:  # noqa: BLE001 - per-document loop: one bad document must not stop the batch
        log.error("document failed", extra={"path": str(path), **_error_extra(exc)}, exc_info=isinstance(exc, _BUGS))
        stats["failed"] += 1
        return None


def _write_outcome(outcome: Result, bbox_dir: Path, results_fh: IO[str], prov_fh: IO[str]) -> None:
    """Write word boxes, then provenance, then the results line that marks the document done."""
    baseline_pipeline.write_sidecar(outcome.sidecar, bbox_dir)
    prov_fh.write(json.dumps(outcome.provenance, ensure_ascii=False) + "\n")
    prov_fh.flush()
    results_fh.write(baseline_pipeline.dumps(outcome.record) + "\n")
    results_fh.flush()


def _output_paths(output: str | Path, bbox_dir: str | Path | None) -> tuple[Path, Path, Path]:
    """Results, provenance and word box paths; creates the results directory."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path = out_path.with_name(out_path.name + ".provenance")
    return out_path, prov_path, Path(bbox_dir) if bbox_dir else out_path.with_name(out_path.name + ".bboxes")


def run_batch(
    input_dir: str | Path,
    output: str | Path,
    cfg: Config,
    ccfg: CascadeConfig,
    backend: Backend,
    models: baseline_pipeline.Models | None = None,
    bbox_dir: str | Path | None = None,
    resume: bool = True,
) -> BatchStats:
    """Process a directory into JSONL plus a provenance sidecar. Resumable.

    Args:
        input_dir: Directory of documents.
        output: Results JSONL path.
        cfg: The loaded config.
        ccfg: The cascade config.
        backend: The model client.
        models: Baseline models; None runs rules only.
        bbox_dir: Word box sidecar directory; defaults to <output>.bboxes.
        resume: Skip documents already written, after repairing a crash.

    Returns:
        Counts of documents found, skipped, written, failed and fallen back.

    Raises:
        ValueError: A class or field has no description in config.
    """
    check_descriptions(cfg)
    out_path, prov_path, bdir = _output_paths(output, bbox_dir)
    paths, excluded = scan_documents(input_dir)
    skip = _prepare_resume(out_path, prov_path) if resume else set()
    excluded_counts = {reason: len(items) for reason, items in sorted(excluded.items())}
    stats = BatchStats(found=len(paths), skipped=0, written=0, failed=0, fallback=0, excluded=excluded_counts)
    mode = "a" if resume else "w"
    with (
        open(out_path, mode, encoding="utf-8", newline="\n") as results_fh,
        open(prov_path, mode, encoding="utf-8", newline="\n") as prov_fh,
    ):
        for p in paths:
            outcome = _process_or_count(p, skip, stats, cfg, ccfg, backend, models)
            if outcome is None:
                continue
            _write_outcome(outcome, bdir, results_fh, prov_fh)
            stats["written"] += 1
            stats["fallback"] += int(outcome.provenance["fallback"])

    log.info("cascade batch done", extra=stats)
    return stats

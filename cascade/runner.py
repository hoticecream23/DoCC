"""One document in, one record out, through the model and the grounding step.

The baseline runs first and its record is the fallback. If the model cannot
be reached, or its answer is not the shape asked for, the document gets the
baseline's answer and an error saying so, never an empty record.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baseline import pipeline as baseline_pipeline
from baseline.classify import rule_scores
from baseline.config import Config
from baseline.logging_setup import get_logger
from baseline.scan import scan_documents
from baseline.schema import BBoxSidecar, Diagnostics, DocumentRecord, MetadataField, compute_doc_id
from baseline.tagging import apply_tags

from .config import CascadeConfig
from .merge import decide
from .prompt import build_messages, check_descriptions, parse_answer, response_schema
from .render import render_pages
from .vlm import Backend, VLMBadRequest

log = get_logger(__name__)


@dataclass
class Result:
    record: DocumentRecord
    sidecar: BBoxSidecar
    provenance: dict


def _diagnostics(base: DocumentRecord, timings: dict[str, float], errors: list[str]) -> Diagnostics:
    d = base.diagnostics
    return Diagnostics(
        page_count=d.page_count,
        is_scanned=d.is_scanned,
        has_native_text=d.has_native_text,
        timings_ms={**d.timings_ms, **timings},
        errors=[*d.errors, *errors],
    )


def _record(base: DocumentRecord, classification, metadata, tags, diagnostics) -> DocumentRecord:
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


def process_document(
    path: str | Path,
    cfg: Config,
    ccfg: CascadeConfig,
    backend: Backend,
    models: baseline_pipeline.Models | None = None,
) -> Result:
    """Raises only if the baseline cannot read the file at all."""
    p = Path(path)
    base, sidecar = baseline_pipeline.process_document(p, cfg, models)
    deterministic = bool(cfg.pipeline.get("runtime", {}).get("deterministic_timings", False))
    timings: dict[str, float] = {}
    prov: dict[str, Any] = {
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

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            timings[f"cascade_{name}"] = 0.0 if deterministic else round((time.perf_counter() - t0) * 1000.0, 3)

    full, pages = base.text.full, base.text.pages
    try:
        r = ccfg.render
        images = timed("render", lambda: render_pages(p, r.max_pages, r.dpi, r.max_long_edge))
        sent = pages[: r.max_pages]
        prov["pages_sent"], prov["images_sent"] = len(sent), len(images)
        if not images and not full.strip():
            raise ValueError("nothing to send: no page image and no text")
        messages = build_messages(cfg, sent, images, r.max_text_chars)
        try:
            reply = timed("vlm", lambda: backend.complete(messages, response_schema(cfg)))
        except VLMBadRequest:
            # ponytail: one retry with half the page budget for a context overflow;
            # a one page document is resent as is, downscale images if that keeps failing.
            n = max(1, r.max_pages // 2)
            sent, images = sent[:n], images[:n]
            prov["reduced_pages"], prov["pages_sent"], prov["images_sent"] = n, len(sent), len(images)
            messages = build_messages(cfg, sent, images, r.max_text_chars)
            reply = timed("vlm", lambda: backend.complete(messages, response_schema(cfg)))
        prov["model"], prov["finish_reason"], prov["raw"] = reply.model, reply.finish_reason, reply.content
        answer = parse_answer(reply.content, cfg)
        decision = timed(
            "decide",
            lambda: decide(
                full,
                pages,
                cfg,
                ccfg.confidence,
                rule_scores(full, pages, cfg),
                answer,
                base.metadata,
                sidecar.words,
                sidecar.page_sizes,
            ),
        )
        meta, problems = _drop_bad_offsets(decision.metadata, full)
        facts = baseline_pipeline.tag_facts(base.text, base.diagnostics.page_count, base.diagnostics.is_scanned)
        tags = apply_tags(full, pages, cfg, meta, facts, models.tagger if models else None)
    except Exception as exc:
        msg = f"cascade: {type(exc).__name__}: {exc}"
        log.error("cascade fell back to the baseline", extra={"doc_id": base.doc_id, "error": msg}, exc_info=True)
        prov["fallback"] = True
        prov["errors"].append(msg)
        meta, problems = _drop_bad_offsets(base.metadata, full)
        prov["errors"].extend(problems)
        record = _record(base, base.classification, meta, base.tags, _diagnostics(base, timings, [msg, *problems]))
        return Result(record, sidecar, prov)

    prov["class"], prov["fields"] = decision.class_provenance, decision.field_provenance
    prov["errors"].extend(problems)
    record = _record(base, decision.classification, meta, tags, _diagnostics(base, timings, problems))
    return Result(record, sidecar, prov)


def run_batch(
    input_dir: str | Path,
    output: str | Path,
    cfg: Config,
    ccfg: CascadeConfig,
    backend: Backend,
    models: baseline_pipeline.Models | None = None,
    bbox_dir: str | Path | None = None,
    resume: bool = True,
) -> dict:
    """Process a directory into JSONL plus a provenance sidecar. Resumable."""
    check_descriptions(cfg)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path = out_path.with_name(out_path.name + ".provenance")
    bdir = Path(bbox_dir) if bbox_dir else out_path.with_name(out_path.name + ".bboxes")

    paths, excluded = scan_documents(input_dir)
    skip = baseline_pipeline.existing_doc_ids(out_path) if resume else set()
    stats = {
        "found": len(paths),
        "skipped": 0,
        "written": 0,
        "failed": 0,
        "fallback": 0,
        "excluded": {reason: len(items) for reason, items in sorted(excluded.items())},
    }
    mode = "a" if resume else "w"

    with (
        open(out_path, mode, encoding="utf-8", newline="\n") as fh,
        open(prov_path, mode, encoding="utf-8", newline="\n") as ph,
    ):
        for p in paths:
            try:
                if compute_doc_id(p) in skip:
                    stats["skipped"] += 1
                    continue
                result = process_document(p, cfg, ccfg, backend, models)
            except Exception as exc:
                log.error("document failed", extra={"path": str(p), "error": str(exc)})
                stats["failed"] += 1
                continue
            baseline_pipeline.write_sidecar(result.sidecar, bdir)
            fh.write(baseline_pipeline.dumps(result.record) + "\n")
            ph.write(json.dumps(result.provenance, ensure_ascii=False) + "\n")
            fh.flush()
            ph.flush()
            stats["written"] += 1
            stats["fallback"] += int(result.provenance["fallback"])

    log.info("cascade batch done", extra=stats)
    return stats

"""Orchestration. One document in, one record out.

Every stage is wrapped. A stage that blows up records an error in diagnostics
and the record still gets written, partial. One bad PDF never kills a batch.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import classify as classify_mod
from . import tagging as tagging_mod
from .config import Config, load_config_cached
from .extract import ExtractionResult, extract, to_canonical_offsets
from .logging_setup import get_logger, setup_logging
from .metadata import extract_fields
from .scan import scan_documents
from .schema import (
    BBoxSidecar,
    Classification,
    ClassifyMethod,
    Diagnostics,
    DocumentRecord,
    ExtractionMethod,
    TextRecord,
    build_canonical_text,
    compute_doc_id,
)

log = get_logger(__name__)


@dataclass
class Models:
    classifier: Any = None
    tagger: Any = None

    @classmethod
    def load(cls, cfg: Config, root: Path) -> "Models":
        cdir = root / str(cfg.classify_opts().get("model_dir", "models/classifier"))
        tdir = root / str(cfg.tagging_opts().get("model_dir", "models/tagger"))
        return cls(
            classifier=classify_mod.load_model(cdir),
            tagger=tagging_mod.load_model(tdir),
        )


class _Stages:
    """Times each stage and swallows its failures into an error list."""

    def __init__(self, deterministic: bool = False):
        self.timings: dict[str, float] = {}
        self.errors: list[str] = []
        self.deterministic = deterministic

    def run(self, name: str, fn: Callable[[], Any], default: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return fn()
        except Exception as exc:
            log.exception("stage failed", extra={"stage": name})
            self.errors.append(f"{name}: {type(exc).__name__}: {exc}")
            return default
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.timings[name] = 0.0 if self.deterministic else round(elapsed, 3)


def _text_record(res: ExtractionResult) -> TextRecord:
    full = build_canonical_text(res.pages)
    return TextRecord(
        full=full,
        pages=res.pages,
        method=res.method,
        char_count=len(full),
        extraction_confidence=res.confidence(),
        page_methods=res.page_methods,
    )


def process_document(
    path: str | Path, cfg: Config, models: Models | None = None
) -> tuple[DocumentRecord, BBoxSidecar]:
    """The whole pipeline for one file. Raises only if the file is unreadable."""
    p = Path(path)
    models = models or Models()
    deterministic = bool(cfg.pipeline.get("runtime", {}).get("deterministic_timings", False))
    st = _Stages(deterministic)

    doc_id = compute_doc_id(p)

    res: ExtractionResult = st.run("extract", lambda: extract(p, cfg), ExtractionResult())
    if not res.pages:
        res.pages.append("")
        res.page_methods.append(ExtractionMethod.NONE)
        res.page_confidences.append(0.0)
    st.errors.extend(res.errors)
    # Word boxes become canonical here so nothing downstream sees page local ones.
    to_canonical_offsets(res)

    text = _text_record(res)
    text.check_consistent()

    cls: Classification = st.run(
        "classify",
        lambda: classify_mod.classify(text.full, text.pages, cfg, models.classifier),
        Classification(
            label=str(cfg.classify_opts().get("unknown_label", "unknown")),
            confidence=0.0,
            all_scores={},
            method=ClassifyMethod.FALLBACK,
        ),
    )

    def _meta():
        fields, errs = extract_fields(
            text.full, text.pages, cfg, cls.label, res.words, res.page_sizes
        )
        st.errors.extend(errs)
        return fields

    meta = st.run("metadata", _meta, [])

    facts = {
        "page_count": res.page_count,
        "char_count": text.char_count,
        "extraction_confidence": text.extraction_confidence,
        "is_scanned": res.is_scanned,
        "method": text.method.value,
    }
    tags = st.run(
        "tag",
        lambda: tagging_mod.apply_tags(
            text.full, text.pages, cfg, meta, facts, models.tagger
        ),
        [],
    )

    record = DocumentRecord(
        doc_id=doc_id,
        source_path=p.as_posix(),
        filename=p.name,
        text=text,
        classification=cls,
        metadata=meta,
        tags=tags,
        diagnostics=Diagnostics(
            page_count=res.page_count,
            is_scanned=res.is_scanned,
            has_native_text=res.has_native_text,
            timings_ms=st.timings,
            errors=st.errors,
        ),
    )

    # Belt and braces. Offsets were checked at build time, check again at write.
    problems = record.verify_offsets()
    if problems:
        log.error("offset verification failed", extra={"doc_id": doc_id, "n": len(problems)})
        record.diagnostics.errors.extend(problems)

    sidecar = BBoxSidecar(doc_id=doc_id, page_sizes=res.page_sizes, words=res.words)
    return record, sidecar


# --------------------------------------------------------------------------
# batch running
# --------------------------------------------------------------------------


def dumps(record: DocumentRecord) -> str:
    """One JSON line. Key order is fixed by the model, so runs diff cleanly."""
    return json.dumps(record.model_dump(mode="json"), ensure_ascii=False)


def existing_doc_ids(output: str | Path) -> set[str]:
    """doc_ids already in the output file, so a run can resume."""
    p = Path(output)
    if not p.exists():
        return set()
    seen = set()
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["doc_id"])
            except Exception:
                continue
    return seen


def _write_sidecar(sidecar: BBoxSidecar, bbox_dir: Path) -> None:
    if not sidecar.words:
        return
    bbox_dir.mkdir(parents=True, exist_ok=True)
    target = bbox_dir / f"{sidecar.doc_id}.json.gz"
    payload = json.dumps(sidecar.model_dump(mode="json"), ensure_ascii=False)
    # mtime pinned so the bytes are reproducible
    with gzip.GzipFile(target, "wb", mtime=0) as fh:
        fh.write(payload.encode("utf-8"))


_STATE: dict[str, Any] = {}


def _init_worker(config_dir: str | None, root: str, log_level: str) -> None:
    setup_logging(log_level)
    cfg = load_config_cached(config_dir)
    _STATE["cfg"] = cfg
    _STATE["models"] = Models.load(cfg, Path(root))
    _STATE["bbox_dir"] = None


def _run_one(args: tuple) -> str:
    path, bbox_dir = args
    cfg: Config = _STATE["cfg"]
    models: Models = _STATE["models"]
    try:
        record, sidecar = process_document(path, cfg, models)
        if bbox_dir:
            _write_sidecar(sidecar, Path(bbox_dir))
        return dumps(record)
    except Exception as exc:
        # Unreadable file. Emit nothing but say so loudly.
        log.error("document failed", extra={"path": str(path), "error": str(exc)})
        return ""


def run_batch(
    input_dir: str | Path,
    output: str | Path,
    config_dir: str | None = None,
    workers: int = 1,
    bbox_dir: str | Path | None = None,
    resume: bool = True,
    root: str | Path = ".",
) -> dict:
    """Process a directory into JSONL. Resumable, parallel, order stable."""
    cfg = load_config_cached(config_dir)
    log_level = str(cfg.pipeline.get("runtime", {}).get("log_level", "INFO"))
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bdir = Path(bbox_dir) if bbox_dir else out_path.with_suffix(out_path.suffix + ".bboxes")

    paths, excluded = scan_documents(input_dir)
    if excluded:
        # Say what was left out and why. A total that quietly omits input does
        # not mean what it appears to mean.
        log.info("files excluded as not documents",
                 extra={reason: len(items) for reason, items in sorted(excluded.items())})
    skip = existing_doc_ids(out_path) if resume else set()
    todo = []
    for p in paths:
        try:
            if compute_doc_id(p) in skip:
                continue
        except OSError as exc:
            log.error("cannot hash file", extra={"path": str(p), "error": str(exc)})
            continue
        todo.append(p)

    log.info(
        "batch starting",
        extra={"found": len(paths), "skipped": len(paths) - len(todo), "todo": len(todo),
               "excluded": sum(len(v) for v in excluded.values()), "workers": workers},
    )

    stats = {"found": len(paths), "skipped": len(paths) - len(todo), "written": 0, "failed": 0,
             "excluded": {reason: len(items) for reason, items in sorted(excluded.items())}}
    if not todo:
        return stats

    jobs = [(str(p), str(bdir)) for p in todo]
    mode = "a" if out_path.exists() and resume else "w"

    with open(out_path, mode, encoding="utf-8", newline="\n") as fh:
        if workers and workers > 1:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=(config_dir, str(root), log_level),
            ) as pool:
                # imap keeps input order, so the output file is deterministic
                for line in pool.imap(_run_one, jobs, chunksize=1):
                    if line:
                        fh.write(line + "\n")
                        stats["written"] += 1
                    else:
                        stats["failed"] += 1
        else:
            _init_worker(config_dir, str(root), log_level)
            for job in jobs:
                line = _run_one(job)
                if line:
                    fh.write(line + "\n")
                    stats["written"] += 1
                else:
                    stats["failed"] += 1

    log.info("batch done", extra=stats)
    return stats

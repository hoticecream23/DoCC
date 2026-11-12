"""Orchestration. One document in, one record out.

Over 400 lines: each stage wrapper sits beside the one flow it guards; splitting would hide the stage order.

Every stage is wrapped. A stage that blows up records an error in diagnostics
and the record still gets written, partial. One bad PDF never kills a batch.

Write order in run_batch: the results line is written and flushed first; only
then is the same record upserted into the SQLite store (`db.path` in
pipeline.yaml, see baseline/db.py). The JSONL is the log. A store failure is
logged and the batch goes on, so it can never lose or block a results line;
`baseline db-import` rebuilds the store from the JSONL afterwards.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, TypedDict, TypeVar

from . import classify as classify_mod
from . import db, packmode
from . import tagging as tagging_mod
from .config import Config, load_config_cached
from .errors import summarize_exception
from .extract import ExtractionResult, extract, to_canonical_offsets
from .io import cut_partial_tail
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
    MetadataField,
    Tag,
    TextRecord,
    build_canonical_text,
    compute_doc_id,
)

log = get_logger(__name__)

T = TypeVar("T")


@dataclass
class Models:
    """The trained models a run uses; either may be missing.

    Attributes:
        classifier: Document classifier, or None to use rules only.
        tagger: Multi-label tagger, or None to use rules only.
    """

    classifier: classify_mod.ClassifierModel | None = None
    tagger: tagging_mod.TaggerModel | None = None

    @classmethod
    def load(cls, cfg: Config, root: Path) -> Models:
        """Load both models from the directories the config names.

        Args:
            cfg: Loaded configuration.
            root: Directory the configured model paths are relative to.

        Returns:
            The models; a missing model directory leaves that slot None.
        """
        cdir = root / str(cfg.classify_opts().get("model_dir", "models/classifier"))
        tdir = root / str(cfg.tagging_opts().get("model_dir", "models/tagger"))
        return cls(
            classifier=classify_mod.load_model(cdir),
            tagger=tagging_mod.load_model(tdir),
        )


class _Stages:
    """Times each stage and swallows its failures into an error list."""

    def __init__(self, deterministic: bool = False) -> None:
        self.timings: dict[str, float] = {}
        self.errors: list[str] = []
        self.deterministic = deterministic

    def run(self, name: str, fn: Callable[[], T], default: T) -> T:
        t0 = time.perf_counter()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - boundary: a failed stage records an error, the document goes on
            log.error("stage failed", extra={"stage": name, "error": summarize_exception(exc)})
            log.debug("stage traceback", exc_info=True)  # debug only: the message can hold document text
            self.errors.append(f"{name}: {summarize_exception(exc)}")
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


def tag_facts(text: TextRecord, page_count: int, is_scanned: bool) -> dict[str, Any]:
    """Document level facts the derived tag rules read.

    Cascade builds its tags from the same function, so the two pipelines
    cannot drift on what a tag rule sees.

    Args:
        text: The document's text record.
        page_count: Number of pages.
        is_scanned: True when no page had a text layer.

    Returns:
        Fact name to value.
    """
    return {
        "page_count": page_count,
        "char_count": text.char_count,
        "extraction_confidence": text.extraction_confidence,
        "is_scanned": is_scanned,
        "method": text.method.value,
    }


def _extract_stage(st: _Stages, path: Path, cfg: Config) -> ExtractionResult:
    """Extract text, guarantee at least one page, and make word offsets canonical."""
    res: ExtractionResult = st.run("extract", lambda: extract(path, cfg), ExtractionResult())
    if not res.pages:
        res.pages.append("")
        res.page_methods.append(ExtractionMethod.NONE)
        res.page_confidences.append(0.0)
    st.errors.extend(res.errors)
    # Word boxes become canonical here so nothing downstream sees page local ones.
    to_canonical_offsets(res)
    return res


def _classify_stage(st: _Stages, text: TextRecord, cfg: Config, models: Models) -> Classification:
    fallback = Classification(
        label=str(cfg.classify_opts().get("unknown_label", "unknown")),
        confidence=0.0,
        all_scores={},
        method=ClassifyMethod.FALLBACK,
    )
    if packmode.is_active(cfg):
        # The trained model does not know the pack's classes: the keyword prior decides.
        return st.run("classify", lambda: packmode.classify(text.full, text.pages, cfg), fallback)
    return st.run("classify", lambda: classify_mod.classify(text.full, text.pages, cfg, models.classifier), fallback)


def _metadata_stage(
    st: _Stages, text: TextRecord, cfg: Config, label: str, res: ExtractionResult
) -> list[MetadataField]:
    def fields_or_raise() -> list[MetadataField]:
        fields, errs = extract_fields(text.full, text.pages, cfg, label, res.words, res.page_sizes)
        st.errors.extend(errs)
        return fields

    empty: list[MetadataField] = []
    return st.run("metadata", fields_or_raise, empty)


def _tag_stage(
    st: _Stages, text: TextRecord, cfg: Config, meta: list[MetadataField], res: ExtractionResult, models: Models
) -> list[Tag]:
    facts = tag_facts(text, res.page_count, res.is_scanned)
    empty: list[Tag] = []
    return st.run("tag", lambda: tagging_mod.apply_tags(text.full, text.pages, cfg, meta, facts, models.tagger), empty)


def process_document(
    path: str | Path, cfg: Config, models: Models | None = None, cabinet_id: str | None = None
) -> tuple[DocumentRecord, BBoxSidecar]:
    """The whole pipeline for one file.

    Args:
        path: Document to process.
        cfg: Loaded configuration.
        models: Trained models; None means rules only.
        cabinet_id: Cabinet to stamp; None uses the config's cabinet_id.

    Returns:
        The record and its word-box sidecar.

    Raises:
        OSError: Only if the file cannot be read to hash it.
    """
    p = Path(path)
    models = models or Models()
    deterministic = bool(cfg.pipeline.get("runtime", {}).get("deterministic_timings", False))
    st = _Stages(deterministic)
    doc_id = compute_doc_id(p)

    res = _extract_stage(st, p, cfg)
    text = _text_record(res)
    text.check_consistent()
    cls = _classify_stage(st, text, cfg, models)
    meta = _metadata_stage(st, text, cfg, cls.label, res)
    tags = _tag_stage(st, text, cfg, meta, res, models)

    diagnostics = Diagnostics(
        page_count=res.page_count,
        is_scanned=res.is_scanned,
        has_native_text=res.has_native_text,
        timings_ms=st.timings,
        errors=st.errors,
    )
    record = DocumentRecord(
        doc_id=doc_id,
        source_path=p.as_posix(),
        filename=p.name,
        cabinet_id=cabinet_id or cfg.cabinet_id,
        text=text,
        classification=cls,
        metadata=meta,
        tags=tags,
        diagnostics=diagnostics,
    )
    _verify_offsets(record)
    return record, BBoxSidecar(doc_id=doc_id, page_sizes=res.page_sizes, words=res.words)


def _verify_offsets(record: DocumentRecord) -> None:
    # Belt and braces. Offsets were checked at build time, check again at write.
    problems = record.verify_offsets()
    if problems:
        log.error("offset verification failed", extra={"doc_id": record.doc_id, "n": len(problems)})
        record.diagnostics.errors.extend(problems)


# --------------------------------------------------------------------------
# batch running
# --------------------------------------------------------------------------


def dumps(record: DocumentRecord) -> str:
    """One JSON line. Key order is fixed by the model, so runs diff cleanly.

    Args:
        record: The record to serialise.

    Returns:
        The JSON text, without a trailing newline.
    """
    return json.dumps(record.model_dump(mode="json"), ensure_ascii=False)


def existing_doc_ids(output: str | Path) -> set[str]:
    """Read the doc_ids already in the output file, so a run can resume.

    Args:
        output: The results JSONL.

    Returns:
        Every doc_id on a readable line; empty if the file is missing.
    """
    p = Path(output)
    if not p.exists():
        return set()
    seen = set()
    with open(p, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["doc_id"])
            except (ValueError, KeyError, TypeError) as exc:
                # The doc is re-run, but a bad line here means the output file is damaged.
                log.warning(
                    "unreadable line in output, doc will be re-run",
                    extra={"output": str(p), "error": summarize_exception(exc)},
                )
    return seen


def write_sidecar(sidecar: BBoxSidecar, bbox_dir: Path) -> None:
    """Write the word boxes as gzipped JSON; nothing when there are none.

    Args:
        sidecar: The document's word boxes.
        bbox_dir: Directory for `<doc_id>.json.gz`.
    """
    if not sidecar.words:
        return
    bbox_dir.mkdir(parents=True, exist_ok=True)
    target = bbox_dir / f"{sidecar.doc_id}.json.gz"
    payload = json.dumps(sidecar.model_dump(mode="json"), ensure_ascii=False)
    # mtime pinned so the bytes are reproducible
    with gzip.GzipFile(target, "wb", mtime=0) as fh:
        fh.write(payload.encode("utf-8"))


class BatchStats(TypedDict):
    """Counts a batch run returns and logs."""

    found: int
    skipped: int
    written: int
    failed: int
    excluded: dict[str, int]


_STATE: dict[str, Any] = {}


def _init_worker(config_dir: str | None, root: str, log_level: str, client_pack: tuple[str, str] | None = None) -> None:
    setup_logging(log_level)
    cfg = packmode.resolve(load_config_cached(config_dir), client_pack)
    _STATE["cfg"] = cfg
    _STATE["models"] = Models.load(cfg, Path(root))
    _STATE["bbox_dir"] = None


def _run_one(job: tuple[str, str, str]) -> str:
    path, bbox_dir, cabinet_id = job
    cfg: Config = _STATE["cfg"]
    models: Models = _STATE["models"]
    try:
        record, sidecar = process_document(path, cfg, models, cabinet_id)
        if bbox_dir:
            write_sidecar(sidecar, Path(bbox_dir))
        return dumps(record)
    except Exception as exc:  # noqa: BLE001 - boundary: per-document loop of the batch runner
        # Unreadable file. Emit nothing but say so loudly.
        log.error("document failed", extra={"path": str(path), "error": summarize_exception(exc)})
        return ""


def _pending(paths: list[Path], out_path: Path, resume: bool) -> list[Path]:
    """Paths not yet in the output. On resume, first cut a line a crash left half written."""
    skip: set[str] = set()
    if resume:
        cut_partial_tail(out_path)
        skip = existing_doc_ids(out_path)
    todo = []
    for p in paths:
        try:
            if compute_doc_id(p) in skip:
                continue
        except OSError as exc:
            log.error("cannot hash file", extra={"path": str(p), "error": str(exc)})
            continue
        todo.append(p)
    return todo


_InitArgs = tuple[str | None, str, str, tuple[str, str] | None]


def _results(jobs: list[tuple[str, str, str]], workers: int, init_args: _InitArgs) -> Iterable[str]:
    """One JSON line per job, in input order; an empty string marks a failure."""
    if workers and workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_init_worker, initargs=init_args) as pool:
            # imap keeps input order, so the output file is deterministic
            yield from pool.imap(_run_one, jobs, chunksize=1)
    else:
        _init_worker(*init_args)
        for job in jobs:
            yield _run_one(job)


@dataclass
class _Store:
    """An open SQLite store and the run it records."""

    conn: sqlite3.Connection
    run_id: str
    bbox_dir: Path

    def mirror(self, line: str) -> None:
        """Upsert one committed results line. A store failure is logged, never raised."""
        record = DocumentRecord.model_validate_json(line)
        try:
            db.upsert_document(self.conn, record, self.run_id, self.bbox_dir / f"{record.doc_id}.json.gz")
        except sqlite3.Error as exc:
            log.error(
                "db write failed, results line kept; run db-import to repair",
                extra={"doc_id": record.doc_id, "error": summarize_exception(exc)},
            )


def _start_store(conn: sqlite3.Connection, cfg: Config, bbox_dir: Path, cabinet_id: str, root: Path) -> _Store:
    """Record the run and the model manifest, and wrap the connection."""
    run_id = db.start_run(conn, cfg.dump(), cabinet_id)
    model_dir = root / str(cfg.classify_opts().get("model_dir", "models/classifier"))
    db.record_models(conn, classify_mod.read_manifest(model_dir.parent / classify_mod.MANIFEST))
    return _Store(conn, run_id, bbox_dir)


@contextmanager
def _open_store(cfg: Config, out_path: Path, bbox_dir: Path, cabinet_id: str, root: Path) -> Iterator[_Store | None]:
    """The run's store, or None when db.path is null or the store cannot be opened."""
    path = cfg.db_path(out_path)
    if path is None:
        yield None
        return
    try:
        conn = db.connect(path)
    except sqlite3.Error as exc:
        log.error("db unavailable, writing JSONL only", extra={"db": str(path), "error": summarize_exception(exc)})
        yield None
        return
    with closing(conn):
        try:
            store: _Store | None = _start_store(conn, cfg, bbox_dir, cabinet_id, root)
        except sqlite3.Error as exc:
            log.error("db unavailable, writing JSONL only", extra={"db": str(path), "error": summarize_exception(exc)})
            store = None
        yield store


def _write_lines(fh: IO[str], lines: Iterable[str], stats: BatchStats, store: _Store | None = None) -> None:
    for line in lines:
        if not line:
            stats["failed"] += 1
            continue
        fh.write(line + "\n")
        stats["written"] += 1
        if store is not None:
            # The line must be on disk before the store may hold it.
            fh.flush()
            store.mirror(line)


def _start_stats(found: int, todo: int, excluded: dict[str, int], workers: int) -> BatchStats:
    """Fresh batch counts, logged as the batch starts."""
    stats: BatchStats = {"found": found, "skipped": found - todo, "written": 0, "failed": 0, "excluded": excluded}
    log.info(
        "batch starting",
        extra={
            "found": found,
            "skipped": found - todo,
            "todo": todo,
            "excluded": sum(excluded.values()),
            "workers": workers,
        },
    )
    return stats


def _scan_logged(input_dir: str | Path, cfg: Config) -> tuple[list[Path], dict[str, int]]:
    """Scan for documents and log what was excluded, with counts per reason. A pack run reads its manifest."""
    if packmode.is_active(cfg):
        return packmode.inputs(cfg, input_dir), {}
    paths, excluded = scan_documents(input_dir)
    excluded_counts = {reason: len(items) for reason, items in sorted(excluded.items())}
    if excluded:
        # Say what was left out and why. A total that quietly omits input does
        # not mean what it appears to mean.
        log.info("files excluded as not documents", extra=excluded_counts)
    return paths, excluded_counts


def run_batch(
    input_dir: str | Path,
    output: str | Path,
    config_dir: str | None = None,
    workers: int = 1,
    bbox_dir: str | Path | None = None,
    resume: bool = True,
    root: str | Path = ".",
    cabinet_id: str | None = None,
    client_pack: tuple[str, str] | None = None,
) -> BatchStats:
    """Process a directory into JSONL. Resumable, parallel, order stable.

    Args:
        input_dir: Directory to scan for documents.
        output: Results JSONL; appended to on resume.
        config_dir: Config directory, None for the default.
        workers: Process count; 1 runs in this process.
        bbox_dir: Sidecar directory, defaults to `<output>.bboxes`.
        resume: Skip documents already in the output.
        root: Directory model paths are relative to.
        cabinet_id: Cabinet stamped on every record; None uses the config's. Ignored in pack mode.
        client_pack: (pack directory, cabinet) for a pack-mode run; None falls back to pipeline.yaml client_pack.

    Returns:
        Batch counts.
    """
    cfg = packmode.resolve(load_config_cached(config_dir), client_pack)
    log_level = str(cfg.pipeline.get("runtime", {}).get("log_level", "INFO"))
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bdir = Path(bbox_dir) if bbox_dir else out_path.with_suffix(out_path.suffix + ".bboxes")

    paths, excluded_counts = _scan_logged(input_dir, cfg)
    todo = _pending(paths, out_path, resume)
    stats = _start_stats(len(paths), len(todo), excluded_counts, workers)
    if not todo:
        return stats

    cabinet = cfg.cabinet_id if packmode.is_active(cfg) else cabinet_id or cfg.cabinet_id
    jobs = [(str(p), str(bdir), cabinet) for p in todo]
    mode = "a" if out_path.exists() and resume else "w"
    with _open_store(cfg, out_path, bdir, cabinet, Path(root)) as store:
        with open(out_path, mode, encoding="utf-8", newline="\n") as fh:
            _write_lines(fh, _results(jobs, workers, (config_dir, str(root), log_level, client_pack)), stats, store)

    log.info("batch done", extra=stats)
    return stats

"""Corpus characterisation. Measurement only, no models.

Run this first on real data. It tells you where the floor actually is before
you spend anything on training.
"""

from __future__ import annotations

import collections
import statistics
from pathlib import Path

from .classify import classify
from .config import Config, load_config_cached
from .extract import extract, scan_documents
from .logging_setup import get_logger
from .metadata import _pattern_candidates
from .schema import ExtractionMethod, build_canonical_text, compute_doc_id

log = get_logger(__name__)


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def _dist(values: list[int]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    return {
        "min": s[0],
        "p50": s[len(s) // 2],
        "p90": s[min(len(s) - 1, int(0.9 * len(s)))],
        "max": s[-1],
        "mean": round(statistics.fmean(s), 2),
    }


def profile_corpus(input_dir: str | Path, cfg: Config) -> dict:
    """Walk the corpus and count things. Nothing here needs a trained model."""
    paths, excluded = scan_documents(input_dir)
    stats: dict = {
        "n_documents": len(paths),
        # What the corpus holds that is not a document, so the count above can
        # be checked against the file count on disk rather than trusted.
        "excluded": {reason: len(items) for reason, items in sorted(excluded.items())},
        "n_files_seen": len(paths) + sum(len(v) for v in excluded.values()),
        "by_extension": collections.Counter(),
        "methods": collections.Counter(),
        "page_methods": collections.Counter(),
        "with_native_text": 0,
        "scanned": 0,
        "empty": 0,
        "page_counts": [],
        "char_counts": [],
        "extraction_confidence": [],
        "rule_labels": collections.Counter(),
        "rule_methods": collections.Counter(),
        "field_hits": collections.Counter(),
        "field_validated": collections.Counter(),
        "errors": collections.Counter(),
        "unreadable": [],
        # OCR quality proxy inputs
        "conf_by_method": collections.defaultdict(list),
        "hybrid_deltas": [],
    }

    pattern_fields = [f for f in cfg.field_defs if "pattern" in f.get("strategies", [])]
    limit = int(cfg.metadata_opts().get("max_candidates_per_field", 25))

    for p in paths:
        stats["by_extension"][p.suffix.lower()] += 1
        try:
            compute_doc_id(p)
        except OSError as exc:
            stats["unreadable"].append(f"{p.name}: {exc}")
            continue

        res = extract(p, cfg)
        for e in res.errors:
            stats["errors"][e.split(":")[0]] += 1

        full = build_canonical_text(res.pages)
        stats["methods"][res.method.value] += 1
        for pm in res.page_methods:
            stats["page_methods"][pm.value] += 1
        stats["page_counts"].append(res.page_count)
        stats["char_counts"].append(len(full))
        stats["extraction_confidence"].append(res.confidence())
        for pm, pc in zip(res.page_methods, res.page_confidences):
            stats["conf_by_method"][pm.value].append(pc)
        if res.method == ExtractionMethod.HYBRID:
            # Same document, both routes. This is the cleanest comparison there is.
            nat = [c for m, c in zip(res.page_methods, res.page_confidences)
                   if m == ExtractionMethod.NATIVE]
            ocr = [c for m, c in zip(res.page_methods, res.page_confidences)
                   if m == ExtractionMethod.OCR]
            if nat and ocr:
                stats["hybrid_deltas"].append(
                    (p.name, statistics.fmean(nat), statistics.fmean(ocr))
                )
        if res.has_native_text:
            stats["with_native_text"] += 1
        if res.is_scanned:
            stats["scanned"] += 1
        if not full.strip():
            stats["empty"] += 1

        # Rule layer only, so this is the no-model classification floor.
        c = classify(full, res.pages, cfg, model=None)
        stats["rule_labels"][c.label] += 1
        stats["rule_methods"][c.method.value] += 1

        # Pattern layer only, per field, so anchors do not flatter the numbers.
        for fdef in pattern_fields:
            cands = _pattern_candidates(full, fdef, limit)
            if cands:
                stats["field_hits"][fdef["name"]] += 1
                if any(c2.validated for c2 in cands):
                    stats["field_validated"][fdef["name"]] += 1

    return stats


def render_markdown(stats: dict, input_dir: str, cfg: Config) -> str:
    n = stats["n_documents"]
    L: list[str] = []
    a = L.append

    a("# Corpus profile")
    a("")
    a(f"Source: `{input_dir}`")
    a(f"Documents found: **{n}**")
    excluded = stats.get("excluded") or {}
    if excluded:
        seen = stats.get("n_files_seen", n)
        a("")
        a(f"`{input_dir}` holds {seen} files. {seen - n} of them are not documents "
          "and were excluded before anything ran:")
        a("")
        a("| reason | files |")
        a("| --- | --- |")
        for reason, count in excluded.items():
            a(f"| `{reason}` | {count} |")
        a("")
        a("See `EXCLUDED_EXT` and `_exclusion_reason` in `extract.py` for what each "
          "reason means and why it was decided that way.")
    a("")
    a("No models were used. Everything below is the deterministic floor.")
    a("")

    a("## Extraction")
    a("")
    a("| metric | value |")
    a("| --- | --- |")
    a(f"| has native text | {stats['with_native_text']} ({_pct(stats['with_native_text'], n)}) |")
    a(f"| fully scanned | {stats['scanned']} ({_pct(stats['scanned'], n)}) |")
    a(f"| no text at all | {stats['empty']} ({_pct(stats['empty'], n)}) |")
    a("")
    a("Document level routing:")
    a("")
    a("| method | docs |")
    a("| --- | --- |")
    for k, v in sorted(stats["methods"].items()):
        a(f"| {k} | {v} |")
    a("")
    a("Page level routing:")
    a("")
    a("| page method | pages |")
    a("| --- | --- |")
    for k, v in sorted(stats["page_methods"].items()):
        a(f"| {k} | {v} |")
    a("")

    a("## Size distribution")
    a("")
    pc = _dist(stats["page_counts"])
    cc = _dist(stats["char_counts"])
    a("| metric | min | p50 | p90 | max | mean |")
    a("| --- | --- | --- | --- | --- | --- |")
    if pc:
        a(f"| pages | {pc['min']} | {pc['p50']} | {pc['p90']} | {pc['max']} | {pc['mean']} |")
    if cc:
        a(f"| chars | {cc['min']} | {cc['p50']} | {cc['p90']} | {cc['max']} | {cc['mean']} |")
    a("")

    conf = stats["extraction_confidence"]
    if conf:
        a(f"Mean extraction confidence: **{statistics.fmean(conf):.3f}**")
        a("")
        low = sum(1 for c in conf if c < 0.6)
        a(f"Documents below 0.6 confidence: {low} ({_pct(low, n)})")
        a("")

    a("## OCR quality proxy")
    a("")
    by_method = stats["conf_by_method"]
    if by_method:
        a("Mean page confidence by route:")
        a("")
        a("| route | pages | mean confidence |")
        a("| --- | --- | --- |")
        for k in sorted(by_method):
            vals = by_method[k]
            a(f"| {k} | {len(vals)} | {statistics.fmean(vals):.3f} |")
        a("")

    deltas = stats["hybrid_deltas"]
    if deltas:
        a("Documents carrying both routes, so the comparison is like for like:")
        a("")
        a("| document | native | ocr | drop |")
        a("| --- | --- | --- | --- |")
        for name, nat, ocr in deltas:
            a(f"| {name} | {nat:.3f} | {ocr:.3f} | {nat - ocr:.3f} |")
        a("")
        mean_drop = statistics.fmean(nat - ocr for _, nat, ocr in deltas)
        a(f"Mean confidence drop when a page is OCR'd rather than read: **{mean_drop:.3f}**")
        a("")
    else:
        a("No document produced both a native and an OCR page, so there is no")
        a("like for like comparison. The table above is still indicative.")
        a("")

    a("## Classification, rule layer only")
    a("")
    a("| label | docs | share |")
    a("| --- | --- | --- |")
    for k, v in sorted(stats["rule_labels"].items(), key=lambda kv: (-kv[1], kv[0])):
        a(f"| {k} | {v} | {_pct(v, n)} |")
    a("")
    unknown = stats["rule_labels"].get(str(cfg.classify_opts().get("unknown_label", "unknown")), 0)
    a(f"Rule layer coverage: **{_pct(n - unknown, n)}**. The rest need the model.")
    a("")

    a("## Metadata, pattern layer only")
    a("")
    a("Anchors and templates are excluded, so this is the regex plus checksum floor.")
    a("")
    a("| field | docs with a hit | hit rate | validated | validated rate |")
    a("| --- | --- | --- | --- | --- |")
    names = sorted({f["name"] for f in cfg.field_defs if "pattern" in f.get("strategies", [])})
    for name in names:
        h = stats["field_hits"].get(name, 0)
        v = stats["field_validated"].get(name, 0)
        a(f"| {name} | {h} | {_pct(h, n)} | {v} | {_pct(v, h)} |")
    a("")

    if stats["errors"]:
        a("## Errors")
        a("")
        a("| stage | count |")
        a("| --- | --- |")
        for k, v in sorted(stats["errors"].items()):
            a(f"| {k} | {v} |")
        a("")
    if stats["unreadable"]:
        a("Unreadable files:")
        a("")
        for u in stats["unreadable"][:20]:
            a(f"- {u}")
        a("")

    a("## Extensions")
    a("")
    a("| ext | count |")
    a("| --- | --- |")
    for k, v in sorted(stats["by_extension"].items()):
        a(f"| {k} | {v} |")
    a("")
    return "\n".join(L)


def run_profile(input_dir: str | Path, output: str | Path, config_dir: str | None = None) -> dict:
    cfg = load_config_cached(config_dir)
    stats = profile_corpus(input_dir, cfg)
    md = render_markdown(stats, str(input_dir), cfg)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8", newline="\n")
    log.info("profile written", extra={"output": str(out), "documents": stats["n_documents"]})
    return stats

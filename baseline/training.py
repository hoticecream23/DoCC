"""Training the classifier and tagger from annotations, and fitting tag thresholds.

The logic behind `baseline train` and `baseline tune-thresholds`; cli.py only
parses arguments and calls in here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import classify as classify_mod
from . import tagging as tagging_mod
from .config import Config, save_yaml
from .extract import extract
from .logging_setup import get_logger
from .schema import build_canonical_text, read_jsonl

log = get_logger(__name__)

SORT_KEY_CHARS = 200  # annotation text prefix used as a sort key when a row has no id


def _annotation_sort_key(row: dict[str, Any]) -> str:
    """Stable order for annotation rows: doc_id, else path, else a text prefix."""
    return str(row.get("doc_id") or row.get("path") or row.get("text", ""))[:SORT_KEY_CHARS]


def read_annotations(path: str | Path) -> list[dict[str, Any]]:
    """Read annotation rows, sorted so training is reproducible.

    Args:
        path: JSONL of {path|text, label?, tags?}.

    Returns:
        The rows.
    """
    rows = read_jsonl(path)
    rows.sort(key=_annotation_sort_key)
    return rows


def _text_for(row: dict[str, Any], cfg: Config) -> str:
    """Annotation may carry text inline, otherwise we extract it."""
    if row.get("text"):
        return str(row["text"])
    path = row.get("path")
    if not path:
        raise ValueError("annotation row needs either text or path")
    return build_canonical_text(extract(path, cfg).pages)


def train_models(rows: list[dict[str, Any]], cfg: Config, out: Path, kind: str) -> dict[str, Any]:
    """Fit and save the classifier, the tagger, or both.

    Args:
        rows: Annotation rows.
        cfg: Loaded configuration.
        out: Model directory; models go in its classifier/ and tagger/.
        kind: classifier, tagger or both.

    Returns:
        Training report per model trained; empty when nothing was trained.

    Raises:
        ValueError: A row carries neither text nor path.
    """
    texts = [_text_for(r, cfg) for r in rows]
    report: dict[str, Any] = {}

    labelled = [(t, r["label"]) for t, r in zip(texts, rows, strict=True) if r.get("label")]
    if labelled and kind in ("classifier", "both"):
        report["classifier"] = classify_mod.train(
            [t for t, _ in labelled], [label for _, label in labelled], out / "classifier"
        )
    elif kind in ("classifier", "both"):
        log.warning("no label field in annotations, skipping classifier")

    tagged = [(t, r["tags"]) for t, r in zip(texts, rows, strict=True) if r.get("tags")]
    if tagged and kind in ("tagger", "both"):
        report["tagger"] = tagging_mod.train([t for t, _ in tagged], [list(g) for _, g in tagged], out / "tagger")
    elif kind in ("tagger", "both"):
        log.warning("no tags field in annotations, skipping tagger")
    return report


# --------------------------------------------------------------------------
# threshold tuning
# --------------------------------------------------------------------------


def _truth_tags(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    truth: dict[str, list[str]] = {}
    for row in rows:
        key = row.get("doc_id") or row.get("path")
        if key and row.get("tags") is not None:
            truth[str(key)] = list(row["tags"])
    return truth


def _prediction_texts(predictions: str | Path) -> dict[str, str]:
    """Canonical text per doc_id and source_path, skipping malformed lines."""
    texts: dict[str, str] = {}
    skipped = 0
    with open(predictions, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            body = rec.get("text", {}).get("full", "")
            for key in (rec.get("doc_id"), rec.get("source_path")):
                if key:
                    texts[str(key)] = body
    if skipped:
        log.warning(
            f"skipped {skipped} malformed prediction lines",
            extra={"predictions": str(predictions), "skipped": skipped},
        )
    return texts


def _score_tags(
    model: tagging_mod.TaggerModel, truth: dict[str, list[str]], texts: dict[str, str], inline: dict[str, str]
) -> tuple[dict[str, list[int]], dict[str, list[float]], int]:
    """Truth and model score per tag, over every annotation with a prediction or inline text."""
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
    return y_true, y_prob, matched


def fit_tag_thresholds(
    model_dir: str | Path, annotations: str | Path, predictions: str | Path
) -> dict[str, dict[str, float | int]] | None:
    """Re-score predictions with the tagger and fit F1 optimal cutoffs.

    Args:
        model_dir: Tagger model directory.
        annotations: Annotation JSONL with tags.
        predictions: Results JSONL supplying each document's text.

    Returns:
        Tag to {threshold, f1, positives}; None when nothing could be fitted
        (the reason is logged).
    """
    model = tagging_mod.load_model(model_dir)
    if model is None:
        log.error("no tagger model, train one first", extra={"dir": str(model_dir)})
        return None
    rows = read_annotations(annotations)
    truth = _truth_tags(rows)
    if not truth:
        log.error("annotations carry no tags")
        return None
    # Annotations that carry their own text can be tuned without a join.
    inline = {
        str(r.get("doc_id") or r.get("path")): r["text"]
        for r in rows
        if r.get("text") and (r.get("doc_id") or r.get("path"))
    }
    texts = _prediction_texts(predictions)
    y_true, y_prob, matched = _score_tags(model, truth, texts, inline)
    if not matched:
        log.error(
            "no annotation matched a prediction and none carried inline text, " "check that doc_id or path keys line up"
        )
        return None
    tuned = tagging_mod.tune_thresholds(y_true, y_prob)
    log.info("thresholds fitted", extra={"matched": matched, "tags": len(tuned)})
    for tag, info in sorted(tuned.items()):
        log.info(
            "tag threshold",
            extra={"tag": tag, "threshold": info["threshold"], "f1": info["f1"], "positives": info["positives"]},
        )
    return tuned


def save_tag_thresholds(cfg: Config, tuned: dict[str, dict[str, float | int]]) -> Path:
    """Write fitted thresholds back into the config's tags.yaml.

    Args:
        cfg: Loaded configuration.
        tuned: Output of fit_tag_thresholds.

    Returns:
        The tags.yaml path written.
    """
    tags_doc = dict(cfg.tags)
    for entry in tags_doc.get("tags", []):
        info = tuned.get(entry.get("name"))
        if info:
            entry["threshold"] = info["threshold"]
    path = Path(cfg.config_dir) / "tags.yaml"
    save_yaml(path, tags_doc)
    return path

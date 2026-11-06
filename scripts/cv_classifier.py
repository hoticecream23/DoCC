"""Cross-validate the classifier on stored client text, against the current model.

Reads canonical text from a results JSONL (text.full, text.pages, source_path).
The label is the folder name in source_path, mapped through config/taxonomy.yaml.
Each fold fits the same pipeline as baseline.classify.train (build_estimator)
on the training docs and scores the held-out docs two ways: model alone
(top probability >= model_min) and the full rule+model classify(). The current
model needs no fitting; it is scored on the same held-out docs. It is read
from --baseline-models (default models/previous), never from models/, because
models/ holds the model fitted on these same docs.

A doc_id filed under two folders is kept once (first by source_path), so no
copy is trained on while its twin is tested.

    python scripts/cv_classifier.py results_classification.jsonl OUT_DIR [--baseline-models DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Literal, TypedDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline import classify  # noqa: E402
from baseline.config import Config, load_config_cached  # noqa: E402
from baseline.io import atomic_open, atomic_write_text, report_dry_run  # noqa: E402
from baseline.logging_setup import setup_logging  # noqa: E402

PredKey = Literal["current_model", "new_model", "current_full", "new_full"]
MetricKey = Literal["coverage", "min_precision", "macro_f1"]
PRED_KEYS: tuple[PredKey, ...] = ("current_model", "new_model", "current_full", "new_full")
METRICS: tuple[MetricKey, ...] = ("coverage", "min_precision", "macro_f1")
DEFAULT_FOLDS = 5
COLUMN_WIDTH = 24


class Doc(TypedDict):
    """One labelled document."""

    doc_id: str | None
    full: str
    pages: list[str]
    gold: str


class PredRow(TypedDict):
    """Predictions for one held-out document."""

    fold: int
    doc_id: str | None
    gold: str
    current_model: str
    current_full: str
    new_model: str
    new_full: str


class Score(TypedDict):
    """Coverage, per class precision among answered, and macro F1."""

    n: int
    coverage: float
    precision: dict[str, tuple[int, int]]
    min_precision: float
    macro_f1: float


class Report(TypedDict):
    """cv_report.json."""

    n_docs: int
    class_counts: dict[str, int]
    dropped: dict[str, int]
    duplicate_doc_ids_dropped: list[str]
    current_model_dir: str
    new_model: str
    folds: int
    seed: int
    per_fold: list[dict[PredKey, Score]]
    pooled: dict[PredKey, Score]


def load_docs(
    path: str | Path, label_map: dict[str, str | None]
) -> tuple[list[Doc], collections.Counter[str], list[str]]:
    """Read labelled documents from a results JSONL.

    Args:
        path: Results JSONL with text and source_path.
        label_map: Client folder name to class; None or missing drops the document.

    Returns:
        The documents, dropped counts per folder, and source paths of duplicate doc_ids.
    """
    docs: list[Doc] = []
    dropped: collections.Counter[str] = collections.Counter()
    seen: set[str | None] = set()
    dupes: list[str] = []
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    for row in sorted(rows, key=lambda r: r["source_path"]):
        if row.get("doc_id") in seen:
            dupes.append(row["source_path"])
            continue
        seen.add(row.get("doc_id"))
        folder = row["source_path"].replace("\\", "/").split("/")[-2]
        cls = label_map.get(folder)
        if not cls:
            dropped[folder] += 1
            continue
        text = row["text"]
        pages = [p if isinstance(p, str) else p.get("text", "") for p in text.get("pages", [])]
        docs.append({"doc_id": row.get("doc_id"), "full": text["full"], "pages": pages, "gold": cls})
    return docs, dropped, dupes


def model_alone(model: classify.ClassifierModel | None, full: str, model_min: float, unknown: str) -> str:
    """The model's top label if it reaches model_min, else unknown.

    Args:
        model: The classifier; None answers unknown.
        full: Document text.
        model_min: Minimum top probability to answer.
        unknown: The unknown label.

    Returns:
        A class label or unknown.
    """
    if model is None or not full.strip():
        return unknown
    label, prob = sorted(model.predict_scores(full).items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return label if prob >= model_min else unknown


def score(pairs: list[tuple[str, str]], unknown: str) -> Score:
    """Coverage, per-class precision among answered, macro F1 over gold classes.

    Args:
        pairs: (gold, predicted) per document.
        unknown: The label that means no answer.

    Returns:
        The scores.
    """
    answered = [(g, p) for g, p in pairs if p != unknown]
    prec = {}
    for c in sorted({p for _, p in answered}):
        hits = [g == c for g, p in answered if p == c]
        prec[c] = (sum(hits), len(hits))
    f1s = []
    for c in sorted({g for g, _ in pairs}):
        tp = sum(1 for g, p in pairs if g == c and p == c)
        npred = sum(1 for _, p in pairs if p == c)
        ngold = sum(1 for g, _ in pairs if g == c)
        precision = tp / npred if npred else 0.0
        recall = tp / ngold if ngold else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "n": len(pairs),
        "coverage": len(answered) / len(pairs) if pairs else 0.0,
        "precision": prec,
        "min_precision": min((a / b for a, b in prec.values()), default=1.0),
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }


def _predict(
    docs: list[Doc],
    current: classify.ClassifierModel,
    cfg: Config,
    folds: int,
    seed: int,
) -> list[PredRow]:
    """Fit a model per fold and predict each held-out document with both models."""
    from sklearn.model_selection import StratifiedKFold

    opts = cfg.classify_opts()
    unknown = str(opts.get("unknown_label", "unknown"))
    model_min = float(opts.get("model_min_confidence", 0.45))
    golds = [d["gold"] for d in docs]
    preds: list[PredRow] = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(docs, golds)):
        train_labels = [golds[i] for i in train_idx]
        pipe = classify.build_estimator(min(collections.Counter(train_labels).values()))
        pipe.fit([docs[i]["full"] for i in train_idx], train_labels)
        new = classify.ClassifierModel(pipeline=pipe, labels=[str(c) for c in pipe.classes_])
        for i in test_idx:
            doc = docs[i]
            preds.append(
                {
                    "fold": fold,
                    "doc_id": doc["doc_id"],
                    "gold": doc["gold"],
                    "current_model": model_alone(current, doc["full"], model_min, unknown),
                    "current_full": classify.classify(doc["full"], doc["pages"], cfg, current).label,
                    "new_model": model_alone(new, doc["full"], model_min, unknown),
                    "new_full": classify.classify(doc["full"], doc["pages"], cfg, new).label,
                }
            )
    return preds


def _scores(rows: list[PredRow], unknown: str) -> dict[PredKey, Score]:
    """Score every prediction column over the given rows."""
    return {key: score([(r["gold"], r[key]) for r in rows], unknown) for key in PRED_KEYS}


def _metrics_cell(values: list[float]) -> str:
    """Coverage / min class precision / macro F1 as one right aligned column."""
    return "/".join(f"{v:.3f}" for v in values).rjust(COLUMN_WIDTH)


def _mean_metrics(per_fold: list[dict[PredKey, Score]], key: PredKey) -> list[float]:
    """Each metric for one prediction column, averaged over the folds."""
    return [sum(fold[key][m] for fold in per_fold) / len(per_fold) for m in METRICS]


def _print_report(report: Report, base_dir: Path) -> None:
    print("docs", report["n_docs"], "classes", report["class_counts"], "dropped", report["dropped"])
    print("duplicate doc_id copies dropped:", report["duplicate_doc_ids_dropped"])
    print("current_* columns use", base_dir, "| new_* columns use a model fitted per fold")
    print("fold  " + "  ".join(f"{k:>{COLUMN_WIDTH}}" for k in PRED_KEYS) + "   (coverage / min class P / macro F1)")
    for k, fold in enumerate(report["per_fold"]):
        print(f"{k:>4}  " + "  ".join(_metrics_cell([fold[x][m] for m in METRICS]) for x in PRED_KEYS))
    print("mean  " + "  ".join(_metrics_cell(_mean_metrics(report["per_fold"], x)) for x in PRED_KEYS))
    print("pool  " + "  ".join(_metrics_cell([report["pooled"][x][m] for m in METRICS]) for x in PRED_KEYS))
    for x in PRED_KEYS:
        print(x, {c: f"{a}/{b}" for c, (a, b) in report["pooled"][x]["precision"].items()})


def main(
    results: str | Path, out_dir: str | Path, baseline_models: str | Path, folds: int = DEFAULT_FOLDS, seed: int = 0
) -> Report:
    """Cross-validate, write cv_preds.jsonl and cv_report.json, and print a summary.

    Args:
        results: Results JSONL with stored text.
        out_dir: Directory for the outputs.
        baseline_models: Models directory holding the current classifier.
        folds: Number of folds.
        seed: Shuffle seed.

    Returns:
        The report written to cv_report.json.

    Raises:
        SystemExit: No classifier in baseline_models.
    """
    cfg = load_config_cached(ROOT / "config")
    unknown = str(cfg.classify_opts().get("unknown_label", "unknown"))
    docs, dropped, dupes = load_docs(results, cfg.client_label_map)
    base_dir = Path(baseline_models) / "classifier"
    current = classify.load_model(base_dir)
    if current is None:
        raise SystemExit(f"no baseline model in {base_dir}")
    preds = _predict(docs, current, cfg, folds, seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with atomic_open(out / "cv_preds.jsonl") as fh:
        for row in preds:
            fh.write(json.dumps(row) + "\n")

    report: Report = {
        "n_docs": len(docs),
        "class_counts": dict(collections.Counter(d["gold"] for d in docs)),
        "dropped": dict(dropped),
        "duplicate_doc_ids_dropped": dupes,
        "current_model_dir": str(base_dir),
        "new_model": "fitted per fold",
        "folds": folds,
        "seed": seed,
        "per_fold": [_scores([r for r in preds if r["fold"] == k], unknown) for k in range(folds)],
        "pooled": _scores(preds, unknown),
    }
    atomic_write_text(out / "cv_report.json", json.dumps(report, indent=2))
    _print_report(report, base_dir)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("out_dir")
    parser.add_argument("--baseline-models", default=str(ROOT / "models" / "previous"))
    parser.add_argument("--dry-run", action="store_true", help="log what would be written and stop")
    args = parser.parse_args()
    setup_logging()
    if args.dry_run:
        with open(args.results, encoding="utf-8") as fh:
            n_docs = sum(1 for line in fh if line.strip())
        report_dry_run([Path(args.out_dir) / "cv_preds.jsonl", Path(args.out_dir) / "cv_report.json"], n_docs)
    else:
        main(args.results, args.out_dir, args.baseline_models)

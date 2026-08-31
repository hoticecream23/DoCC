"""Document classification: rule layer first, then a calibrated linear model.

Train and predict are separate entry points. Nothing here knows the name of
any document class, that all lives in classes.yaml.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .logging_setup import get_logger
from .rules import best_marker_score
from .schema import Classification, ClassifyMethod

log = get_logger(__name__)

MODEL_FILE = "model.joblib"
META_FILE = "meta.json"


def rule_scores(full: str, pages: list[str], cfg: Config) -> dict[str, float]:
    """Best firing marker specificity per class. Independent, not a softmax."""
    header = int(cfg.classify_opts().get("header_chars", 600))
    out: dict[str, float] = {}
    for cls in cfg.classes.get("classes", []):
        score = best_marker_score(cls.get("markers", []), full, pages, header)
        if score > 0:
            out[cls["name"]] = round(score, 4)
    return out


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def build_estimator(min_class_count: int):
    """Word ngrams plus char ngrams, calibrated into real probabilities.

    Char ngrams carry the load when OCR noise breaks word tokens.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=1,
                    sublinear_tf=True, lowercase=True, strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                    sublinear_tf=True, lowercase=True, strip_accents="unicode",
                ),
            ),
        ]
    )
    folds = max(2, min(5, min_class_count))
    # shuffle off so a rerun on the same data gives the same model
    cv = StratifiedKFold(n_splits=folds, shuffle=False)
    clf = CalibratedClassifierCV(
        LinearSVC(random_state=0, dual="auto"), method="sigmoid", cv=cv
    )
    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class ClassifierModel:
    pipeline: object
    labels: list[str]

    def predict_scores(self, text: str) -> dict[str, float]:
        probs = self.pipeline.predict_proba([text])[0]
        return {lab: round(float(p), 4) for lab, p in zip(self.labels, probs)}


def train(texts: list[str], labels: list[str], out_dir: str | Path) -> dict:
    """Fit and persist. Returns a small training report."""
    import collections

    import joblib

    if len(texts) != len(labels):
        raise ValueError("texts and labels differ in length")
    counts = collections.Counter(labels)
    if len(counts) < 2:
        raise ValueError("need at least 2 distinct classes, got " + str(list(counts)))
    smallest = min(counts.values())
    if smallest < 2:
        rare = [k for k, v in counts.items() if v < 2]
        raise ValueError(
            "every class needs at least 2 examples for calibration, thin: " + str(rare)
        )

    pipe = build_estimator(smallest)
    pipe.fit(texts, labels)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / MODEL_FILE)
    meta = {
        "kind": "classifier",
        "labels": sorted(counts),
        "class_counts": dict(sorted(counts.items())),
        "n_samples": len(texts),
        "cv_folds": max(2, min(5, smallest)),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    log.info("classifier trained", extra=meta)
    return meta


def load_model(model_dir: str | Path) -> ClassifierModel | None:
    """None when no model has been trained yet. The rule layer still works."""
    d = Path(model_dir)
    if not (d / MODEL_FILE).exists():
        return None
    try:
        import joblib

        pipe = joblib.load(d / MODEL_FILE)
        # sklearn hands back numpy strings, force plain str for clean JSON keys
        return ClassifierModel(pipeline=pipe, labels=[str(c) for c in pipe.classes_])
    except Exception as exc:
        log.error("classifier load failed", extra={"dir": str(d), "error": str(exc)})
        return None


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


def classify(
    full: str, pages: list[str], cfg: Config, model: ClassifierModel | None = None
) -> Classification:
    """Cascade: strong rule, then model, then weak rule, then unknown.

    Never forces a guess. Coverage at a fixed error rate is the metric.
    """
    opts = cfg.classify_opts()
    unknown = str(opts.get("unknown_label", "unknown"))
    short_circuit = float(opts.get("rule_short_circuit", 0.9))
    rule_min = float(opts.get("rule_min_confidence", 0.7))
    model_min = float(opts.get("model_min_confidence", 0.45))

    rules = rule_scores(full, pages, cfg)
    best_rule_label, best_rule_score = "", 0.0
    if rules:
        # ties break on name so the output is stable
        best_rule_label, best_rule_score = sorted(rules.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    if best_rule_score >= short_circuit:
        return Classification(
            label=best_rule_label,
            confidence=round(best_rule_score, 4),
            all_scores=dict(sorted(rules.items())),
            method=ClassifyMethod.RULE,
        )

    if model is not None and full.strip():
        try:
            scores = model.predict_scores(full)
            label, prob = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if prob >= model_min:
                return Classification(
                    label=label,
                    confidence=round(prob, 4),
                    all_scores=dict(sorted(scores.items())),
                    method=ClassifyMethod.TFIDF_SVM,
                )
            # Model spoke but not loudly enough. Keep its distribution.
            return Classification(
                label=unknown,
                confidence=round(prob, 4),
                all_scores=dict(sorted(scores.items())),
                method=ClassifyMethod.FALLBACK,
            )
        except Exception as exc:
            log.error("classifier predict failed", extra={"error": str(exc)})

    if best_rule_score >= rule_min:
        return Classification(
            label=best_rule_label,
            confidence=round(best_rule_score, 4),
            all_scores=dict(sorted(rules.items())),
            method=ClassifyMethod.RULE,
        )

    return Classification(
        label=unknown,
        confidence=round(best_rule_score, 4),
        all_scores=dict(sorted(rules.items())),
        method=ClassifyMethod.FALLBACK,
    )

"""Multi label tagging. Independent of classification.

A document has one class but many tags, so nothing here is a softmax.
Vocabulary, patterns, thresholds and derived rules all live in tags.yaml.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .logging_setup import get_logger
from .rules import best_marker_score
from .schema import MetadataField, Tag, TagMethod

log = get_logger(__name__)

MODEL_FILE = "model.joblib"
META_FILE = "meta.json"

# Higher wins when two sources claim the same tag at equal confidence.
_PRIORITY = {TagMethod.DERIVED: 3, TagMethod.TFIDF_SVM: 2, TagMethod.KEYWORD: 1}


def tag_thresholds(cfg: Config) -> dict[str, float]:
    default = float(cfg.tagging_opts().get("default_threshold", 0.5))
    return {t["name"]: float(t.get("threshold", default)) for t in cfg.tag_defs}


def keyword_scores(full: str, pages: list[str], cfg: Config) -> dict[str, float]:
    header = int(cfg.classify_opts().get("header_chars", 600))
    out: dict[str, float] = {}
    for t in cfg.tag_defs:
        score = best_marker_score(t.get("patterns", []), full, pages, header)
        if score > 0:
            out[t["name"]] = round(score, 4)
    return out


# --------------------------------------------------------------------------
# derived tags, read off the metadata output and document level facts
# --------------------------------------------------------------------------


def _eval_condition(cond: dict, by_field: dict[str, list[MetadataField]], facts: dict) -> bool:
    if "doc" in cond:
        val = facts.get(cond["doc"])
        if val is None:
            return False
        for op, key in (("gt", "gt"), ("lt", "lt"), ("gte", "gte"), ("lte", "lte")):
            if key in cond:
                try:
                    a, b = float(val), float(cond[key])
                except (TypeError, ValueError):
                    return False
                if op == "gt" and not a > b:
                    return False
                if op == "lt" and not a < b:
                    return False
                if op == "gte" and not a >= b:
                    return False
                if op == "lte" and not a <= b:
                    return False
        if "eq" in cond and val != cond["eq"]:
            return False
        return True

    name = cond.get("field")
    if not name:
        return False
    hits = by_field.get(name, [])
    if not hits:
        return False
    if cond.get("validated") and not any(h.validated for h in hits):
        return False
    if "min_confidence" in cond:
        floor = float(cond["min_confidence"])
        if not any(h.confidence >= floor for h in hits):
            return False
    if "value_matches" in cond:
        import re

        rx = re.compile(str(cond["value_matches"]), re.IGNORECASE)
        if not any(rx.search(h.value) for h in hits):
            return False
    return True


def derived_scores(
    metadata: list[MetadataField], facts: dict, cfg: Config
) -> dict[str, float]:
    by_field: dict[str, list[MetadataField]] = {}
    for m in metadata:
        by_field.setdefault(m.field, []).append(m)

    out: dict[str, float] = {}
    for rule in cfg.derived_tag_rules:
        tag = rule.get("tag")
        if not tag:
            continue
        all_of = rule.get("all_of") or []
        any_of = rule.get("any_of") or []
        if all_of and not all(_eval_condition(c, by_field, facts) for c in all_of):
            continue
        if any_of and not any(_eval_condition(c, by_field, facts) for c in any_of):
            continue
        if not all_of and not any_of:
            continue
        conf = float(rule.get("confidence", 0.9))
        out[tag] = max(out.get(tag, 0.0), conf)
    return out


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def build_estimator(cv_folds: int):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.multiclass import OneVsRestClassifier
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
    cv = StratifiedKFold(n_splits=max(2, cv_folds), shuffle=False)
    base = CalibratedClassifierCV(LinearSVC(random_state=0, dual="auto"), method="sigmoid", cv=cv)
    # One independent calibrated model per tag, so probabilities do not compete.
    return Pipeline([("features", features), ("clf", OneVsRestClassifier(base))])


@dataclass
class TaggerModel:
    pipeline: object
    tags: list[str]

    def predict_scores(self, text: str) -> dict[str, float]:
        probs = self.pipeline.predict_proba([text])[0]
        return {t: round(float(p), 4) for t, p in zip(self.tags, probs)}


def train(texts: list[str], tag_lists: list[list[str]], out_dir: str | Path) -> dict:
    """Fit one calibrated binary model per tag. Skips tags too rare to calibrate."""
    import collections

    import joblib

    if len(texts) != len(tag_lists):
        raise ValueError("texts and tag_lists differ in length")

    counts = collections.Counter(t for tags in tag_lists for t in set(tags))
    n = len(texts)
    # A tag needs both positives and negatives, at least 2 of each, to calibrate.
    usable = sorted(t for t, c in counts.items() if c >= 2 and (n - c) >= 2)
    skipped = sorted(set(counts) - set(usable))
    if not usable:
        raise ValueError("no tag has enough positive and negative examples to train")

    y = [[1 if t in set(tags) else 0 for t in usable] for tags in tag_lists]
    folds = min(5, min(min(counts[t], n - counts[t]) for t in usable))
    pipe = build_estimator(folds)
    pipe.fit(texts, y)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "tags": usable}, out / MODEL_FILE)
    meta = {
        "kind": "tagger",
        "tags": usable,
        "skipped_tags": skipped,
        "tag_counts": dict(sorted(counts.items())),
        "n_samples": n,
        "cv_folds": max(2, folds),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    log.info("tagger trained", extra=meta)
    return meta


def load_model(model_dir: str | Path) -> TaggerModel | None:
    d = Path(model_dir)
    if not (d / MODEL_FILE).exists():
        return None
    try:
        import joblib

        blob = joblib.load(d / MODEL_FILE)
        return TaggerModel(pipeline=blob["pipeline"], tags=[str(t) for t in blob["tags"]])
    except Exception as exc:
        log.error("tagger load failed", extra={"dir": str(d), "error": str(exc)})
        return None


# --------------------------------------------------------------------------
# threshold tuning
# --------------------------------------------------------------------------


def fit_threshold(y_true: list[int], y_prob: list[float]) -> tuple[float, float]:
    """Threshold that maximises F1 for one tag. Returns (threshold, f1)."""
    if not y_true or not any(y_true):
        return 0.5, 0.0
    candidates = sorted({round(p, 4) for p in y_prob} | {0.05, 0.5, 0.95})
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        tp = sum(1 for a, p in zip(y_true, y_prob) if a == 1 and p >= t)
        fp = sum(1 for a, p in zip(y_true, y_prob) if a == 0 and p >= t)
        fn = sum(1 for a, p in zip(y_true, y_prob) if a == 1 and p < t)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        # ties go to the lower threshold, which favours recall on rare tags
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return round(best_t, 4), round(best_f1, 4)


def tune_thresholds(
    truth: dict[str, list[int]], probs: dict[str, list[float]]
) -> dict[str, dict]:
    """Per tag F1 optimal thresholds. Rare tags land lower, as they should."""
    out: dict[str, dict] = {}
    for tag in sorted(truth):
        if tag not in probs:
            continue
        t, f1 = fit_threshold(truth[tag], probs[tag])
        out[tag] = {"threshold": t, "f1": f1, "positives": sum(truth[tag])}
    return out


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


def apply_tags(
    full: str,
    pages: list[str],
    cfg: Config,
    metadata: list[MetadataField],
    facts: dict,
    model: TaggerModel | None = None,
) -> list[Tag]:
    """Merge keyword, model and derived evidence, then gate on per tag thresholds."""
    thresholds = tag_thresholds(cfg)
    default_t = float(cfg.tagging_opts().get("default_threshold", 0.5))
    best: dict[str, tuple[float, TagMethod]] = {}

    def offer(tag: str, conf: float, method: TagMethod) -> None:
        cur = best.get(tag)
        if cur is None or (conf, _PRIORITY[method]) > (cur[0], _PRIORITY[cur[1]]):
            best[tag] = (conf, method)

    for tag, score in keyword_scores(full, pages, cfg).items():
        offer(tag, score, TagMethod.KEYWORD)

    if model is not None and full.strip():
        try:
            for tag, prob in model.predict_scores(full).items():
                offer(tag, prob, TagMethod.TFIDF_SVM)
        except Exception as exc:
            log.error("tagger predict failed", extra={"error": str(exc)})

    # Validated metadata beats any keyword, so this runs last and wins ties.
    for tag, conf in derived_scores(metadata, facts, cfg).items():
        offer(tag, conf, TagMethod.DERIVED)

    out = [
        Tag(tag=tag, confidence=round(conf, 4), method=method)
        for tag, (conf, method) in best.items()
        if conf >= thresholds.get(tag, default_t)
    ]
    out.sort(key=lambda t: (-t.confidence, t.tag))
    return out

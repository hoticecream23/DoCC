"""Multi label tagging. Independent of classification.

A document has one class but many tags, so nothing here is a softmax.
Vocabulary, patterns, thresholds and derived rules all live in tags.yaml.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .classify import ModelIntegrityError, ProbaModel, load_verified, record_hash
from .config import Config
from .errors import summarize_exception
from .logging_setup import get_logger
from .rules import best_marker_score
from .schema import MetadataField, Tag, TagMethod

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

log = get_logger(__name__)

MODEL_FILE = "model.joblib"
META_FILE = "meta.json"

# Higher wins when two sources claim the same tag at equal confidence.
_PRIORITY = {TagMethod.DERIVED: 3, TagMethod.TFIDF_SVM: 2, TagMethod.KEYWORD: 1}

MIN_CALIBRATION_EXAMPLES = 2  # a tag needs this many positives and negatives
MIN_CV_FOLDS = 2
MAX_CV_FOLDS = 5


def tag_thresholds(cfg: Config) -> dict[str, float]:
    """Per-tag decision thresholds from tags.yaml.

    Args:
        cfg: Loaded configuration.

    Returns:
        {tag name: threshold}; tags without one get the default threshold.
    """
    default = float(cfg.tagging_opts().get("default_threshold", 0.5))
    return {t["name"]: float(t.get("threshold", default)) for t in cfg.tag_defs}


def keyword_scores(full: str, pages: list[str], cfg: Config) -> dict[str, float]:
    """Best firing pattern specificity per tag.

    Args:
        full: Full document text.
        pages: Per-page texts.
        cfg: Loaded configuration.

    Returns:
        {tag name: score} for every tag with a firing pattern.
    """
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


def _eval_condition(cond: dict[str, Any], by_field: dict[str, list[MetadataField]], facts: dict[str, Any]) -> bool:
    """True when one derived-tag condition holds on the facts or metadata."""
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


def derived_scores(metadata: list[MetadataField], facts: dict[str, Any], cfg: Config) -> dict[str, float]:
    """Tags implied by derived rules over extracted metadata and document facts.

    Args:
        metadata: Extracted metadata fields.
        facts: Document-level facts, such as page count.
        cfg: Loaded configuration.

    Returns:
        {tag name: confidence} for every rule that fires.
    """
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


def build_estimator(cv_folds: int) -> Pipeline:
    """Word plus char ngrams into one calibrated binary model per tag.

    Args:
        cv_folds: Calibration folds; raised to at least two.

    Returns:
        An unfitted sklearn Pipeline.
    """
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
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=max(MIN_CV_FOLDS, cv_folds), shuffle=False)
    base = CalibratedClassifierCV(LinearSVC(random_state=0, dual="auto"), method="sigmoid", cv=cv)
    # One independent calibrated model per tag, so probabilities do not compete.
    return Pipeline([("features", features), ("clf", OneVsRestClassifier(base))])


@dataclass
class TaggerModel:
    """A fitted one-vs-rest pipeline and its tags, in predict_proba column order."""

    pipeline: ProbaModel
    tags: list[str]

    def predict_scores(self, text: str) -> dict[str, float]:
        """Score one text against every tag independently.

        Args:
            text: Document text.

        Returns:
            {tag: probability}.
        """
        probs = self.pipeline.predict_proba([text])[0]
        return {t: round(float(p), 4) for t, p in zip(self.tags, probs, strict=True)}


def _usable_tags(counts: Mapping[str, int], n: int) -> list[str]:
    """Tags with at least MIN_CALIBRATION_EXAMPLES positives and negatives each, sorted."""
    # A tag needs both positives and negatives to calibrate.
    return sorted(t for t, c in counts.items() if c >= MIN_CALIBRATION_EXAMPLES and (n - c) >= MIN_CALIBRATION_EXAMPLES)


def train(texts: list[str], tag_lists: list[list[str]], out_dir: str | Path) -> dict[str, Any]:
    """Fit one calibrated binary model per tag. Skips tags too rare to calibrate.

    Args:
        texts: Training texts.
        tag_lists: Tags per text.
        out_dir: Directory for the model and meta files.

    Returns:
        A small training report, also written to meta.json.

    Raises:
        ValueError: Lengths differ, or no tag has enough examples.
    """
    import collections

    import joblib

    if len(texts) != len(tag_lists):
        raise ValueError("texts and tag_lists differ in length")

    counts = collections.Counter(t for tags in tag_lists for t in set(tags))
    n = len(texts)
    usable = _usable_tags(counts, n)
    skipped = sorted(set(counts) - set(usable))
    if not usable:
        raise ValueError("no tag has enough positive and negative examples to train")

    y = [[1 if t in set(tags) else 0 for t in usable] for tags in tag_lists]
    folds = min(MAX_CV_FOLDS, min(min(counts[t], n - counts[t]) for t in usable))
    pipe = build_estimator(folds)
    pipe.fit(texts, y)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "tags": usable}, out / MODEL_FILE)
    record_hash(out / MODEL_FILE)
    meta = {
        "kind": "tagger",
        "tags": usable,
        "skipped_tags": skipped,
        "tag_counts": dict(sorted(counts.items())),
        "n_samples": n,
        "cv_folds": max(MIN_CV_FOLDS, folds),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    log.info("tagger trained", extra=meta)
    return meta


def load_model(model_dir: str | Path) -> TaggerModel | None:
    """Load a trained tagger.

    Args:
        model_dir: Directory written by train().

    Returns:
        The model, or None when no model exists or it fails to unpickle.

    Raises:
        ModelIntegrityError: The model file fails the manifest check.
    """
    d = Path(model_dir)
    if not (d / MODEL_FILE).exists():
        return None
    try:
        blob = load_verified(d / MODEL_FILE)
        return TaggerModel(pipeline=blob["pipeline"], tags=[str(t) for t in blob["tags"]])
    except ModelIntegrityError:
        raise
    except Exception as exc:  # noqa: BLE001 - unpickling raises arbitrary types; degrade to keyword tags
        log.error("tagger load failed", extra={"dir": str(d), "error": summarize_exception(exc)})
        return None


# --------------------------------------------------------------------------
# threshold tuning
# --------------------------------------------------------------------------


def fit_threshold(y_true: list[int], y_prob: list[float]) -> tuple[float, float]:
    """Threshold that maximises F1 for one tag.

    Args:
        y_true: 0/1 truth per document.
        y_prob: Predicted probability per document, same order.

    Returns:
        (threshold, f1).
    """
    if not y_true or not any(y_true):
        return 0.5, 0.0
    candidates = sorted({round(p, 4) for p in y_prob} | {0.05, 0.5, 0.95})
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        pairs = list(zip(y_true, y_prob, strict=False))
        tp = sum(1 for a, p in pairs if a == 1 and p >= t)
        fp = sum(1 for a, p in pairs if a == 0 and p >= t)
        fn = sum(1 for a, p in pairs if a == 1 and p < t)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        # ties go to the lower threshold, which favours recall on rare tags
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return round(best_t, 4), round(best_f1, 4)


def tune_thresholds(truth: dict[str, list[int]], probs: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    """Per tag F1 optimal thresholds. Rare tags land lower, as they should.

    Args:
        truth: {tag: 0/1 truth per document}.
        probs: {tag: probability per document}.

    Returns:
        {tag: {"threshold", "f1", "positives"}} for tags present in both.
    """
    out: dict[str, dict[str, float | int]] = {}
    for tag in sorted(truth):
        if tag not in probs:
            continue
        t, f1 = fit_threshold(truth[tag], probs[tag])
        out[tag] = {"threshold": t, "f1": f1, "positives": sum(truth[tag])}
    return out


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


def _model_tag_scores(full: str, model: TaggerModel | None) -> dict[str, float]:
    """Model tag scores; empty with no model, blank text, or a predict failure."""
    if model is None or not full.strip():
        return {}
    try:
        return model.predict_scores(full)
    except Exception as exc:  # noqa: BLE001 - sklearn raises varied types; degrade to keyword tags
        log.error("tagger predict failed", extra={"error": summarize_exception(exc)})
        return {}


def apply_tags(
    full: str,
    pages: list[str],
    cfg: Config,
    metadata: list[MetadataField],
    facts: dict[str, Any],
    model: TaggerModel | None = None,
) -> list[Tag]:
    """Merge keyword, model and derived evidence, then gate on per tag thresholds.

    Args:
        full: Full document text.
        pages: Per-page texts.
        cfg: Loaded configuration.
        metadata: Extracted metadata fields.
        facts: Document-level facts for derived rules.
        model: Optional trained tagger.

    Returns:
        Tags above threshold, highest confidence first.
    """
    thresholds = tag_thresholds(cfg)
    default_t = float(cfg.tagging_opts().get("default_threshold", 0.5))
    best: dict[str, tuple[float, TagMethod]] = {}

    def offer(tag: str, conf: float, method: TagMethod) -> None:
        cur = best.get(tag)
        if cur is None or (conf, _PRIORITY[method]) > (cur[0], _PRIORITY[cur[1]]):
            best[tag] = (conf, method)

    for tag, score in keyword_scores(full, pages, cfg).items():
        offer(tag, score, TagMethod.KEYWORD)

    for tag, prob in _model_tag_scores(full, model).items():
        offer(tag, prob, TagMethod.TFIDF_SVM)

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

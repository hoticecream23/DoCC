"""Document classification: rule layer first, then a calibrated linear model.

Train and predict are separate entry points. Nothing here knows the name of
any document class, that all lives in classes.yaml.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .config import Config
from .errors import ModelIntegrityError, summarize_exception
from .io import atomic_open, atomic_write_text
from .logging_setup import get_logger
from .rules import best_marker_score
from .schema import Classification, ClassifyMethod

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

log = get_logger(__name__)

MODEL_FILE = "model.joblib"
META_FILE = "meta.json"
MANIFEST = "MANIFEST.sha256"

MIN_CLASSES = 2  # a classifier needs at least two labels to separate
MIN_CV_FOLDS = 2  # calibration needs two folds, so two examples per class
MAX_CV_FOLDS = 5


class ProbaModel(Protocol):
    """Anything with an sklearn-style ``predict_proba``."""

    def predict_proba(self, x: list[str]) -> Sequence[Sequence[float]]:
        """Return one row of class probabilities per input text.

        Args:
            x: Input texts.

        Returns:
            Rows of shape (len(x), n_classes).
        """
        ...


def _manifest_for(path: Path) -> tuple[Path, str]:
    """Manifest sits one level above the model dir: models/MANIFEST.sha256."""
    root = path.parent.parent
    return root / MANIFEST, path.relative_to(root).as_posix()


def _read_manifest(mf: Path) -> dict[str, str]:
    """Parse a sha256sum-format manifest into {relative name: digest}."""
    if not mf.exists():
        return {}
    out = {}
    for line in mf.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.lstrip("*").strip()] = digest.lower()
    return out


def record_hash(path: str | Path) -> None:
    """Add or update path's sha256 in the manifest (sha256sum format).

    Args:
        path: Model file to hash.
    """
    path = Path(path)
    mf, name = _manifest_for(path)
    entries = _read_manifest(mf)
    entries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = "".join(f"{h}  {n}\n" for n, h in sorted(entries.items()))
    with atomic_open(mf, newline="\n") as fh:
        fh.write(lines)


def load_verified(path: str | Path) -> Any:
    """Run joblib.load only if path's sha256 matches the manifest. Unpickling runs code.

    Args:
        path: Model file to load.

    Returns:
        The unpickled object.

    Raises:
        ModelIntegrityError: The manifest has no entry for path, or the hash differs.
    """
    import joblib

    path = Path(path)
    mf, name = _manifest_for(path)
    want = _read_manifest(mf).get(name)
    if want is None:
        raise ModelIntegrityError(f"model file {path} has no entry in {mf}, refusing to load")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise ModelIntegrityError(f"model file {path} sha256 does not match {mf}, refusing to load")
    return joblib.load(path)


def rule_scores(full: str, pages: list[str], cfg: Config) -> dict[str, float]:
    """Best firing marker specificity per class. Independent, not a softmax.

    Args:
        full: Full document text.
        pages: Per-page texts.
        cfg: Loaded configuration.

    Returns:
        {class name: score} for every class with a firing marker.
    """
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


def _cv_folds(min_class_count: int) -> int:
    """Calibration folds: as many as the rarest class allows, clamped."""
    return max(MIN_CV_FOLDS, min(MAX_CV_FOLDS, min_class_count))


def build_estimator(min_class_count: int) -> Pipeline:
    """Word ngrams plus char ngrams, calibrated into real probabilities.

    Char ngrams carry the load when OCR noise breaks word tokens.

    Args:
        min_class_count: Example count of the rarest class; bounds the CV folds.

    Returns:
        An unfitted sklearn Pipeline.
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
    folds = _cv_folds(min_class_count)
    # shuffle off so a rerun on the same data gives the same model
    cv = StratifiedKFold(n_splits=folds, shuffle=False)
    clf = CalibratedClassifierCV(LinearSVC(random_state=0, dual="auto"), method="sigmoid", cv=cv)
    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class ClassifierModel:
    """A fitted pipeline and its class labels, in predict_proba column order."""

    pipeline: ProbaModel
    labels: list[str]

    def predict_scores(self, text: str) -> dict[str, float]:
        """Score one text against every class.

        Args:
            text: Document text.

        Returns:
            {label: probability}.
        """
        probs = self.pipeline.predict_proba([text])[0]
        return {lab: round(float(p), 4) for lab, p in zip(self.labels, probs, strict=True)}


def train(texts: list[str], labels: list[str], out_dir: str | Path) -> dict[str, Any]:
    """Fit and persist.

    Args:
        texts: Training texts.
        labels: One label per text.
        out_dir: Directory for the model and meta files.

    Returns:
        A small training report, also written to meta.json.

    Raises:
        ValueError: Lengths differ, fewer than two classes, or a class has one example.
    """
    import collections

    import joblib

    if len(texts) != len(labels):
        raise ValueError("texts and labels differ in length")
    counts = collections.Counter(labels)
    if len(counts) < MIN_CLASSES:
        raise ValueError("need at least 2 distinct classes, got " + str(list(counts)))
    smallest = min(counts.values())
    if smallest < MIN_CV_FOLDS:
        rare = [k for k, v in counts.items() if v < MIN_CV_FOLDS]
        raise ValueError("every class needs at least 2 examples for calibration, thin: " + str(rare))

    pipe = build_estimator(smallest)
    pipe.fit(texts, labels)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with atomic_open(out / MODEL_FILE, mode="wb") as fh:
        joblib.dump(pipe, fh)
    record_hash(out / MODEL_FILE)
    meta = {
        "kind": "classifier",
        "labels": sorted(counts),
        "class_counts": dict(sorted(counts.items())),
        "n_samples": len(texts),
        "cv_folds": _cv_folds(smallest),
    }
    atomic_write_text(out / META_FILE, json.dumps(meta, indent=2, sort_keys=True))
    log.info("classifier trained", extra=meta)
    return meta


def load_model(model_dir: str | Path) -> ClassifierModel | None:
    """Load a trained classifier.

    Args:
        model_dir: Directory written by train().

    Returns:
        The model, or None when no model exists or it fails to unpickle.
        The rule layer still works without it.

    Raises:
        ModelIntegrityError: The model file fails the manifest check.
    """
    d = Path(model_dir)
    if not (d / MODEL_FILE).exists():
        return None
    try:
        pipe = load_verified(d / MODEL_FILE)
        # sklearn hands back numpy strings, force plain str for clean JSON keys
        return ClassifierModel(pipeline=pipe, labels=[str(c) for c in pipe.classes_])
    except ModelIntegrityError:
        raise
    except Exception as exc:  # noqa: BLE001 - unpickling raises arbitrary types; degrade to the rule layer
        log.error("classifier load failed", extra={"dir": str(d), "error": summarize_exception(exc)})
        return None


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


def _top(scores: dict[str, float]) -> tuple[str, float]:
    """Highest score; ties break on name so the output is stable."""
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _rule_result(label: str, score: float, rules: dict[str, float]) -> Classification:
    """Build a rule-layer Classification."""
    return Classification(
        label=label,
        confidence=round(score, 4),
        all_scores=dict(sorted(rules.items())),
        method=ClassifyMethod.RULE,
    )


def _model_scores(full: str, model: ClassifierModel | None) -> dict[str, float] | None:
    """Sorted model distribution; None with no model, blank text, or a predict failure."""
    if model is None or not full.strip():
        return None
    try:
        scores = model.predict_scores(full)
        _top(scores)  # an empty distribution is a predict failure
    except Exception as exc:  # noqa: BLE001 - sklearn raises varied types; degrade to the rule layer
        log.error("classifier predict failed", extra={"error": summarize_exception(exc)})
        return None
    return dict(sorted(scores.items()))


def _abstain(unknown: str, confidence: float, scores: dict[str, float]) -> Classification:
    """Build a fallback Classification that keeps the distribution for diagnosis."""
    return Classification(
        label=unknown, confidence=round(confidence, 4), all_scores=scores, method=ClassifyMethod.FALLBACK
    )


def classify(full: str, pages: list[str], cfg: Config, model: ClassifierModel | None = None) -> Classification:
    """Cascade: strong rule, then model, then weak rule, then unknown.

    Never forces a guess. Coverage at a fixed error rate is the metric.

    Args:
        full: Full document text.
        pages: Per-page texts.
        cfg: Loaded configuration.
        model: Optional trained classifier.

    Returns:
        The Classification; the label is the configured unknown label on abstention.
    """
    opts = cfg.classify_opts()
    unknown = str(opts.get("unknown_label", "unknown"))
    short_circuit = float(opts.get("rule_short_circuit", 0.9))
    rule_min = float(opts.get("rule_min_confidence", 0.7))
    model_min = float(opts.get("model_min_confidence", 0.45))

    rules = rule_scores(full, pages, cfg)
    best_rule_label, best_rule_score = _top(rules) if rules else ("", 0.0)
    if best_rule_score >= short_circuit:
        return _rule_result(best_rule_label, best_rule_score, rules)

    model_scores = _model_scores(full, model)
    model_prob = 0.0
    if model_scores is not None:
        label, model_prob = _top(model_scores)
        if model_prob >= model_min:
            return Classification(
                label=label, confidence=round(model_prob, 4), all_scores=model_scores, method=ClassifyMethod.TFIDF_SVM
            )
        # Model spoke but not loudly enough, which is the model declining,
        # not the model disagreeing. Fall through to the weak rule layer
        # below. Returning unknown here threw away correct rule answers:
        # on the client set it cost 11 invoices that the rule layer had
        # already scored at 0.88 against a rule_min of 0.7, because a model
        # trained on 42 synthetic samples cannot speak loudly about
        # anything and its silence was being read as a verdict.

    if best_rule_score >= rule_min:
        return _rule_result(best_rule_label, best_rule_score, rules)

    # Nothing was confident enough. Keep whichever distribution we have, so an
    # abstention can still be diagnosed: the model's if it ran, else the rules'.
    if model_scores is not None:
        return _abstain(unknown, model_prob, model_scores)
    return _abstain(unknown, best_rule_score, dict(sorted(rules.items())))

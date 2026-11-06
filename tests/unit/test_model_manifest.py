"""Model files load only when their sha256 matches models/MANIFEST.sha256."""

import shutil
from pathlib import Path

import pytest

from baseline import classify, tagging

REPO_MODELS = Path(__file__).resolve().parents[2] / "models"


def _copy_models(tmp_path: Path) -> Path:
    dst = tmp_path / "models"
    shutil.copytree(REPO_MODELS, dst)
    return dst


def test_matching_manifest_loads(tmp_path):
    models = _copy_models(tmp_path)
    assert classify.load_model(models / "classifier") is not None
    assert tagging.load_model(models / "tagger") is not None


def test_tampered_model_refused(tmp_path):
    models = _copy_models(tmp_path)
    f = models / "classifier" / classify.MODEL_FILE
    with open(f, "ab") as fh:
        fh.write(b"\x00")
    with pytest.raises(classify.ModelIntegrityError, match="classifier.*does not match"):
        classify.load_model(models / "classifier")


def test_missing_manifest_entry_refused(tmp_path):
    models = _copy_models(tmp_path)
    (models / classify.MANIFEST).write_text("", encoding="utf-8")
    with pytest.raises(classify.ModelIntegrityError, match="no entry"):
        tagging.load_model(models / "tagger")


def test_record_hash_updates_manifest(tmp_path):
    models = _copy_models(tmp_path)
    (models / classify.MANIFEST).unlink()
    classify.record_hash(models / "tagger" / tagging.MODEL_FILE)
    text = (models / classify.MANIFEST).read_text(encoding="utf-8")
    assert text.endswith("  tagger/model.joblib\n") and "classifier" not in text
    assert tagging.load_model(models / "tagger") is not None


def test_train_writes_manifest_and_loads(tmp_path):
    texts = ["invoice total due net 30"] * 3 + ["salary slip basic pay net pay"] * 3
    labels = ["invoice"] * 3 + ["salary_slip"] * 3
    classify.train(texts, labels, tmp_path / "models" / "classifier")
    text = (tmp_path / "models" / classify.MANIFEST).read_text(encoding="utf-8")
    assert text.endswith("  classifier/model.joblib\n")
    m = classify.load_model(tmp_path / "models" / "classifier")
    assert m.labels == ["invoice", "salary_slip"]

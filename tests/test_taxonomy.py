"""The client label set and our class list must stay reconciled.

Classification.csv is the only real gold this project has, so the mapping
between their 14 folder labels and our classes is load bearing: the gold
builder and the workbook's Taxonomy tab both key on it. These tests fail the
moment the two drift apart, which is the failure that would otherwise show up
as an unexplained accuracy drop months later.

Nothing here touches Client_Documents or Classification_Doc. Those are client
material, gitignored, and not present on every machine.
"""

import pytest
import yaml

from baseline.classify import classify
from baseline.config import load_config


@pytest.fixture(scope="module")
def taxonomy(cfg):
    tax = cfg.taxonomy
    assert tax, "config/taxonomy.yaml is missing or empty"
    return tax


def test_every_client_label_maps_to_a_declared_class(cfg, taxonomy):
    known = set(cfg.class_names)
    for m in taxonomy["mappings"]:
        target = m.get("class")
        if target is None:
            continue  # a deliberate refusal, recorded rather than invented
        assert target in known, f"{m['label']} maps to unknown class {target}"


def test_a_many_to_one_mapping_must_declare_itself_lossy(taxonomy):
    """Two labels sharing a class means the label cannot be recovered."""
    seen: dict[str, list[str]] = {}
    for m in taxonomy["mappings"]:
        seen.setdefault(m.get("class"), []).append(m["label"])
    for cls, labels in seen.items():
        if cls is not None and len(labels) > 1:
            for m in taxonomy["mappings"]:
                if m["label"] in labels:
                    assert m.get("lossy") is True, (
                        f"{m['label']} shares class {cls} with "
                        f"{len(labels) - 1} other label(s) but is not marked lossy"
                    )


def test_the_declared_counts_match_the_labelled_set_size(taxonomy):
    """112 rows in Classification.csv. If this drifts, the CSV changed."""
    assert sum(m["n"] for m in taxonomy["mappings"]) == 112


def test_conflicting_rows_are_recorded_not_deduplicated(taxonomy):
    """Four file names carry two labels each. Silently dropping one loses gold."""
    conflicts = taxonomy.get("conflicts", [])
    assert conflicts, "the doubly labelled rows must stay written down"
    for c in conflicts:
        assert len(c["labels"]) == 2
        # Every conflict collapses to one class, so none of them can change a
        # class level score. That is the only reason they are safe to carry.
        classes = {
            m["class"] for m in taxonomy["mappings"] if m["label"] in c["labels"]
        }
        assert len(classes) == 1, f"{c['file']} spans classes {classes}"


def test_unscored_classes_are_genuinely_absent_from_the_labels(cfg, taxonomy):
    mapped = {m.get("class") for m in taxonomy["mappings"]}
    for name in taxonomy.get("unscored_classes", []):
        assert name in cfg.class_names, f"{name} is not a declared class"
        if name in mapped:
            # court_document is both mapped and listed, because the labelled
            # set reaches it only through lossy labels.
            lossy = [
                m["label"] for m in taxonomy["mappings"]
                if m.get("class") == name and not m.get("lossy")
            ]
            assert not lossy, f"{name} is reachable losslessly from {lossy}"


# ------------------------------------------------------- the new class

COURT_TEXT = (
    "Afroznisha vs Delhi Wakf Board & Ors on 15 December, 2021\n"
    "Author: Prathiba M. Singh\nBench: Prathiba M. Singh\n"
    "IN THE HIGH COURT OF DELHI AT NEW DELHI\n"
    "Reserved on: 18th November, 2021\nDate of decision: 15th December, 2021\n"
    "C.R.P. 223/2019\nLearned counsel for the petitioner submits that ...\n"
)


def test_a_court_judgment_classifies_as_court_document(cfg):
    result = classify(COURT_TEXT, [COURT_TEXT], cfg, model=None)
    assert result.label == "court_document"
    assert result.method.value == "rule"


def test_court_markers_do_not_fire_on_a_business_document(cfg):
    text = (
        "TAX INVOICE\nInvoice No: INV-2024-0042\nGSTIN: 27AABCU9603R1ZM\n"
        "Place of supply: Maharashtra\nTotal: 11800.00\n"
    )
    result = classify(text, [text], cfg, model=None)
    assert result.label == "invoice"
    assert result.all_scores.get("court_document", 0.0) == 0.0


def test_the_provenance_marker_alone_cannot_short_circuit(cfg):
    """The indiankanoon banner is a download artefact, not evidence of a court."""
    short_circuit = float(cfg.classify_opts().get("rule_short_circuit", 0.9))
    classes = {c["name"]: c for c in cfg.classes["classes"]}
    banner = [
        m for m in classes["court_document"]["markers"]
        if "indiankanoon" not in str(m) and r"\bon\s+\d{1,2}" in str(m.get("pattern", ""))
    ]
    assert banner, "the provenance marker is gone; drop this test with it"
    for m in banner:
        assert float(m["specificity"]) < short_circuit


def test_court_document_needs_only_a_yaml_edit(tmp_path, cfg):
    """The class name must not have leaked into any module."""
    import shutil

    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    for name in ("pipeline.yaml", "fields.yaml", "tags.yaml"):
        shutil.copyfile(cfg.config_dir / name, cfgdir / name)
    classes = yaml.safe_load(
        (cfg.config_dir / "classes.yaml").read_text(encoding="utf-8")
    )
    for c in classes["classes"]:
        if c["name"] == "court_document":
            c["name"] = "tribunal_ruling"
    (cfgdir / "classes.yaml").write_text(yaml.safe_dump(classes), encoding="utf-8")

    custom = load_config(cfgdir)
    result = classify(COURT_TEXT, [COURT_TEXT], custom, model=None)
    assert result.label == "tribunal_ruling"

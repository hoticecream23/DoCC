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


# ------------------------------------------- the cascade must reach the rules

def test_a_quiet_model_does_not_veto_a_confident_rule(cfg, tmp_path):
    """A model below its threshold has declined, not disagreed.

    This cost 11 client invoices. The rule layer scored them 0.88 against a
    rule_min of 0.7, and a model trained on 42 synthetic samples returned
    something under model_min_confidence, which was being read as a verdict
    and returned as unknown before the weak rule layer was ever consulted.
    """
    from baseline.classify import classify

    class QuietModel:
        def predict_scores(self, text):
            return {"contract": 0.01, "receipt": 0.02}

    text = "Invoice No: 1043\nInvoice Date: 12-30-2025\nDue Date: 01-29-2026\n"
    result = classify(text, [text], cfg, model=QuietModel())
    assert result.label == "invoice"
    assert result.method.value == "rule"


def test_a_confident_model_still_wins_over_a_weak_rule(cfg):
    from baseline.classify import classify

    class LoudModel:
        def predict_scores(self, text):
            return {"salary_slip": 0.99}

    text = "Invoice No: 1043\n"
    result = classify(text, [text], cfg, model=LoudModel())
    assert result.label == "salary_slip"
    assert result.method.value == "tfidf_svm"


# ------------------------------------------------- the deck's banking classes

# The eight classes taken from the POC 1 deck have no documents behind them, so
# these are the only checks that exist for them. Each pair asks the two
# questions the corpus cannot: does the title classify, and does a document
# merely mentioning the title stay where it was.

BANKING = {
    "address_proof": "PROOF OF ADDRESS\nThis is to certify the residence of ...\n",
    "kyc_form": "KYC FORM\nKnow Your Customer declaration\nCustomer due diligence\n",
    "customer_application": "CUSTOMER APPLICATION FORM\nCustomer ID: 4471\nApplicant details\n",
    "loan_application": "LOAN APPLICATION FORM\nApplicant: R Mehta\nLoan amount: 500000\n",
    "sanction_letter": "LOAN SANCTION LETTER\nWe are pleased to sanction the facility\nRate of interest: 9.1%\n",
    "repayment_schedule": "REPAYMENT SCHEDULE\nEMI No | Principal | Interest | Outstanding balance\n1 | 4210 | 1880 | 495790\n",
    "account_opening_form": "ACCOUNT OPENING FORM\nSavings account\nNomination details\n",
    "account_closure_form": "ACCOUNT CLOSURE FORM\nReason for closure: relocation\nUnused cheque leaves surrendered\n",
}


@pytest.mark.parametrize("name,text", sorted(BANKING.items()))
def test_each_banking_class_classifies_its_own_title(cfg, name, text):
    result = classify(text, [text], cfg, model=None)
    assert result.label == name, f"{name} scored {result.all_scores}"


def test_no_banking_marker_fires_on_a_court_judgment(cfg):
    """The corpus is 41% court PDFs. A banking marker reaching them is the
    whole risk of adding classes with no documents to tune against."""
    result = classify(COURT_TEXT, [COURT_TEXT], cfg, model=None)
    assert result.label == "court_document"
    for name in BANKING:
        assert result.all_scores.get(name, 0.0) == 0.0, f"{name} fired on a judgment"


def test_sanction_in_the_criminal_sense_is_not_a_sanction_letter(cfg):
    """s.197 CrPC sanction for prosecution. A bare \\bsanction\\b marker would
    claim every one of these, and there are a lot of them in the corpus."""
    text = (
        "IN THE HIGH COURT OF DELHI AT NEW DELHI\n"
        "The question is whether sanction under Section 197 of the Code was "
        "obtained before cognizance. The sanction accorded by the competent "
        "authority is assailed by the petitioner.\n"
    )
    result = classify(text, [text], cfg, model=None)
    assert result.all_scores.get("sanction_letter", 0.0) == 0.0
    assert result.label == "court_document"


def test_a_credit_memo_seeking_sanction_is_not_a_sanction_letter(cfg):
    """Regression: Axis_Memorandum to ALCO.docx, found on the first real run
    over Client_Documents. A sanction letter is the reply, not the request."""
    text = (
        "Memorandum to the Asset Liability Management Committee\n"
        "Approval of Interest Rate for Refinance\n"
        "The rate of interest, tenure and moratorium have been negotiated.\n"
        "Recommendations would be submitted to the competent authority, viz., "
        "IFCC-CGM for sanction. Submitted please.\n"
    )
    result = classify(text, [text], cfg, model=None)
    assert result.label != "sanction_letter", f"scored {result.all_scores}"


def test_an_agreement_citing_a_repayment_schedule_stays_a_contract(cfg):
    """Regression: payment_agreement sample.jpg. Reached repayment_schedule at
    0.95 on a body clause, off contract, before the determiner lookbehinds."""
    text = (
        "PAYMENT PLAN AGREEMENT\n"
        "This Payment Plan Agreement (hereinafter referred to as the "
        '"Agreement") is made and effective as of the Effective Date.\n'
        "The Debtor hereby represents and warrants that this Agreement, as "
        "well as the repayment schedule, have been created in a way that the "
        "Debtor believes they can satisfy the Creditor's demands.\n"
    )
    result = classify(text, [text], cfg, model=None)
    assert result.label == "contract", f"scored {result.all_scores}"


def test_a_utility_bill_still_reads_as_an_invoice(cfg):
    """address_proof markers sit below the invoice layer deliberately. If that
    ordering is ever reversed it should be a decision, not this test failing."""
    text = (
        "ELECTRICITY BILL\nBill To: R Mehta\nSupply address: 14 Link Road\n"
        "Invoice Number: EB-9931\nInvoice Date: 01-08-2026\nAmount due: 2,410.00\n"
    )
    result = classify(text, [text], cfg, model=None)
    assert result.label == "invoice"


def test_an_abstention_still_carries_a_distribution_to_diagnose_it(cfg):
    from baseline.classify import classify

    class QuietModel:
        def predict_scores(self, text):
            return {"contract": 0.03, "receipt": 0.04}

    junk = "qq ww ee rr tt yy uu ii oo pp"
    result = classify(junk, [junk], cfg, model=QuietModel())
    assert result.label == "unknown"
    assert result.all_scores, "an abstention with no scores cannot be diagnosed"

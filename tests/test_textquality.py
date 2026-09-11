"""The text quality gate: does it fire on substituted text and stay quiet on clean text.

The numbers pinned here were measured on the client corpus, which is
gitignored, so the corpus itself cannot appear in a test. What can be pinned is
the behaviour those measurements justify, on strings short enough to read.
"""


from baseline.textquality import prefer, score_text

# Verbatim from 1-2-2026 (1)Ev.pdf and 1-28-202610011.pdf, which are the
# documents that motivated this module. Both fell to unknown on this text.
CORRUPT = (
    "lnvoice 160219808 lnvoice Date 12122t2025 Payment Due Date 'v16t2026 "
    "PO Number Amount Due $120.00 lnvoico No. 1OO11 NuDb€r"
)
# The same document after re-rendering the page and running Tesseract.
CLEAN = (
    "Invoice 160219808 Invoice Date 12/12/2025 Payment Due Date 1/16/2026 "
    "PO Number Amount Due $120.00 Invoice No. 10011 Number"
)


def test_substituted_text_scores_far_above_clean_text():
    assert score_text(CORRUPT).score > score_text(CLEAN).score * 5


def test_clean_text_scores_essentially_zero():
    assert score_text(CLEAN).score < 0.02


def test_the_measured_threshold_separates_the_two():
    threshold = 0.02
    assert score_text(CORRUPT).is_suspect(threshold)
    assert not score_text(CLEAN).is_suspect(threshold)


def test_lowercase_l_standing_in_for_a_capital_i_is_caught():
    assert score_text("lnvoice lnvolce lnvoic6").l_for_i > 0


def test_ordinary_words_starting_with_l_are_not():
    """The consonant restriction is what makes this safe."""
    q = score_text("like lost lease long lawful leave license lien loan")
    assert q.l_for_i == 0.0


def test_initials_and_case_numbers_are_not_corruption():
    """These are why the case shape rule skips tokens with a dot or slash.

    Rathnavel_vs_State_By.PDF is clean and scored 0.043 before this rule,
    entirely on citations like these.
    """
    q = score_text("G.Jayachandran V.Palanivel Crl.A.259 W.P.No.26378/2023 N.Balasundaram")
    assert q.case_shape == 0.0
    assert not q.is_suspect(0.02)


def test_uppercase_identifiers_are_not_corruption():
    """Real invoice and reference numbers must not read as substitution."""
    q = score_text("INV-2024-0042 IN3308526 GSTIN 27AABCU9603R1ZM HDFC0001234")
    assert not q.is_suspect(0.02)


def test_non_latin_script_is_reported_separately_not_scored_as_corrupt():
    """A Karnataka judgment in a legacy Kannada font is not corrupt English.

    On an earlier version of this metric it scored 0.233, the highest in the
    whole corpus, and re-running English OCR would not have improved it.
    """
    kannada_legacy = '"»A§gÀºÀ ªÀÄÆ®®Pr ¤ªÀÄUÉ ¤ÃªÀÅ'
    q = score_text(kannada_legacy + " IN THE HIGH COURT OF KARNATAKA AT BENGALURU")
    assert q.non_latin_rate > 0.3
    assert not q.is_suspect(0.02)


def test_empty_text_is_not_suspect():
    q = score_text("")
    assert q.tokens == 0
    assert not q.is_suspect(0.02)
    assert not score_text("   \n\f\n  ").is_suspect(0.02)


# ------------------------------------------------------- prefer()

def test_prefer_takes_the_cleaner_reading():
    chosen, used_ocr = prefer(CORRUPT, CLEAN)
    assert chosen == CLEAN and used_ocr is True


def test_prefer_discards_an_ocr_pass_that_came_out_worse():
    """The gate must not be able to make a page worse by its own metric."""
    chosen, used_ocr = prefer(CLEAN, CORRUPT)
    assert chosen == CLEAN and used_ocr is False


def test_prefer_keeps_native_when_ocr_returns_nothing():
    chosen, used_ocr = prefer(CORRUPT, "   ")
    assert chosen == CORRUPT and used_ocr is False


def test_prefer_takes_ocr_when_the_page_had_no_native_text():
    chosen, used_ocr = prefer("", CLEAN)
    assert chosen == CLEAN and used_ocr is True


def test_the_module_holds_no_vocabulary():
    """Same rule as the rest of the repo: no domain words in a .py file."""
    import pathlib

    src = pathlib.Path("baseline/textquality.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in src.splitlines()
        if not line.strip().startswith("#")
    ).split('"""')
    # keep only the parts outside docstrings
    body = "".join(code[::2]).lower()
    for word in ("invoice", "receipt", "salary", "gstin", "aadhaar", "court"):
        assert word not in body, f"{word} must not appear in the code"


# ------------------------------------------------- the gate inside extraction

def test_a_clean_native_pdf_is_not_rescued(corpus, cfg):
    """The gate must be invisible on documents whose text layer is fine."""
    from baseline.extract import extract
    from baseline.schema import ExtractionMethod

    res = extract(corpus / "invoice_native.pdf", cfg)
    assert ExtractionMethod.OCR_RESCUED not in res.page_methods
    assert res.method == ExtractionMethod.NATIVE


def test_a_rescued_page_is_recorded_as_such_not_as_plain_ocr(cfg):
    """A reader must be able to tell 'never had text' from 'text discarded'."""
    from baseline.schema import ExtractionMethod

    assert ExtractionMethod.OCR_RESCUED.value == "ocr_rescued"
    assert ExtractionMethod.OCR_RESCUED is not ExtractionMethod.OCR


def test_the_gate_ignores_pages_with_too_little_text_to_judge(cfg):
    """A rate over a handful of tokens is noise, not evidence."""
    gate = cfg.extract_opts().get("quality_gate", {})
    assert int(gate.get("min_tokens", 0)) >= 20


def test_the_configured_threshold_is_the_measured_one(cfg):
    gate = cfg.extract_opts().get("quality_gate", {})
    assert float(gate["threshold"]) == 0.02

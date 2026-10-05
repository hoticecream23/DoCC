# Cascade v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `cascade` package that reads each document with a local vision language model, grounds every value it returns back into the canonical text, and writes `DocumentRecord` JSONL that the unchanged `baseline eval` and `baseline compare` score.

**Architecture:** The baseline runs first on every document and its record is the fallback. Cascade renders page images, asks an OpenAI compatible local server for `{class, fields}` under a JSON schema built from config, locates each value in the canonical text (exact, then whitespace and case insensitive, then through the field's own normalizer), and merges the result under the rule and checksum layer as a hard prior. Provenance for every decision goes to a sidecar.

**Tech Stack:** Python 3.10+, pydantic 2, PyYAML, PyMuPDF, Pillow, stdlib `urllib` for HTTP, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-05-cascade-v2-vlm-grounding-design.md`

## Global Constraints

- Branch: `cascade-v2`. Every commit lands there.
- **No new dependencies.** HTTP is `urllib.request`. Config validation is pydantic, already a dependency.
- **`cascade` never modifies `baseline` behaviour.** The only baseline changes are in Tasks 1, 2 and 5, and after Task 2 the baseline's records over the synthetic corpus must be identical to before, ignoring `schema_version` and timings.
- **Config over code.** No class name, field name, tag name or pattern appears in any `cascade/` module. They come from `config/*.yaml`.
- **Every emitted metadata value satisfies `full[char_start:char_end] == value`.**
- **`validated` is true only when a checksum validator (`CHECKSUM_VALIDATORS`) passed.**
- **Local models only.** The VLM client refuses any host that is not loopback or listed in `vlm.allowed_hosts`, does not follow redirects, and ignores proxy environment variables.
- **No GPU in tests.** Tests use a fake backend or a stub HTTP server on 127.0.0.1.
- Run tests with `python -m pytest`. The full suite was 365 passing before this plan.
- Commit messages: plain, imperative, no `Co-Authored-By` or "Generated with" lines.
- Never commit `.check/`, `*.provenance`, or anything under `Client_Documents/`, `Classification_Doc/` or `ML_Documents/`.

## File map

| File | Responsibility |
|---|---|
| `baseline/schema.py` | modify: schema 1.2.0, three new method values |
| `baseline/pipeline.py` | modify: `write_sidecar` made public, `tag_facts` extracted |
| `baseline/metadata/fields.py` | modify: add `candidates_for`, `scan_spans` |
| `config/fields.yaml` | modify: one line `description` per field |
| `config/cascade.yaml` | create: cascade settings |
| `cascade/__init__.py`, `cascade/__main__.py` | create: package and `python -m cascade` |
| `cascade/config.py` | create: load and validate `cascade.yaml` |
| `cascade/ground.py` | create: value to span in the canonical text |
| `cascade/prompt.py` | create: JSON schema, messages, answer parsing |
| `cascade/render.py` | create: page images |
| `cascade/vlm.py` | create: local only HTTP client |
| `cascade/merge.py` | create: class decision, field decision, confidence tiers |
| `cascade/runner.py` | create: one document and batch, provenance sidecar |
| `cascade/cli.py` | create: `cascade run` |
| `tests/vlm_stub.py` | create: stub OpenAI compatible server for tests |
| `tests/test_cascade_*.py` | create: one per module |

---

### Task 1: Baseline preparation: schema 1.2.0, public sidecar writer, shared tag facts

**Files:**
- Modify: `baseline/schema.py:17`, `baseline/schema.py:44-55`
- Modify: `baseline/pipeline.py:129-135`, `baseline/pipeline.py:199-228`
- Modify: `.gitignore`
- Test: `tests/test_cascade_baseline_prep.py`

**Interfaces:**
- Produces: `SCHEMA_VERSION == "1.2.0"`; `ClassifyMethod.VLM` (`"vlm"`); `MetadataMethod.VLM` (`"vlm"`); `MetadataMethod.VLM_RULE_AGREE` (`"vlm_rule_agree"`); `baseline.pipeline.write_sidecar(sidecar: BBoxSidecar, bbox_dir: Path) -> None`; `baseline.pipeline.tag_facts(text: TextRecord, page_count: int, is_scanned: bool) -> dict`.

- [ ] **Step 1: Capture the baseline's output before any change**

This is the reference that Task 2 diffs against. Run it before editing anything.

```bash
mkdir -p .check
python tests/make_corpus.py .check/corpus
python -m baseline run --input .check/corpus --output .check/before.jsonl --no-resume
```

Expected: `.check/before.jsonl` exists with one line per document.

- [ ] **Step 2: Ignore the check directory and provenance sidecars**

Append to `.gitignore`:

```
# Cascade: local verification output, and provenance sidecars, which carry the
# model's raw answer and so the document's values. Client material on a real run.
.check/
*.provenance
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_cascade_baseline_prep.py`:

```python
import gzip
import json

from baseline.schema import (
    SCHEMA_VERSION,
    BBoxSidecar,
    ClassifyMethod,
    MetadataField,
    MetadataMethod,
    WordBox,
)


def test_schema_version_is_1_2_0():
    assert SCHEMA_VERSION == "1.2.0"


def test_vlm_methods_round_trip_through_json():
    m = MetadataField(
        field="x", value="a", normalized_value="a", page=0, char_start=0, char_end=1,
        confidence=0.5, method=MetadataMethod.VLM_RULE_AGREE, validated=False,
    )
    back = MetadataField.model_validate(m.model_dump(mode="json"))
    assert back.method is MetadataMethod.VLM_RULE_AGREE
    assert MetadataMethod("vlm") is MetadataMethod.VLM
    assert ClassifyMethod("vlm") is ClassifyMethod.VLM


def test_write_sidecar_is_public_and_skips_a_sidecar_with_no_words(tmp_path):
    from baseline.pipeline import write_sidecar

    doc_id = "0" * 16
    write_sidecar(BBoxSidecar(doc_id=doc_id), tmp_path / "b")
    assert not (tmp_path / "b").exists()

    sc = BBoxSidecar(
        doc_id=doc_id, page_sizes=[(10.0, 10.0)],
        words=[WordBox(page=0, text="a", x0=0, y0=0, x1=1, y1=1)],
    )
    write_sidecar(sc, tmp_path / "b")
    with gzip.open(tmp_path / "b" / f"{doc_id}.json.gz") as fh:
        assert json.loads(fh.read())["doc_id"] == doc_id


def test_tag_facts_matches_what_the_pipeline_feeds_the_tagger(corpus, cfg):
    from baseline.pipeline import process_document, tag_facts

    record, _ = process_document(corpus / "invoice_native.pdf", cfg)
    facts = tag_facts(record.text, record.diagnostics.page_count, record.diagnostics.is_scanned)
    assert facts == {
        "page_count": record.diagnostics.page_count,
        "char_count": record.text.char_count,
        "extraction_confidence": record.text.extraction_confidence,
        "is_scanned": record.diagnostics.is_scanned,
        "method": record.text.method.value,
    }
```

- [ ] **Step 4: Run them to verify they fail**

Run: `python -m pytest tests/test_cascade_baseline_prep.py -v`
Expected: FAIL. `SCHEMA_VERSION` is `"1.1.0"`, `MetadataMethod` has no `VLM_RULE_AGREE`, and `write_sidecar` and `tag_facts` do not exist.

- [ ] **Step 5: Bump the schema and add the method values**

In `baseline/schema.py`, change line 17:

```python
SCHEMA_VERSION = "1.2.0"
```

Replace the `ClassifyMethod` and `MetadataMethod` enums with:

```python
class ClassifyMethod(str, Enum):
    TFIDF_SVM = "tfidf_svm"
    RULE = "rule"
    FALLBACK = "fallback"
    # Decided by a vision language model in cascade. Added in schema 1.2.0.
    VLM = "vlm"


class MetadataMethod(str, Enum):
    REGEX_CHECKSUM = "regex_checksum"
    REGEX_FORMAT = "regex_format"
    ANCHOR = "anchor"
    POSITIONAL = "positional"
    # Cascade: a model value located in the canonical text. VLM_RULE_AGREE
    # when a rule candidate produced the same value at the same place.
    # Added in schema 1.2.0.
    VLM = "vlm"
    VLM_RULE_AGREE = "vlm_rule_agree"
```

- [ ] **Step 6: Make the sidecar writer public and extract `tag_facts`**

In `baseline/pipeline.py`, rename `_write_sidecar` to `write_sidecar` (definition at line 199 and the call at line 228). No other change to its body.

Add this function directly above `process_document`:

```python
def tag_facts(text: TextRecord, page_count: int, is_scanned: bool) -> dict:
    """Document level facts the derived tag rules read.

    Cascade builds its tags from the same function, so the two pipelines
    cannot drift on what a tag rule sees.
    """
    return {
        "page_count": page_count,
        "char_count": text.char_count,
        "extraction_confidence": text.extraction_confidence,
        "is_scanned": is_scanned,
        "method": text.method.value,
    }
```

In `process_document`, replace the `facts = {...}` literal with:

```python
    facts = tag_facts(text, res.page_count, res.is_scanned)
```

- [ ] **Step 7: Run the new tests and the full suite**

Run: `python -m pytest tests/test_cascade_baseline_prep.py -v`
Expected: PASS, 4 tests.

Run: `python -m pytest -q`
Expected: all pass (365 + 4).

- [ ] **Step 8: Commit**

```bash
git add .gitignore baseline/schema.py baseline/pipeline.py tests/test_cascade_baseline_prep.py
git commit -m "Schema 1.2.0 for cascade, and share the sidecar writer and tag facts"
```

---

### Task 2: Baseline field helpers: every candidate, and every printed value of a field's shape

**Files:**
- Modify: `baseline/metadata/fields.py` (add two functions after `_rank`, before `extract_fields`)
- Test: `tests/test_cascade_field_helpers.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `candidates_for(full: str, pages: list[str], fdef: dict, cfg: Config, doc_class: str = "", words: list[WordBox] | None = None, page_sizes: list[tuple[float, float]] | None = None) -> list[Candidate]`: every strategy's candidates for one field, ranked by `_rank`.
  - `scan_spans(full: str, fdef: dict) -> list[tuple[int, int, str]]`: every match of the field's `pattern.regex` and `anchor.value_regex` anywhere in `full`, as `(start, end, raw)`, sorted and deduplicated.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cascade_field_helpers.py`:

```python
from baseline.metadata.fields import candidates_for, extract_fields, scan_spans
from baseline.schema import MetadataMethod


def _fdef(cfg, name):
    return next(f for f in cfg.field_defs if f["name"] == name)


TEXT = "Invoice No: INV-2024-0042\nRef INV-2024-0099\n"


def test_candidates_for_returns_every_strategy_not_just_the_first_hit(cfg):
    fdef = _fdef(cfg, "invoice_number")  # strategies: [anchor, pattern]
    cands = candidates_for(TEXT, [TEXT], fdef, cfg, "invoice")
    got = {(c.value, c.method) for c in cands}
    assert ("INV-2024-0042", MetadataMethod.ANCHOR) in got
    assert ("INV-2024-0099", MetadataMethod.REGEX_FORMAT) in got
    # extract_fields stops at the first strategy that hits, so it never sees 0099.
    fields, _ = extract_fields(TEXT, [TEXT], cfg, "invoice")
    assert [m.value for m in fields if m.field == "invoice_number"] == ["INV-2024-0042"]


def test_candidates_for_offsets_slice_back_to_their_values(cfg):
    fdef = _fdef(cfg, "invoice_number")
    for c in candidates_for(TEXT, [TEXT], fdef, cfg, "invoice"):
        assert TEXT[c.start : c.end] == c.value


def test_scan_spans_finds_values_with_no_anchor_label_near_them(cfg):
    text = "Tax 1800.00\nGrand Total: Rs. 11,800.00\n"
    spans = scan_spans(text, _fdef(cfg, "total_amount"))
    raws = [raw for _, _, raw in spans]
    assert "1800.00" in raws
    assert "11,800.00" in raws
    for s, e, raw in spans:
        assert text[s:e] == raw


def test_scan_spans_is_empty_for_a_field_with_no_value_patterns():
    assert scan_spans("anything at all", {"name": "x", "strategies": []}) == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_cascade_field_helpers.py -v`
Expected: FAIL with `ImportError: cannot import name 'candidates_for'`.

- [ ] **Step 3: Implement both helpers**

In `baseline/metadata/fields.py`, insert after `_rank` and before `extract_fields`:

```python
def candidates_for(
    full: str,
    pages: list[str],
    fdef: dict,
    cfg: Config,
    doc_class: str = "",
    words: list[WordBox] | None = None,
    page_sizes: list[tuple[float, float]] | None = None,
) -> list[Candidate]:
    """Every strategy's candidates for one field, ranked.

    extract_fields stops at the first strategy that hits. Cascade asks a
    different question, whether any rule would have produced a given span,
    and that needs all of them. extract_fields does not call this, so the
    baseline's own output cannot change because of it.
    """
    opts = cfg.metadata_opts()
    limit = int(opts.get("max_candidates_per_field", 25))
    default_window = int(opts.get("default_anchor_window", 120))
    out: list[Candidate] = []
    for strategy in fdef.get("strategies", []):
        if strategy == "pattern":
            out.extend(_pattern_candidates(full, fdef, limit))
        elif strategy == "anchor":
            out.extend(_anchor_candidates(full, fdef, default_window, limit))
        elif strategy == "positional":
            out.extend(
                _positional_candidates(
                    full, fdef, cfg.templates, doc_class, words or [],
                    page_spans(pages), page_sizes or [],
                )
            )
    out.sort(key=_rank)
    return out


def scan_spans(full: str, fdef: dict) -> list[tuple[int, int, str]]:
    """Every place a value of this field's shape is printed, label or not.

    Runs the field's own `pattern.regex` and `anchor.value_regex` over the
    whole text, with no anchor label and no validator. These are not claims
    that a match is this field. Cascade uses them only to locate a value the
    model has already chosen, when the model wrote it in another format.
    """
    out: set[tuple[int, int, str]] = set()
    for spec, key in ((fdef.get("pattern") or {}, "regex"), (fdef.get("anchor") or {}, "value_regex")):
        rx = spec.get(key)
        if not rx:
            continue
        for m in _compile(rx, bool(spec.get("ignorecase", False))).finditer(full):
            s, e, val = _span_of(m)
            if 0 <= s < e:
                out.add((s, e, val))
    return sorted(out)
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `python -m pytest tests/test_cascade_field_helpers.py -v`
Expected: PASS, 4 tests.

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Prove the baseline's output did not change**

```bash
python -m baseline run --input .check/corpus --output .check/after.jsonl --no-resume
python - <<'EOF'
import json

def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            r.pop("schema_version")
            r["diagnostics"].pop("timings_ms")
            out[r["doc_id"]] = r
    return out

a, b = load(".check/before.jsonl"), load(".check/after.jsonl")
assert a.keys() == b.keys(), "different documents"
diff = [k for k in a if a[k] != b[k]]
assert not diff, diff
print("identical", len(a))
EOF
```

Expected: `identical N`, where N is the number of documents. If anything differs, stop and find out why before going on.

- [ ] **Step 6: Commit**

```bash
git add baseline/metadata/fields.py tests/test_cascade_field_helpers.py
git commit -m "Expose every field candidate and every printed value of a field's shape"
```

---

### Task 3: Cascade package and validated config

**Files:**
- Create: `cascade/__init__.py`, `cascade/config.py`, `config/cascade.yaml`
- Modify: `pyproject.toml`
- Test: `tests/test_cascade_config.py`

**Interfaces:**
- Produces: `cascade.config.VLMSettings`, `RenderSettings`, `ConfidenceTiers`, `CascadeConfig` (pydantic, `extra="forbid"`), and `load_cascade_config(config_dir: str | Path | None = None) -> CascadeConfig`.
  - `VLMSettings`: `base_url: str`, `model: str | None`, `api_key_env: str | None`, `timeout_s: float`, `retries: int`, `backoff_s: float`, `max_tokens: int`, `allowed_hosts: list[str]`.
  - `RenderSettings`: `dpi: int`, `max_long_edge: int`, `max_pages: int`, `max_text_chars: int`.
  - `ConfidenceTiers`: `agree`, `grounded`, `ambiguous`, `vlm_class`, all floats in [0, 1], with `agree >= grounded >= ambiguous`.
  - `CascadeConfig`: `version: int`, `vlm`, `render`, `confidence`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cascade_config.py`:

```python
import pytest
import yaml
from pydantic import ValidationError

from conftest import ROOT


def _write(tmp_path, data):
    (tmp_path / "cascade.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return tmp_path


def _repo_data():
    return yaml.safe_load((ROOT / "config" / "cascade.yaml").read_text(encoding="utf-8"))


def test_repo_config_loads():
    from cascade.config import load_cascade_config

    c = load_cascade_config(ROOT / "config")
    assert c.vlm.base_url.startswith("http://127.0.0.1")
    assert c.vlm.model is None  # chosen per deployment, never defaulted
    assert c.confidence.agree >= c.confidence.grounded >= c.confidence.ambiguous


def test_a_misspelt_key_fails_at_load(tmp_path):
    from cascade.config import load_cascade_config

    data = _repo_data()
    data["vlm"]["timeout"] = 5  # should be timeout_s
    with pytest.raises(ValidationError):
        load_cascade_config(_write(tmp_path, data))


def test_confidence_tiers_must_be_ordered(tmp_path):
    from cascade.config import load_cascade_config

    data = _repo_data()
    data["confidence"]["ambiguous"] = 0.95
    with pytest.raises(ValidationError, match="agree >= grounded >= ambiguous"):
        load_cascade_config(_write(tmp_path, data))


def test_a_confidence_above_one_is_rejected(tmp_path):
    from cascade.config import load_cascade_config

    data = _repo_data()
    data["confidence"]["vlm_class"] = 1.5
    with pytest.raises(ValidationError):
        load_cascade_config(_write(tmp_path, data))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_cascade_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cascade'`.

- [ ] **Step 3: Create the package, config file and loader**

Create `cascade/__init__.py`:

```python
"""Cascade v2: a vision language model reads each page, and every value it
returns is located in the canonical text before it may be emitted.

Writes the same DocumentRecord JSONL as the baseline, so `baseline eval` and
`baseline compare` score it unchanged. See
docs/superpowers/specs/2026-10-05-cascade-v2-vlm-grounding-design.md.
"""
```

Create `config/cascade.yaml`:

```yaml
# Cascade v2 settings. See
# docs/superpowers/specs/2026-10-05-cascade-v2-vlm-grounding-design.md.
# Class and field vocabulary lives in classes.yaml and fields.yaml, not here.
version: 1

vlm:
  # An OpenAI compatible server: vLLM on a GPU box, Ollama on a laptop.
  # Only loopback hosts are accepted unless listed in allowed_hosts.
  base_url: http://127.0.0.1:8000/v1
  # Chosen by experiment on the client's hardware, after a licence check.
  # Null refuses to run: set it here or pass --model.
  model: null
  # Name of an environment variable holding the server's API key, if it wants one.
  api_key_env: null
  timeout_s: 180
  retries: 2
  backoff_s: 2.0
  max_tokens: 2048
  allowed_hosts: []

render:
  dpi: 150
  # Caps one oversized scan so it cannot exhaust GPU memory.
  max_long_edge: 1600
  # Only the first pages are sent to the model.
  max_pages: 4
  # Page text sent beside the images, in total across the pages sent.
  max_text_chars: 12000

# Starting points, not measured. Replaced by fitted calibration once there is
# reviewed gold to fit on. Every report built on these must say so.
confidence:
  agree: 0.9
  grounded: 0.75
  ambiguous: 0.5
  vlm_class: 0.8
```

Create `cascade/config.py`:

```python
"""cascade.yaml, validated when it is loaded.

A misspelt key or an out of range value fails the run before the first
document, not halfway through a batch.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, model_validator

from baseline.config import DEFAULT_CONFIG_DIR
from baseline.schema import Confidence, Strict


class VLMSettings(Strict):
    base_url: str
    # Null until a deployment chooses one. VLMClient refuses to start without it.
    model: str | None = None
    api_key_env: str | None = None
    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0, le=10)
    backoff_s: float = Field(ge=0)
    max_tokens: int = Field(gt=0)
    allowed_hosts: list[str] = Field(default_factory=list)


class RenderSettings(Strict):
    dpi: int = Field(gt=0, le=600)
    max_long_edge: int = Field(gt=0)
    max_pages: int = Field(gt=0)
    max_text_chars: int = Field(ge=0)


class ConfidenceTiers(Strict):
    agree: Confidence
    grounded: Confidence
    ambiguous: Confidence
    vlm_class: Confidence

    @model_validator(mode="after")
    def _ordered(self) -> "ConfidenceTiers":
        if not (self.agree >= self.grounded >= self.ambiguous):
            raise ValueError("confidence tiers must satisfy agree >= grounded >= ambiguous")
        return self


class CascadeConfig(Strict):
    version: int
    vlm: VLMSettings
    render: RenderSettings
    confidence: ConfidenceTiers


def load_cascade_config(config_dir: str | Path | None = None) -> CascadeConfig:
    d = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    with open(d / "cascade.yaml", "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return CascadeConfig.model_validate(data or {})
```

In `pyproject.toml`, change the package discovery line to:

```toml
include = ["baseline*", "cascade*"]
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_cascade_config.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add cascade/__init__.py cascade/config.py config/cascade.yaml pyproject.toml tests/test_cascade_config.py
git commit -m "Add the cascade package and its validated config"
```

---

### Task 4: Grounding: a model value to a span in the canonical text

**Files:**
- Create: `cascade/ground.py`
- Test: `tests/test_cascade_ground.py`

**Interfaces:**
- Consumes: `baseline.metadata.fields.scan_spans` (Task 2); `baseline.metadata.Candidate`, `NormCtx`, `normalize`; `baseline.schema.page_spans`.
- Produces:
  - Outcome constants `EXACT = "exact"`, `LOOSE = "loose"`, `NORMALISED = "normalised"`, `UNGROUNDED = "ungrounded"`.
  - `Grounding` (frozen dataclass): `outcome: str`, `start: int | None`, `end: int | None`, `n_matches: int`, `rule_agrees: bool`, property `grounded -> bool`.
  - `TextIndex(full: str, pages: list[str])` with attributes `full`, `spans`.
  - `ground(value: str, index: TextIndex, fdef: dict, rule_cands: list[Candidate], norm_opts: dict) -> Grounding`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cascade_ground.py`:

```python
from baseline.metadata import Candidate
from baseline.schema import PAGE_SEP, MetadataMethod


def _fdef(cfg, name):
    return next(f for f in cfg.field_defs if f["name"] == name)


def _opts(cfg):
    return cfg.fields.get("normalizer_options", {}) or {}


def _ground(cfg, field, value, pages, rule_cands=()):
    from cascade.ground import TextIndex, ground

    index = TextIndex(PAGE_SEP.join(pages), pages)
    return index.full, ground(value, index, _fdef(cfg, field), list(rule_cands), _opts(cfg))


def _cand(full, value, start):
    assert full[start : start + len(value)] == value
    return Candidate(value=value, start=start, end=start + len(value),
                     confidence=0.9, method=MetadataMethod.ANCHOR, validated=False)


def test_exact_unique_match(cfg):
    full, g = _ground(cfg, "invoice_number", "INV-2024-0042", ["Invoice No: INV-2024-0042\n"])
    assert (g.outcome, g.n_matches, g.rule_agrees) == ("exact", 1, False)
    assert full[g.start : g.end] == "INV-2024-0042"


def test_exact_ambiguous_without_a_rule_takes_the_first(cfg):
    text = "Acme Ltd\nSold By: Acme Ltd\n"
    full, g = _ground(cfg, "vendor_name", "Acme Ltd", [text])
    assert (g.outcome, g.n_matches, g.rule_agrees, g.start) == ("exact", 2, False, 0)


def test_exact_ambiguous_prefers_the_span_a_rule_also_produced(cfg):
    text = "Acme Ltd\nSold By: Acme Ltd\n"
    second = text.index("Acme Ltd", 1)
    full, g = _ground(cfg, "vendor_name", "Acme Ltd", [text], [_cand(text, "Acme Ltd", second)])
    assert (g.outcome, g.n_matches, g.rule_agrees, g.start) == ("exact", 2, True, second)


def test_loose_match_spans_the_documents_own_whitespace(cfg):
    full, g = _ground(cfg, "vendor_name", "Acme Traders Ltd", ["Sold By: Acme   Traders\nLtd\n"])
    assert g.outcome == "loose"
    assert full[g.start : g.end] == "Acme   Traders\nLtd"


def test_loose_match_ignores_case(cfg):
    full, g = _ground(cfg, "buyer_name", "Globex Corporation", ["Bill To: GLOBEX CORPORATION\n"])
    assert g.outcome == "loose"
    assert full[g.start : g.end] == "GLOBEX CORPORATION"


def test_loose_match_survives_a_casefold_that_changes_length(cfg):
    full, g = _ground(cfg, "vendor_name", "Hauptstraße GmbH", ["From: HAUPTSTRASSE GMBH\n"])
    assert g.outcome == "loose"
    assert full[g.start : g.end] == "HAUPTSTRASSE GMBH"


def test_normalised_date_grounds_a_reformatted_value(cfg):
    full, g = _ground(cfg, "invoice_date", "2024-03-14", ["Invoice Date: 14/03/2024\n"])
    assert g.outcome == "normalised"
    assert full[g.start : g.end] == "14/03/2024"


def test_normalised_amount_grounds_without_the_thousands_separator(cfg):
    full, g = _ground(cfg, "total_amount", "11800", ["Grand Total: Rs. 11,800.00\n"])
    assert g.outcome == "normalised"
    assert full[g.start : g.end] == "11,800.00"


def test_a_value_inside_a_longer_number_is_not_grounded_there(cfg):
    # "11,800" is printed only as part of "11,800.00". It must not be located
    # as a shorter span inside it, which would cost span F1 for no reason.
    full, g = _ground(cfg, "total_amount", "11,800", ["Grand Total: 11,800.00\n"])
    assert g.outcome == "normalised"
    assert full[g.start : g.end] == "11,800.00"


def test_a_value_inside_a_longer_token_is_ungrounded(cfg):
    _, g = _ground(cfg, "invoice_number", "100", ["Ref 10042\n"])
    assert g.outcome == "ungrounded"
    assert not g.grounded


def test_an_invented_value_is_ungrounded(cfg):
    _, g = _ground(cfg, "buyer_name", "Hallucinated Holdings", ["Bill To: Globex Corporation\n"])
    assert (g.outcome, g.start, g.end) == ("ungrounded", None, None)


def test_a_match_never_crosses_a_page_boundary(cfg):
    _, g = _ground(cfg, "vendor_name", "Acme Traders", ["Sold By: Acme", "Traders and more"])
    assert g.outcome == "ungrounded"


def test_an_empty_value_is_ungrounded(cfg):
    _, g = _ground(cfg, "vendor_name", "   ", ["Acme Ltd\n"])
    assert g.outcome == "ungrounded"


def test_every_grounded_span_slices_back_to_the_document_text(cfg):
    cases = [
        ("vendor_name", "Acme Traders Ltd", ["Sold By: Acme   Traders\nLtd\n"]),
        ("invoice_date", "2024-03-14", ["Invoice Date: 14/03/2024\n"]),
        ("total_amount", "11800", ["Grand Total: Rs. 11,800.00\n"]),
    ]
    for field, value, pages in cases:
        full, g = _ground(cfg, field, value, pages)
        assert g.grounded
        assert 0 <= g.start < g.end <= len(full)
        assert PAGE_SEP not in full[g.start : g.end]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_cascade_ground.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cascade.ground'`.

- [ ] **Step 3: Implement grounding**

Create `cascade/ground.py`:

```python
"""Locate a value the model returned inside the canonical text.

A value that cannot be located is not emitted. The record contract requires
full[char_start:char_end] == value, and a value the document does not contain
is exactly the value a model invented, so this step is both the offset step
and the guard against invention.

Three steps, and the first that matches wins:

1. exact: the value occurs verbatim;
2. loose: it occurs once whitespace runs are collapsed and case is folded;
3. normalised: the field's own normalizer gives the same value for it and for
   a place a value of the field's shape is printed (`scan_spans`) or a rule
   candidate.

A match never crosses a page boundary and never starts or ends inside a
longer word or number. The span reported is always the document's own text.
"""

from __future__ import annotations

from dataclasses import dataclass

from baseline.metadata import Candidate, NormCtx, normalize
from baseline.metadata.fields import scan_spans
from baseline.schema import page_spans

EXACT = "exact"
LOOSE = "loose"
NORMALISED = "normalised"
UNGROUNDED = "ungrounded"


@dataclass(frozen=True)
class Grounding:
    outcome: str
    start: int | None = None
    end: int | None = None
    n_matches: int = 0
    rule_agrees: bool = False

    @property
    def grounded(self) -> bool:
        return self.outcome != UNGROUNDED


def _fold(text: str) -> tuple[str, list[int]]:
    """Casefold, and collapse each whitespace run to one space.

    Returns the folded string and, for each of its characters, the index in
    `text` it came from. Casefolding can lengthen a character (ß to ss), which
    is why this is a map and not a constant shift.
    """
    out: list[str] = []
    src: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_space:
                out.append(" ")
                src.append(i)
            in_space = True
            continue
        in_space = False
        for f in ch.casefold():
            out.append(f)
            src.append(i)
    return "".join(out), src


def _folded(text: str) -> str:
    return _fold(text.strip())[0]


class TextIndex:
    """Per document lookups, built once and shared by every value."""

    def __init__(self, full: str, pages: list[str]):
        self.full = full
        self.spans = page_spans(pages)
        self.folded, self.src = _fold(full)

    def _on_one_page(self, s: int, e: int) -> bool:
        return any(ps <= s and e <= pe for ps, pe in self.spans)

    def _on_boundary(self, s: int, e: int) -> bool:
        full = self.full
        first, last = full[s], full[e - 1]
        if first.isalnum() and s > 0 and full[s - 1].isalnum():
            return False
        if last.isalnum() and e < len(full) and full[e].isalnum():
            return False
        # A number continues through a separator followed by a digit:
        # "11,800" is not a value inside "11,800.00", nor "800.00" inside it.
        if last.isdigit() and e + 1 < len(full) and full[e] in ".," and full[e + 1].isdigit():
            return False
        if first.isdigit() and s >= 2 and full[s - 1] in ".," and full[s - 2].isdigit():
            return False
        return True

    def usable(self, s: int, e: int) -> bool:
        return 0 <= s < e <= len(self.full) and self._on_one_page(s, e) and self._on_boundary(s, e)


def _norm(fdef: dict, value: str, full: str, s: int, e: int, opts: dict):
    normalized, _ = normalize(fdef.get("normalizer"), value, NormCtx(full=full, start=s, end=e, opts=opts))
    return normalized


def _exact(index: TextIndex, value: str) -> list[tuple[int, int]]:
    out = []
    i = index.full.find(value)
    while i >= 0:
        if index.usable(i, i + len(value)):
            out.append((i, i + len(value)))
        i = index.full.find(value, i + 1)
    return out


def _loose(index: TextIndex, value: str) -> list[tuple[int, int]]:
    target = _folded(value)
    if not target:
        return []
    out: set[tuple[int, int]] = set()
    i = index.folded.find(target)
    while i >= 0:
        s = index.src[i]
        e = index.src[i + len(target) - 1] + 1
        # Re-check on the original slice. A match that starts or ends partway
        # through a character's casefold expansion fails here.
        if _folded(index.full[s:e]) == target and index.usable(s, e):
            out.add((s, e))
        i = index.folded.find(target, i + 1)
    return sorted(out)


def _normalised(
    index: TextIndex, value: str, fdef: dict, rule_cands: list[Candidate], opts: dict
) -> list[tuple[int, int]]:
    target = _norm(fdef, value, value, 0, len(value), opts)
    if not target:
        return []
    places = scan_spans(index.full, fdef) + [(c.start, c.end, c.value) for c in rule_cands]
    return sorted(
        {
            (s, e)
            for s, e, raw in places
            if index.usable(s, e) and _norm(fdef, raw, index.full, s, e, opts) == target
        }
    )


def _agrees(c: Candidate, s: int, e: int, index: TextIndex, fdef: dict, opts: dict) -> bool:
    """The rule candidate is this span, or overlaps it with the same value."""
    if not (c.start < e and s < c.end):
        return False
    if (c.start, c.end) == (s, e):
        return True
    mine = _norm(fdef, index.full[s:e], index.full, s, e, opts)
    return mine is not None and _norm(fdef, c.value, index.full, c.start, c.end, opts) == mine


def ground(
    value: str, index: TextIndex, fdef: dict, rule_cands: list[Candidate], norm_opts: dict
) -> Grounding:
    value = value.strip()
    if not value:
        return Grounding(UNGROUNDED)
    steps = (
        (EXACT, lambda: _exact(index, value)),
        (LOOSE, lambda: _loose(index, value)),
        (NORMALISED, lambda: _normalised(index, value, fdef, rule_cands, norm_opts)),
    )
    for outcome, find in steps:
        matches = find()
        if not matches:
            continue
        # rule_cands arrive ranked best first (candidates_for sorts them), so
        # a value printed twice lands where the strongest rule put it, which
        # is the occurrence the baseline itself would have emitted.
        for c in rule_cands:
            for s, e in matches:
                if _agrees(c, s, e, index, fdef, norm_opts):
                    return Grounding(outcome, s, e, len(matches), True)
        s, e = matches[0]
        return Grounding(outcome, s, e, len(matches), False)
    return Grounding(UNGROUNDED)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_cascade_ground.py -v`
Expected: PASS, 14 tests.

- [ ] **Step 5: Commit**

```bash
git add cascade/ground.py tests/test_cascade_ground.py
git commit -m "Ground model values in the canonical text, or refuse them"
```

---

### Task 5: Prompt: schema from config, messages, strict answer parsing

**Files:**
- Modify: `config/fields.yaml` (one `description` line per field)
- Create: `cascade/prompt.py`
- Test: `tests/test_cascade_prompt.py`

**Interfaces:**
- Consumes: `baseline.config.Config` (`class_names`, `field_defs`, `classes`, `classify_opts()`).
- Produces:
  - `VLMAnswer` (frozen dataclass): `label: str`, `fields: dict[str, list[str]]` (only fields with at least one non empty value; values stripped and deduplicated in order).
  - `VLMOutputError(ValueError)`.
  - `check_descriptions(cfg: Config) -> None`: raises `ValueError` naming every class or field with no description.
  - `response_schema(cfg: Config) -> dict`.
  - `build_messages(cfg: Config, pages: list[str], images: list[bytes], max_text_chars: int) -> list[dict]`.
  - `parse_answer(content: str, cfg: Config) -> VLMAnswer`.

- [ ] **Step 1: Add a description to every field**

The workbook exporter already writes `description` into the Taxonomy tab, so this also fills that column. Run this once from the repo root. It inserts one line under each `  - name:` line and fails loudly if a name is missing or repeated.

```bash
python - <<'EOF'
from pathlib import Path

DESCRIPTIONS = {
    "pan": "Indian Permanent Account Number: five letters, four digits, one letter.",
    "aadhaar": "Indian Aadhaar number: 12 digits, often printed in groups of four.",
    "gstin": "Indian GST identification number, 15 characters.",
    "ifsc": "Indian bank branch IFSC code, 11 characters.",
    "card_number": "Payment card number, 16 digits.",
    "email": "An email address printed on the document.",
    "phone": "A phone number printed on the document.",
    "invoice_number": "The document's own invoice, bill, receipt or order reference number.",
    "invoice_date": "The date the invoice, receipt or order was issued.",
    "due_date": "The date payment is due.",
    "total_amount": "The final amount payable, including tax.",
    "tax_amount": "The total tax charged.",
    "account_number": "A bank account number.",
    "vendor_name": "The business that issued the document and is to be paid.",
    "buyer_name": "The customer the document is addressed to or billed to.",
    "petitioner": "The first named party bringing the case.",
    "respondent": "The first named party the case is brought against.",
    "judgment_date": "The date the judgment was delivered.",
    "bench": "The judges who heard the case.",
    "court": "The court that delivered the judgment.",
    "citation": "The law report citation of the judgment.",
    "case_number": "The court's own case, appeal or petition number.",
}
path = Path("config/fields.yaml")
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
out, seen = [], []
for line in lines:
    out.append(line)
    if line.startswith("  - name: "):
        name = line.split(":", 1)[1].strip()
        assert name in DESCRIPTIONS, f"no description written for {name}"
        seen.append(name)
        out.append(f'    description: "{DESCRIPTIONS[name]}"\n')
assert sorted(seen) == sorted(DESCRIPTIONS), (sorted(seen), sorted(DESCRIPTIONS))
path.write_text("".join(out), encoding="utf-8")
print("added", len(seen))
EOF
```

Expected: `added 22`. Then run `git diff --stat config/fields.yaml` and confirm 22 insertions and 0 deletions.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_cascade_prompt.py`:

```python
import base64
import dataclasses
import json

import pytest


def _names(cfg):
    return [f["name"] for f in cfg.field_defs]


def test_every_class_and_field_in_the_repo_config_has_a_description(cfg):
    from cascade.prompt import check_descriptions

    check_descriptions(cfg)


def test_a_missing_description_is_named(cfg):
    from cascade.prompt import check_descriptions

    bad = dataclasses.replace(cfg, fields={"fields": [{"name": "orphan_field"}]})
    with pytest.raises(ValueError, match="orphan_field"):
        check_descriptions(bad)


def test_schema_enumerates_classes_and_requires_every_field(cfg):
    from cascade.prompt import response_schema

    s = response_schema(cfg)
    assert s["additionalProperties"] is False
    assert set(s["properties"]["class"]["enum"]) == set(cfg.class_names) | {"unknown"}
    fields = s["properties"]["fields"]
    assert fields["additionalProperties"] is False
    assert sorted(fields["required"]) == sorted(_names(cfg))
    assert fields["properties"]["total_amount"] == {"type": "array", "items": {"type": "string"}}


def test_messages_carry_descriptions_text_and_images(cfg):
    from cascade.prompt import build_messages

    png = b"\x89PNG fake"
    msgs = build_messages(cfg, ["page one text", "page two text"], [png, png], max_text_chars=1000)
    system, user = msgs
    assert system["role"] == "system" and user["role"] == "user"
    for name in cfg.class_names + _names(cfg):
        assert name in system["content"]
    texts = [p["text"] for p in user["content"] if p["type"] == "text"]
    images = [p["image_url"]["url"] for p in user["content"] if p["type"] == "image_url"]
    assert "page one text" in texts[0] and "page two text" in texts[0]
    assert images == ["data:image/png;base64," + base64.b64encode(png).decode("ascii")] * 2


def test_page_text_is_capped_in_total(cfg):
    from cascade.prompt import build_messages

    # Q and Z appear nowhere in the header text, so the counts are the pages'.
    msgs = build_messages(cfg, ["Q" * 80, "Z" * 80], [], max_text_chars=100)
    text = msgs[1]["content"][0]["text"]
    assert text.count("Q") == 80
    assert text.count("Z") == 20


def test_no_text_part_when_the_budget_is_zero(cfg):
    from cascade.prompt import build_messages

    msgs = build_messages(cfg, ["some text"], [b"x"], max_text_chars=0)
    assert [p["type"] for p in msgs[1]["content"]] == ["image_url"]


def test_parse_answer_strips_drops_empties_and_deduplicates(cfg):
    from cascade.prompt import parse_answer

    content = json.dumps({"class": "invoice", "fields": {
        "invoice_number": [" INV-1 ", "INV-1", ""], "total_amount": [], "pan": ["  "],
    }})
    a = parse_answer(content, cfg)
    assert a.label == "invoice"
    assert a.fields == {"invoice_number": ["INV-1"]}


@pytest.mark.parametrize("content, message", [
    ("not json", "not JSON"),
    ("[]", "not an object"),
    (json.dumps({"class": "invoice", "fields": {}, "extra": 1}), "unexpected keys"),
    (json.dumps({"class": "made_up", "fields": {}}), "not a declared class"),
    (json.dumps({"class": ["invoice"], "fields": {}}), "not a declared class"),
    (json.dumps({"fields": {}}), "not a declared class"),
    (json.dumps({"class": "invoice", "fields": {"made_up": ["x"]}}), "undeclared fields"),
    (json.dumps({"class": "invoice", "fields": {"pan": "ABCDE1234F"}}), "list of strings"),
    (json.dumps({"class": "invoice", "fields": {"pan": [1]}}), "list of strings"),
    (json.dumps({"class": "invoice", "fields": []}), "fields is not an object"),
])
def test_parse_answer_rejects_anything_off_schema(cfg, content, message):
    from cascade.prompt import VLMOutputError, parse_answer

    with pytest.raises(VLMOutputError, match=message):
        parse_answer(content, cfg)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `python -m pytest tests/test_cascade_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cascade.prompt'`.

- [ ] **Step 4: Implement the prompt module**

Create `cascade/prompt.py`:

```python
"""What the model is asked, and the only shape of answer accepted back.

Every class and field name, and every description, comes from classes.yaml
and fields.yaml. A new document type for a new client is a YAML edit.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

from baseline.config import Config

_INSTRUCTIONS = """You read business and legal documents and return JSON.

Return the document's class and the value of every field you can find.

Rules:
- Copy each value exactly as it is printed, character for character. Do not
  reformat dates or amounts, do not correct spelling, do not translate.
- When the value appears in the page text provided, copy it as it appears there.
- Return a value only if it is printed on the document. If a field is not
  there, return an empty list for it. Never guess.
- Use "{unknown}" as the class when the document is none of the classes below.
"""


class VLMOutputError(ValueError):
    """The model's answer is not the shape the schema asked for."""


@dataclass(frozen=True)
class VLMAnswer:
    label: str
    fields: dict[str, list[str]]


def _unknown(cfg: Config) -> str:
    return str(cfg.classify_opts().get("unknown_label", "unknown"))


def _field_names(cfg: Config) -> list[str]:
    return [f["name"] for f in cfg.field_defs if f.get("name")]


def check_descriptions(cfg: Config) -> None:
    """Fail before the first document if the prompt would have a blank line."""
    missing = [f"class {c.get('name')}" for c in cfg.classes.get("classes", []) if not c.get("description")]
    missing += [f"field {f.get('name')}" for f in cfg.field_defs if not f.get("description")]
    if missing:
        raise ValueError("no description in config for: " + ", ".join(missing))


def response_schema(cfg: Config) -> dict:
    names = _field_names(cfg)
    return {
        "type": "object",
        "properties": {
            "class": {"type": "string", "enum": [*cfg.class_names, _unknown(cfg)]},
            "fields": {
                "type": "object",
                "properties": {n: {"type": "array", "items": {"type": "string"}} for n in names},
                "required": names,
                "additionalProperties": False,
            },
        },
        "required": ["class", "fields"],
        "additionalProperties": False,
    }


def _system_text(cfg: Config) -> str:
    lines = [_INSTRUCTIONS.format(unknown=_unknown(cfg)), "Classes:"]
    for c in cfg.classes.get("classes", []):
        lines.append(f"- {c['name']}: {c.get('description', '')}")
    lines.append("")
    lines.append("Fields:")
    for f in cfg.field_defs:
        scope = ", ".join(f.get("classes") or []) or "any class"
        lines.append(f"- {f['name']} (applies to: {scope}): {f.get('description', '')}")
    return "\n".join(lines)


def build_messages(cfg: Config, pages: list[str], images: list[bytes], max_text_chars: int) -> list[dict]:
    content: list[dict] = []
    budget = max_text_chars
    parts = []
    for i, page in enumerate(pages):
        if budget <= 0:
            break
        chunk = page[:budget]
        budget -= len(chunk)
        parts.append(f"--- page {i + 1} ---\n{chunk}")
    if parts:
        content.append({
            "type": "text",
            "text": "Page text read from the document by OCR or from its text layer:\n\n"
                    + "\n\n".join(parts),
        })
    for img in images:
        url = "data:image/png;base64," + base64.b64encode(img).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": _system_text(cfg)},
        {"role": "user", "content": content},
    ]


def parse_answer(content: str, cfg: Config) -> VLMAnswer:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise VLMOutputError(f"answer is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise VLMOutputError("answer is not an object")
    extra = set(data) - {"class", "fields"}
    if extra:
        raise VLMOutputError(f"answer has unexpected keys: {sorted(extra)}")

    label = data.get("class")
    allowed = set(cfg.class_names) | {_unknown(cfg)}
    if not isinstance(label, str) or label not in allowed:
        raise VLMOutputError(f"class {label!r} is not a declared class")

    raw = data.get("fields", {})
    if not isinstance(raw, dict):
        raise VLMOutputError("fields is not an object")
    undeclared = set(raw) - set(_field_names(cfg))
    if undeclared:
        raise VLMOutputError(f"undeclared fields: {sorted(undeclared)}")

    fields: dict[str, list[str]] = {}
    for name, values in raw.items():
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise VLMOutputError(f"{name}: expected a list of strings")
        kept = list(dict.fromkeys(v.strip() for v in values if v.strip()))
        if kept:
            fields[name] = kept
    return VLMAnswer(label=label, fields=fields)
```

- [ ] **Step 5: Run the tests and the full suite**

Run: `python -m pytest tests/test_cascade_prompt.py -v`
Expected: PASS, 17 tests.

Run: `python -m pytest -q`
Expected: all pass. The workbook tests still pass with descriptions now filled in.

- [ ] **Step 6: Commit**

```bash
git add config/fields.yaml cascade/prompt.py tests/test_cascade_prompt.py
git commit -m "Build the model's schema and prompt from config, and parse its answer strictly"
```

---

### Task 6: Page images

**Files:**
- Create: `cascade/render.py`
- Modify: `docs/superpowers/specs/2026-10-05-cascade-v2-vlm-grounding-design.md` (the "Page images" TIFF line)
- Test: `tests/test_cascade_render.py`

**Interfaces:**
- Consumes: `baseline.extract.PDF_EXT`, `baseline.extract.IMAGE_EXT`.
- Produces: `render_pages(path: Path, max_pages: int, dpi: int, max_long_edge: int) -> list[bytes]`: PNG bytes, RGB, long edge at most `max_long_edge`, one per page for PDFs (first `max_pages`), one for an image file (frame 0, the same frame the extractor reads), none for any other format.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cascade_render.py`:

```python
import io

from PIL import Image


def _open(png: bytes) -> Image.Image:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return Image.open(io.BytesIO(png))


def test_pdf_renders_one_image_per_page_up_to_the_limit(corpus):
    from cascade.render import render_pages

    pages = render_pages(corpus / "invoice_native.pdf", max_pages=4, dpi=72, max_long_edge=5000)
    assert len(pages) == 2
    assert render_pages(corpus / "invoice_native.pdf", max_pages=1, dpi=72, max_long_edge=5000)[0] == pages[0]


def test_long_edge_is_capped(corpus):
    from cascade.render import render_pages

    img = _open(render_pages(corpus / "invoice_native.pdf", max_pages=1, dpi=300, max_long_edge=500)[0])
    assert max(img.size) == 500


def test_an_image_file_gives_one_rgb_png(tmp_path):
    from cascade.render import render_pages

    path = tmp_path / "scan.jpg"
    Image.new("CMYK", (120, 80)).save(path)
    pages = render_pages(path, max_pages=4, dpi=150, max_long_edge=1600)
    assert len(pages) == 1
    img = _open(pages[0])
    assert img.mode == "RGB" and img.size == (120, 80)


def test_formats_without_a_page_image_give_none(corpus):
    from cascade.render import render_pages

    assert render_pages(corpus / "salary_slip.docx", 4, 150, 1600) == []
    assert render_pages(corpus / "contract.txt", 4, 150, 1600) == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_cascade_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cascade.render'`.

- [ ] **Step 3: Implement rendering**

Create `cascade/render.py`:

```python
"""Page images for the model, one per page the extractor read.

An image file gives frame 0 only, because that is the only frame the
baseline extractor reads, and the model must see the page the text came from.
"""

from __future__ import annotations

import io
from pathlib import Path

from baseline.extract import IMAGE_EXT, PDF_EXT


def _png(img, max_long_edge: int) -> bytes:
    from PIL import Image

    img = img.convert("RGB")
    long_edge = max(img.size)
    if long_edge > max_long_edge:
        k = max_long_edge / long_edge
        size = (max(1, round(img.width * k)), max(1, round(img.height * k)))
        img = img.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pdf(path: Path, max_pages: int, dpi: int, max_long_edge: int) -> list[bytes]:
    import fitz
    from PIL import Image

    out = []
    with fitz.open(path) as doc:
        for i in range(min(doc.page_count, max_pages)):
            pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            out.append(_png(img, max_long_edge))
    return out


def render_pages(path: Path, max_pages: int, dpi: int, max_long_edge: int) -> list[bytes]:
    ext = Path(path).suffix.lower()
    if ext in PDF_EXT:
        return _pdf(Path(path), max_pages, dpi, max_long_edge)
    if ext in IMAGE_EXT:
        from PIL import Image

        with Image.open(path) as img:
            return [_png(img, max_long_edge)]
    return []
```

- [ ] **Step 4: Correct the spec to match the extractor**

In `docs/superpowers/specs/2026-10-05-cascade-v2-vlm-grounding-design.md`, replace:

```
- Image files are read directly. Multi frame TIFFs yield one image per frame,
  in the same order the extractor reads them.
```

with:

```
- Image files are read directly, frame 0 only. That is the only frame the
  baseline extractor reads, and the model must see the page the text came
  from. Multi frame TIFFs need an extractor change first.
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_cascade_render.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add cascade/render.py tests/test_cascade_render.py docs/superpowers/specs/2026-10-05-cascade-v2-vlm-grounding-design.md
git commit -m "Render page images for the model"
```

---

### Task 7: Local only VLM client

**Files:**
- Create: `cascade/vlm.py`, `tests/vlm_stub.py`
- Test: `tests/test_cascade_vlm.py`

**Interfaces:**
- Consumes: `cascade.config.VLMSettings` (Task 3).
- Produces:
  - `VLMError(RuntimeError)`.
  - `VLMReply` (frozen dataclass): `content: str`, `model: str`, `finish_reason: str | None`.
  - `Backend` (`typing.Protocol`): `complete(messages: list[dict], schema: dict) -> VLMReply`.
  - `check_local(url: str, allowed_hosts: list[str]) -> None`: raises `ValueError` for a non http(s) scheme or a host that is not loopback or listed.
  - `VLMClient(settings: VLMSettings, sleep: Callable[[float], None] = time.sleep)`, which implements `Backend`.
  - `tests/vlm_stub.py`: `serve(responses)` context manager yielding `(base_url, requests)`, and `chat(content: str, model: str = "test-model", finish: str = "stop") -> dict`.

- [ ] **Step 1: Write the stub server**

Create `tests/vlm_stub.py`:

```python
"""A local HTTP server that answers like an OpenAI compatible chat endpoint.

Each entry in `responses` is (status, body, delay_s, headers). Request n gets
entry n, and the last entry repeats.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def chat(content: str, model: str = "test-model", finish: str = "stop") -> dict:
    return {"model": model, "choices": [
        {"message": {"role": "assistant", "content": content}, "finish_reason": finish}
    ]}


@contextmanager
def serve(responses):
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            requests.append({
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(self.rfile.read(n) or b"{}"),
            })
            status, body, delay, headers = responses[min(len(requests), len(responses)) - 1]
            if delay:
                time.sleep(delay)
            payload = (body if isinstance(body, str) else json.dumps(body)).encode("utf-8")
            try:
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the client timed out and hung up, which some tests intend

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_cascade_vlm.py`:

```python
import pytest

from vlm_stub import chat, serve

SCHEMA = {"type": "object"}
MESSAGES = [{"role": "user", "content": "hi"}]


def _settings(url, **kw):
    from cascade.config import VLMSettings

    base = dict(base_url=url, model="m", timeout_s=2, retries=2, backoff_s=0.5, max_tokens=100)
    base.update(kw)
    return VLMSettings(**base)


def _client(url, slept=None, **kw):
    from cascade.vlm import VLMClient

    return VLMClient(_settings(url, **kw), sleep=(slept.append if slept is not None else lambda s: None))


def test_returns_content_and_sends_a_deterministic_schema_bound_request():
    with serve([(200, chat('{"a": 1}', model="served-model"), 0, None)]) as (url, reqs):
        reply = _client(url).complete(MESSAGES, SCHEMA)
    assert (reply.content, reply.model, reply.finish_reason) == ('{"a": 1}', "served-model", "stop")
    body = reqs[0]["body"]
    assert reqs[0]["path"] == "/v1/chat/completions"
    assert body["model"] == "m" and body["temperature"] == 0 and body["max_tokens"] == 100
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA


def test_retries_a_503_with_backoff_then_succeeds():
    slept = []
    with serve([(503, "{}", 0, None), (503, "{}", 0, None), (200, chat("{}"), 0, None)]) as (url, reqs):
        reply = _client(url, slept).complete(MESSAGES, SCHEMA)
    assert reply.content == "{}"
    assert len(reqs) == 3
    assert slept == [0.5, 1.0]


def test_gives_up_after_the_configured_retries():
    from cascade.vlm import VLMError

    with serve([(500, "{}", 0, None)]) as (url, reqs):
        with pytest.raises(VLMError, match="gave up after 3 attempts"):
            _client(url).complete(MESSAGES, SCHEMA)
    assert len(reqs) == 3


def test_a_400_is_not_retried():
    from cascade.vlm import VLMError

    with serve([(400, '{"error": "bad"}', 0, None)]) as (url, reqs):
        with pytest.raises(VLMError, match="HTTP 400"):
            _client(url).complete(MESSAGES, SCHEMA)
    assert len(reqs) == 1


def test_a_redirect_is_never_followed():
    from cascade.vlm import VLMError

    redirect = (302, "{}", 0, {"Location": "http://example.com/v1/chat/completions"})
    with serve([redirect]) as (url, reqs):
        with pytest.raises(VLMError, match="HTTP 302"):
            _client(url).complete(MESSAGES, SCHEMA)
    assert len(reqs) == 1


def test_a_timeout_is_retried_then_reported():
    from cascade.vlm import VLMError

    with serve([(200, chat("{}"), 1.0, None)]) as (url, reqs):
        with pytest.raises(VLMError, match="gave up"):
            _client(url, timeout_s=0.2, retries=1).complete(MESSAGES, SCHEMA)
    assert len(reqs) == 2


def test_an_unexpected_response_shape_is_an_error():
    from cascade.vlm import VLMError

    with serve([(200, {"choices": []}, 0, None)]) as (url, _):
        with pytest.raises(VLMError, match="unexpected response shape"):
            _client(url).complete(MESSAGES, SCHEMA)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/v1", "http://localhost:11434/v1", "http://[::1]:8000/v1",
])
def test_loopback_hosts_are_accepted(url):
    from cascade.vlm import check_local

    check_local(url, [])


@pytest.mark.parametrize("url", [
    "http://10.0.0.5:8000/v1", "https://api.example.com/v1", "file:///etc/passwd", "http:///v1",
])
def test_non_local_hosts_are_refused(url):
    from cascade.vlm import check_local

    with pytest.raises(ValueError):
        check_local(url, [])


def test_a_listed_host_is_accepted():
    from cascade.vlm import check_local

    check_local("http://gpu-box.internal:8000/v1", ["gpu-box.internal"])


def test_client_refuses_to_start_without_a_model_or_on_a_remote_host():
    from cascade.vlm import VLMClient

    with pytest.raises(ValueError, match="vlm.model is not set"):
        VLMClient(_settings("http://127.0.0.1:1/v1", model=None))
    with pytest.raises(ValueError, match="not loopback"):
        VLMClient(_settings("http://10.0.0.5:8000/v1"))


def test_api_key_is_read_from_the_named_environment_variable(monkeypatch):
    from cascade.vlm import VLMClient

    monkeypatch.delenv("CASCADE_TEST_KEY", raising=False)
    with pytest.raises(ValueError, match="CASCADE_TEST_KEY"):
        VLMClient(_settings("http://127.0.0.1:1/v1", api_key_env="CASCADE_TEST_KEY"))
    monkeypatch.setenv("CASCADE_TEST_KEY", "secret")
    with serve([(200, chat("{}"), 0, None)]) as (url, reqs):
        _client(url, api_key_env="CASCADE_TEST_KEY").complete(MESSAGES, SCHEMA)
    assert reqs[0]["headers"]["Authorization"] == "Bearer secret"
```

- [ ] **Step 3: Run them to verify they fail**

Run: `python -m pytest tests/test_cascade_vlm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cascade.vlm'`.

- [ ] **Step 4: Implement the client**

Create `cascade/vlm.py`:

```python
"""Client for a local OpenAI compatible chat completions server.

vLLM serves this on a GPU box, Ollama on a laptop. Document content must not
leave the company, so the client:

- refuses any host that is not loopback or listed in vlm.allowed_hosts;
- never follows a redirect, which could point anywhere;
- ignores proxy environment variables, which could route anywhere.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.parse import urlparse

from .config import VLMSettings

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class VLMError(RuntimeError):
    """The server could not give an answer."""


@dataclass(frozen=True)
class VLMReply:
    content: str
    model: str
    finish_reason: str | None


class Backend(Protocol):
    def complete(self, messages: list[dict], schema: dict) -> VLMReply: ...


def check_local(url: str, allowed_hosts: list[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"VLM url must be http or https, got {url!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"VLM url has no host: {url!r}")
    if host == "localhost" or host in allowed_hosts:
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError(f"VLM host {host!r} is not loopback and not in vlm.allowed_hosts")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib then raises HTTPError with the 3xx status


def _reply(payload: dict) -> VLMReply:
    try:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VLMError(f"unexpected response shape: {exc!r}") from exc
    if not isinstance(content, str):
        raise VLMError("message content is not a string")
    return VLMReply(content=content, model=str(payload.get("model", "")),
                    finish_reason=choice.get("finish_reason"))


class VLMClient:
    def __init__(self, settings: VLMSettings, sleep: Callable[[float], None] = time.sleep):
        if not settings.model:
            raise ValueError("vlm.model is not set: set it in config/cascade.yaml or pass --model")
        check_local(settings.base_url, settings.allowed_hosts)
        self._key = None
        if settings.api_key_env:
            self._key = os.environ.get(settings.api_key_env)
            if not self._key:
                raise ValueError(f"environment variable {settings.api_key_env} is not set")
        self._s = settings
        self._sleep = sleep
        self._url = settings.base_url.rstrip("/") + "/chat/completions"
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def complete(self, messages: list[dict], schema: dict) -> VLMReply:
        body = json.dumps({
            "model": self._s.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self._s.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "document", "schema": schema, "strict": True},
            },
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"

        attempts = self._s.retries + 1
        last = ""
        for attempt in range(attempts):
            if attempt:
                self._sleep(self._s.backoff_s * 2 ** (attempt - 1))
            req = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
            try:
                with self._opener.open(req, timeout=self._s.timeout_s) as resp:
                    raw = resp.read()
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}"
                if exc.code in _RETRY_STATUS:
                    continue
                raise VLMError(last) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
                last = f"{type(exc).__name__}: {exc}"
                continue
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise VLMError(f"server returned non JSON: {exc}") from exc
            return _reply(payload)
        raise VLMError(f"gave up after {attempts} attempts: {last}")
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_cascade_vlm.py -v`
Expected: PASS, 17 tests.

- [ ] **Step 6: Commit**

```bash
git add cascade/vlm.py tests/vlm_stub.py tests/test_cascade_vlm.py
git commit -m "Add a local only client for an OpenAI compatible model server"
```

---

### Task 8: Merge and decide

**Files:**
- Create: `cascade/merge.py`
- Test: `tests/test_cascade_merge.py`

**Interfaces:**
- Consumes: `TextIndex`, `ground` (Task 4); `VLMAnswer` (Task 5); `ConfidenceTiers` (Task 3); `candidates_for` (Task 2); `MetadataMethod.VLM`, `VLM_RULE_AGREE`, `ClassifyMethod.VLM` (Task 1).
- Produces:
  - `Decision` (dataclass): `classification: Classification`, `metadata: list[MetadataField]`, `class_provenance: dict`, `field_provenance: list[dict]`.
  - `decide(full: str, pages: list[str], cfg: Config, tiers: ConfidenceTiers, rules: dict[str, float], answer: VLMAnswer, baseline_meta: list[MetadataField], words: list[WordBox] | None = None, page_sizes: list[tuple[float, float]] | None = None) -> Decision`.
  - Field provenance item keys: `field`, `vlm_value`, `outcome` (`exact`, `loose`, `normalised`, `ungrounded`, `checksum_failed`), `n_matches`, `rule_agrees`, `char_start`, `char_end`, `emitted`; or for `rule_only`: `field`, `outcome`, `baseline_values`, `emitted` (always false).
  - Class provenance keys: `source` (`rule` or `vlm`), `label`, `vlm_label`, `disagrees`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cascade_merge.py`:

```python
from baseline.classify import rule_scores
from baseline.metadata import extract_fields
from baseline.schema import ClassifyMethod, MetadataMethod

TEXT = (
    "TAX INVOICE\n"
    "Sold By: Acme Ltd\n"
    "PAN: AAPFU0939F\n"
    "Bad PAN: ABCDE1234F\n"
    "Email: billing@acme.example\n"
    "Invoice No: INV-1\n"
    "Ref INV-2\n"
    "Grand Total: Rs. 11,800.00\n"
)


def _tiers():
    from cascade.config import ConfidenceTiers

    return ConfidenceTiers(agree=0.9, grounded=0.75, ambiguous=0.5, vlm_class=0.8)


def _decide(cfg, fields, label="invoice", text=TEXT, rules=None):
    from cascade.merge import decide
    from cascade.prompt import VLMAnswer

    pages = [text]
    base, _ = extract_fields(text, pages, cfg, label)
    rules = rule_scores(text, pages, cfg) if rules is None else rules
    return decide(text, pages, cfg, _tiers(), rules, VLMAnswer(label=label, fields=fields), base)


def _by_field(decision):
    out = {}
    for m in decision.metadata:
        out.setdefault(m.field, []).append(m)
    return out


def _prov(decision, field):
    return [p for p in decision.field_provenance if p["field"] == field]


def test_a_proven_id_survives_a_model_that_omits_it(cfg):
    d = _decide(cfg, {})
    pan = _by_field(d)["pan"]
    assert [(m.value, m.method, m.validated) for m in pan] == [
        ("AAPFU0939F", MetadataMethod.REGEX_CHECKSUM, True)
    ]


def test_a_model_id_that_fails_its_checksum_is_dropped(cfg):
    d = _decide(cfg, {"pan": ["ABCDE1234F"]})
    assert [m.value for m in _by_field(d)["pan"]] == ["AAPFU0939F"]
    assert [p["outcome"] for p in _prov(d, "pan")] == ["checksum_failed"]


def test_validated_is_never_true_without_a_checksum(cfg):
    d = _decide(cfg, {"email": ["billing@acme.example"], "invoice_number": ["INV-1"],
                      "total_amount": ["11800"]})
    for m in d.metadata:
        assert m.validated == (m.method is MetadataMethod.REGEX_CHECKSUM)
    assert _by_field(d)["email"][0].method in (MetadataMethod.VLM, MetadataMethod.VLM_RULE_AGREE)


def test_a_value_a_rule_also_found_is_marked_as_agreement(cfg):
    d = _decide(cfg, {"invoice_number": ["INV-1"]})
    (m,) = _by_field(d)["invoice_number"]
    assert (m.value, m.method, m.confidence) == ("INV-1", MetadataMethod.VLM_RULE_AGREE, 0.9)


def test_a_non_multi_field_keeps_only_the_first_grounded_value(cfg):
    d = _decide(cfg, {"invoice_number": ["NOT-THERE-9", "INV-2", "INV-1"]})
    assert [m.value for m in _by_field(d)["invoice_number"]] == ["INV-2"]
    assert [p["outcome"] for p in _prov(d, "invoice_number")] == ["ungrounded", "exact"]


def test_a_reformatted_amount_is_emitted_as_printed_and_normalised(cfg):
    d = _decide(cfg, {"total_amount": ["11800"]})
    (m,) = _by_field(d)["total_amount"]
    assert m.value == "11,800.00" and m.normalized_value == "11800.00"
    assert TEXT[m.char_start : m.char_end] == m.value


def test_a_rule_value_the_model_did_not_return_is_not_emitted_and_is_recorded(cfg):
    d = _decide(cfg, {})
    assert "invoice_number" not in _by_field(d)
    (p,) = _prov(d, "invoice_number")
    assert p["outcome"] == "rule_only" and p["baseline_values"] == ["INV-1"]


def test_an_invented_value_is_not_emitted(cfg):
    d = _decide(cfg, {"buyer_name": ["Hallucinated Holdings"]})
    assert "buyer_name" not in _by_field(d)
    assert [p["outcome"] for p in _prov(d, "buyer_name")] == ["ungrounded"]


def test_a_field_out_of_scope_for_the_decided_class_is_not_emitted(cfg):
    text = TEXT + "Dated: 14/03/2024\n"
    d = _decide(cfg, {"judgment_date": ["14/03/2024"]}, text=text)
    assert d.classification.label == "invoice"
    assert "judgment_date" not in _by_field(d)


def test_a_short_circuit_rule_decides_the_class_over_the_model(cfg):
    d = _decide(cfg, {}, label="receipt", rules={"invoice": 0.95})
    c = d.classification
    assert (c.label, c.method, c.confidence) == ("invoice", ClassifyMethod.RULE, 0.95)
    assert d.class_provenance == {"source": "rule", "label": "invoice",
                                  "vlm_label": "receipt", "disagrees": True}


def test_below_the_short_circuit_the_model_decides(cfg):
    d = _decide(cfg, {}, label="receipt", rules={"invoice": 0.6})
    c = d.classification
    assert (c.label, c.method, c.confidence) == ("receipt", ClassifyMethod.VLM, 0.8)
    assert c.all_scores == {"invoice": 0.6}


def test_a_model_abstention_is_unknown_with_zero_confidence(cfg):
    d = _decide(cfg, {}, label="unknown", rules={})
    c = d.classification
    assert (c.label, c.method, c.confidence) == ("unknown", ClassifyMethod.VLM, 0.0)


def test_metadata_is_sorted_and_every_offset_slices_back(cfg):
    d = _decide(cfg, {"email": ["billing@acme.example"], "invoice_number": ["INV-1"],
                      "total_amount": ["11800"], "vendor_name": ["Acme Ltd"]})
    keys = [(m.char_start, m.field) for m in d.metadata]
    assert keys == sorted(keys)
    for m in d.metadata:
        assert TEXT[m.char_start : m.char_end] == m.value
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_cascade_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cascade.merge'`.

- [ ] **Step 3: Implement merge**

Create `cascade/merge.py`:

```python
"""Turn a grounded model answer into the record's class and fields.

The rule and checksum layer is a hard prior over the model:

- a value whose checksum passed is never removed or replaced, and a model
  value for a checksummed field is emitted only if it passes the same check;
- `validated` is true only when a checksum passed, never on a format check;
- a rule at short circuit specificity decides the class.

A rule value the model did not confirm is not emitted. Anchor rules are the
false positive problem this approach exists to fix, and every such drop is
recorded as `rule_only` so the choice is measured, not argued.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from baseline.config import Config
from baseline.metadata import CHECKSUM_VALIDATORS, NormCtx, normalize, run_validator
from baseline.metadata.fields import candidates_for
from baseline.schema import (
    Classification,
    ClassifyMethod,
    MetadataField,
    MetadataMethod,
    WordBox,
    page_for_offset,
)

from .config import ConfidenceTiers
from .ground import TextIndex, ground
from .prompt import VLMAnswer


@dataclass
class Decision:
    classification: Classification
    metadata: list[MetadataField]
    class_provenance: dict
    field_provenance: list[dict] = field(default_factory=list)


def _unknown(cfg: Config) -> str:
    return str(cfg.classify_opts().get("unknown_label", "unknown"))


def decide_class(
    rules: dict[str, float], answer: VLMAnswer, cfg: Config, tiers: ConfidenceTiers
) -> tuple[Classification, dict]:
    unknown = _unknown(cfg)
    short_circuit = float(cfg.classify_opts().get("rule_short_circuit", 0.9))
    all_scores = dict(sorted(rules.items()))
    if rules:
        # Ties break on name so the output is stable, as in the baseline.
        label, score = sorted(rules.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if score >= short_circuit:
            prov = {"source": "rule", "label": label, "vlm_label": answer.label,
                    "disagrees": answer.label not in (label, unknown)}
            return Classification(label=label, confidence=round(score, 4),
                                  all_scores=all_scores, method=ClassifyMethod.RULE), prov
    confidence = 0.0 if answer.label == unknown else tiers.vlm_class
    prov = {"source": "vlm", "label": answer.label, "vlm_label": answer.label, "disagrees": False}
    return Classification(label=answer.label, confidence=confidence,
                          all_scores=all_scores, method=ClassifyMethod.VLM), prov


def _in_scope(fdef: dict, doc_class: str, cfg: Config) -> bool:
    """The same scoping rule extract_fields applies."""
    scope = fdef.get("classes")
    if not scope or doc_class in scope:
        return True
    run_all = bool(cfg.metadata_opts().get("run_scoped_fields_when_unknown", True))
    return doc_class in ("", _unknown(cfg)) and run_all


def _key(normalized: str | None, raw: str) -> str:
    return normalized if normalized is not None else raw


def decide_fields(
    full: str,
    pages: list[str],
    cfg: Config,
    tiers: ConfidenceTiers,
    doc_class: str,
    answer: VLMAnswer,
    baseline_meta: list[MetadataField],
    words: list[WordBox] | None = None,
    page_sizes: list[tuple[float, float]] | None = None,
) -> tuple[list[MetadataField], list[dict]]:
    norm_opts = cfg.fields.get("normalizer_options", {}) or {}
    index = TextIndex(full, pages)
    from_baseline: dict[str, list[MetadataField]] = defaultdict(list)
    for m in baseline_meta:
        from_baseline[m.field].append(m)

    out: list[MetadataField] = []
    prov: list[dict] = []
    for fdef in cfg.field_defs:
        name = fdef.get("name")
        if not name or not _in_scope(fdef, doc_class, cfg):
            continue
        pattern = fdef.get("pattern") or {}
        checksummed = pattern.get("validator") in CHECKSUM_VALIDATORS
        multi = bool(fdef.get("multi"))

        kept: list[MetadataField] = []
        seen: set[str] = set()
        if checksummed:
            for m in from_baseline[name]:
                if m.method is MetadataMethod.REGEX_CHECKSUM:
                    kept.append(m)
                    seen.add(_key(m.normalized_value, m.value))

        rule_cands = candidates_for(full, pages, fdef, cfg, doc_class, words, page_sizes)
        model_emitted = False
        for value in answer.fields.get(name, []):
            if kept and not multi:
                break
            g = ground(value, index, fdef, rule_cands, norm_opts)
            item = {"field": name, "vlm_value": value, "outcome": g.outcome,
                    "n_matches": g.n_matches, "rule_agrees": g.rule_agrees,
                    "char_start": g.start, "char_end": g.end, "emitted": False}
            prov.append(item)
            if not g.grounded:
                continue
            raw = full[g.start : g.end]
            if checksummed:
                if not run_validator(pattern.get("validator"), raw, pattern.get("validator_args") or {}):
                    item["outcome"] = "checksum_failed"
                    continue
                method, confidence, validated = MetadataMethod.REGEX_CHECKSUM, 1.0, True
            elif g.rule_agrees:
                method, confidence, validated = MetadataMethod.VLM_RULE_AGREE, tiers.agree, False
            else:
                method, validated = MetadataMethod.VLM, False
                confidence = tiers.ambiguous if g.n_matches > 1 else tiers.grounded
            normalized, extras = normalize(
                fdef.get("normalizer"), raw, NormCtx(full=full, start=g.start, end=g.end, opts=norm_opts)
            )
            key = _key(normalized, raw)
            if key in seen:
                continue
            seen.add(key)
            item["emitted"] = True
            model_emitted = True
            kept.append(MetadataField(
                field=name,
                value=raw,
                normalized_value=normalized,
                page=page_for_offset(index.spans, g.start),
                char_start=g.start,
                char_end=g.end,
                confidence=round(confidence, 4),
                method=method,
                validated=validated,
                normalizer_version=cfg.normalizer_version,
                currency=extras.get("currency"),
            ))

        if not checksummed and not model_emitted and from_baseline[name]:
            prov.append({"field": name, "outcome": "rule_only",
                         "baseline_values": [m.value for m in from_baseline[name]], "emitted": False})
        out.extend(kept)

    out.sort(key=lambda m: (m.char_start, m.field))
    return out, prov


def decide(
    full: str,
    pages: list[str],
    cfg: Config,
    tiers: ConfidenceTiers,
    rules: dict[str, float],
    answer: VLMAnswer,
    baseline_meta: list[MetadataField],
    words: list[WordBox] | None = None,
    page_sizes: list[tuple[float, float]] | None = None,
) -> Decision:
    classification, class_prov = decide_class(rules, answer, cfg, tiers)
    metadata, field_prov = decide_fields(
        full, pages, cfg, tiers, classification.label, answer, baseline_meta, words, page_sizes
    )
    return Decision(classification, metadata, class_prov, field_prov)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_cascade_merge.py -v`
Expected: PASS, 13 tests.

If `test_a_field_out_of_scope_for_the_decided_class_is_not_emitted` fails because `rule_scores` does not reach the short circuit on `TEXT`, check `rule_scores(TEXT, [TEXT], cfg)`. `tax invoice` on the first page scores 0.95 in `classes.yaml`, so it should.

- [ ] **Step 5: Commit**

```bash
git add cascade/merge.py tests/test_cascade_merge.py
git commit -m "Merge the model's answer under the rule and checksum layer"
```

---

### Task 9: Runner: one document, a batch, and the provenance sidecar

**Files:**
- Create: `cascade/runner.py`
- Test: `tests/test_cascade_runner.py`

**Interfaces:**
- Consumes: `baseline.pipeline.process_document`, `Models`, `dumps`, `existing_doc_ids`, `write_sidecar`, `tag_facts` (Task 1); `baseline.classify.rule_scores`; `baseline.tagging.apply_tags`; `baseline.scan.scan_documents`; `render_pages` (Task 6); `build_messages`, `response_schema`, `parse_answer`, `check_descriptions` (Task 5); `Backend` (Task 7); `decide` (Task 8); `CascadeConfig` (Task 3).
- Produces:
  - `Result` (dataclass): `record: DocumentRecord`, `sidecar: BBoxSidecar`, `provenance: dict`.
  - `process_document(path: str | Path, cfg: Config, ccfg: CascadeConfig, backend: Backend, models: Models | None = None) -> Result`.
  - `run_batch(input_dir: str | Path, output: str | Path, cfg: Config, ccfg: CascadeConfig, backend: Backend, models: Models | None = None, bbox_dir: str | Path | None = None, resume: bool = True) -> dict` returning `found`, `skipped`, `written`, `failed`, `fallback`, `excluded`.
  - The provenance sidecar is `<output>.provenance`, one JSON line per written record, in the same order.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cascade_runner.py`:

```python
import json

import pytest

from baseline.pipeline import process_document as baseline_process
from baseline.schema import ClassifyMethod, MetadataMethod

INVOICE_ANSWER = {"class": "invoice", "fields": {
    "invoice_number": ["INV-2024-0042"],
    "invoice_date": ["2024-03-14"],                # model reformatted it
    "total_amount": ["11800.00"],                  # printed as 11,800.00
    "vendor_name": ["Acme Traders Private Limited"],  # printed twice
    "buyer_name": ["Hallucinated Holdings"],       # not on the document
    "gstin": ["27AAPFU0939F1ZV"],
    "pan": ["AAPFU0939F"],
}}
NOTHING = {"class": "unknown", "fields": {}}


class FakeBackend:
    """Answers from the page text it is sent. No GPU, no network."""

    def __init__(self, answer_for):
        self.answer_for = answer_for
        self.calls = 0

    def complete(self, messages, schema):
        from cascade.vlm import VLMReply

        self.calls += 1
        text = "".join(p.get("text", "") for p in messages[1]["content"])
        return VLMReply(content=json.dumps(self.answer_for(text)), model="fake", finish_reason="stop")


class DownBackend:
    def complete(self, messages, schema):
        from cascade.vlm import VLMError

        raise VLMError("server down")


def _invoice_or_nothing(text):
    return INVOICE_ANSWER if "INV-2024-0042" in text else NOTHING


@pytest.fixture(scope="module")
def ccfg():
    from cascade.config import load_cascade_config

    from conftest import ROOT
    return load_cascade_config(ROOT / "config")


def test_invoice_end_to_end(corpus, cfg, ccfg):
    from cascade.runner import process_document

    r = process_document(corpus / "invoice_native.pdf", cfg, ccfg, FakeBackend(_invoice_or_nothing))
    rec, prov = r.record, r.provenance
    assert rec.verify_offsets() == []
    assert rec.classification.label == "invoice" and rec.classification.method is ClassifyMethod.RULE
    by = {m.field: m for m in rec.metadata}

    assert (by["invoice_number"].value, by["invoice_number"].method) == (
        "INV-2024-0042", MetadataMethod.VLM_RULE_AGREE)
    assert (by["invoice_date"].value, by["invoice_date"].normalized_value) == ("14/03/2024", "2024-03-14")
    assert (by["total_amount"].value, by["total_amount"].normalized_value) == ("11,800.00", "11800.00")

    base, _ = baseline_process(corpus / "invoice_native.pdf", cfg)
    base_vendor = next(m for m in base.metadata if m.field == "vendor_name")
    assert by["vendor_name"].char_start == base_vendor.char_start  # the rule's occurrence

    for proven in ("gstin", "pan", "ifsc"):  # ifsc kept although the model omitted it
        assert by[proven].method is MetadataMethod.REGEX_CHECKSUM and by[proven].validated
    assert "buyer_name" not in by
    assert "tax_amount" not in by  # rule only, the model did not confirm it
    for m in rec.metadata:
        assert m.validated == (m.method is MetadataMethod.REGEX_CHECKSUM)

    outcomes = {(p["field"], p["outcome"]) for p in prov["fields"]}
    assert ("buyer_name", "ungrounded") in outcomes
    assert ("tax_amount", "rule_only") in outcomes
    assert prov["fallback"] is False and prov["model"] == "fake"
    assert prov["class"]["source"] == "rule"
    assert rec.schema_version == "1.2.0"


def test_a_down_server_falls_back_to_the_baseline_answer(corpus, cfg, ccfg):
    from cascade.runner import process_document

    r = process_document(corpus / "invoice_native.pdf", cfg, ccfg, DownBackend())
    base, _ = baseline_process(corpus / "invoice_native.pdf", cfg)
    assert r.record.classification == base.classification
    assert r.record.metadata == base.metadata
    assert r.record.tags == base.tags
    assert any(e.startswith("cascade: VLMError: server down") for e in r.record.diagnostics.errors)
    assert r.provenance["fallback"] is True


def test_a_malformed_answer_falls_back_too(corpus, cfg, ccfg):
    from cascade.runner import process_document

    r = process_document(corpus / "invoice_native.pdf", cfg, ccfg,
                         FakeBackend(lambda text: {"class": "not_a_class", "fields": {}}))
    assert r.provenance["fallback"] is True
    assert any("VLMOutputError" in e for e in r.record.diagnostics.errors)


def test_a_document_with_no_image_is_sent_as_text_only(corpus, cfg, ccfg):
    from cascade.runner import process_document

    backend = FakeBackend(lambda text: NOTHING)
    r = process_document(corpus / "contract.txt", cfg, ccfg, backend)
    assert backend.calls == 1
    assert r.provenance["images_sent"] == 0 and r.provenance["pages_sent"] == 1


def test_batch_writes_one_record_and_one_provenance_line_per_document(corpus, cfg, ccfg, tmp_path):
    from cascade.runner import run_batch

    out = tmp_path / "cascade.jsonl"
    stats = run_batch(corpus, out, cfg, ccfg, FakeBackend(_invoice_or_nothing))
    assert stats["written"] + stats["failed"] == stats["found"]
    records = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    provs = [json.loads(l) for l in (tmp_path / "cascade.jsonl.provenance").read_text(encoding="utf-8").splitlines()]
    assert len(records) == stats["written"] == len(provs)
    assert [r["doc_id"] for r in records] == [p["doc_id"] for p in provs]

    from baseline.schema import DocumentRecord
    for r in records:
        assert DocumentRecord.model_validate(r).verify_offsets() == []

    again = run_batch(corpus, out, cfg, ccfg, FakeBackend(_invoice_or_nothing))
    assert again["written"] == 0 and again["skipped"] == stats["written"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_cascade_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cascade.runner'`.

- [ ] **Step 3: Implement the runner**

Create `cascade/runner.py`:

```python
"""One document in, one record out, through the model and the grounding step.

The baseline runs first and its record is the fallback. If the model cannot
be reached, or its answer is not the shape asked for, the document gets the
baseline's answer and an error saying so, never an empty record.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from baseline import pipeline as baseline_pipeline
from baseline.classify import rule_scores
from baseline.config import Config
from baseline.logging_setup import get_logger
from baseline.scan import scan_documents
from baseline.schema import BBoxSidecar, Diagnostics, DocumentRecord, MetadataField, compute_doc_id
from baseline.tagging import apply_tags

from .config import CascadeConfig
from .merge import decide
from .prompt import build_messages, check_descriptions, parse_answer, response_schema
from .render import render_pages
from .vlm import Backend

log = get_logger(__name__)


@dataclass
class Result:
    record: DocumentRecord
    sidecar: BBoxSidecar
    provenance: dict


def _diagnostics(base: DocumentRecord, timings: dict[str, float], errors: list[str]) -> Diagnostics:
    d = base.diagnostics
    return Diagnostics(
        page_count=d.page_count,
        is_scanned=d.is_scanned,
        has_native_text=d.has_native_text,
        timings_ms={**d.timings_ms, **timings},
        errors=[*d.errors, *errors],
    )


def _record(base: DocumentRecord, classification, metadata, tags, diagnostics) -> DocumentRecord:
    return DocumentRecord(
        doc_id=base.doc_id,
        source_path=base.source_path,
        filename=base.filename,
        text=base.text,
        classification=classification,
        metadata=metadata,
        tags=tags,
        diagnostics=diagnostics,
    )


def _drop_bad_offsets(meta: list[MetadataField], full: str) -> tuple[list[MetadataField], list[str]]:
    """Belt and braces. Grounding guarantees this, and the write checks again."""
    keep, problems = [], []
    for m in meta:
        if m.char_end <= len(full) and full[m.char_start : m.char_end] == m.value:
            keep.append(m)
        else:
            problems.append(f"{m.field}: text[{m.char_start}:{m.char_end}] != {m.value!r}, dropped")
    return keep, problems


def process_document(
    path: str | Path,
    cfg: Config,
    ccfg: CascadeConfig,
    backend: Backend,
    models: baseline_pipeline.Models | None = None,
) -> Result:
    """Raises only if the baseline cannot read the file at all."""
    p = Path(path)
    base, sidecar = baseline_pipeline.process_document(p, cfg, models)
    deterministic = bool(cfg.pipeline.get("runtime", {}).get("deterministic_timings", False))
    timings: dict[str, float] = {}
    prov: dict[str, Any] = {
        "doc_id": base.doc_id, "filename": base.filename, "model": None, "finish_reason": None,
        "pages_sent": 0, "images_sent": 0, "fallback": False, "errors": [], "raw": None,
        "class": None, "fields": [], "timings_ms": timings,
    }

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            timings[f"cascade_{name}"] = 0.0 if deterministic else round((time.perf_counter() - t0) * 1000.0, 3)

    full, pages = base.text.full, base.text.pages
    try:
        r = ccfg.render
        images = timed("render", lambda: render_pages(p, r.max_pages, r.dpi, r.max_long_edge))
        sent = pages[: r.max_pages]
        prov["pages_sent"], prov["images_sent"] = len(sent), len(images)
        if not images and not full.strip():
            raise ValueError("nothing to send: no page image and no text")
        messages = build_messages(cfg, sent, images, r.max_text_chars)
        reply = timed("vlm", lambda: backend.complete(messages, response_schema(cfg)))
        prov["model"], prov["finish_reason"], prov["raw"] = reply.model, reply.finish_reason, reply.content
        answer = parse_answer(reply.content, cfg)
        decision = timed("decide", lambda: decide(
            full, pages, cfg, ccfg.confidence, rule_scores(full, pages, cfg), answer,
            base.metadata, sidecar.words, sidecar.page_sizes,
        ))
    except Exception as exc:
        msg = f"cascade: {type(exc).__name__}: {exc}"
        log.error("cascade fell back to the baseline", extra={"doc_id": base.doc_id, "error": msg})
        prov["fallback"] = True
        prov["errors"].append(msg)
        record = _record(base, base.classification, base.metadata, base.tags,
                         _diagnostics(base, timings, [msg]))
        return Result(record, sidecar, prov)

    meta, problems = _drop_bad_offsets(decision.metadata, full)
    facts = baseline_pipeline.tag_facts(base.text, base.diagnostics.page_count, base.diagnostics.is_scanned)
    tags = apply_tags(full, pages, cfg, meta, facts, models.tagger if models else None)
    prov["class"], prov["fields"] = decision.class_provenance, decision.field_provenance
    prov["errors"].extend(problems)
    record = _record(base, decision.classification, meta, tags, _diagnostics(base, timings, problems))
    return Result(record, sidecar, prov)


def run_batch(
    input_dir: str | Path,
    output: str | Path,
    cfg: Config,
    ccfg: CascadeConfig,
    backend: Backend,
    models: baseline_pipeline.Models | None = None,
    bbox_dir: str | Path | None = None,
    resume: bool = True,
) -> dict:
    """Process a directory into JSONL plus a provenance sidecar. Resumable."""
    check_descriptions(cfg)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path = out_path.with_name(out_path.name + ".provenance")
    bdir = Path(bbox_dir) if bbox_dir else out_path.with_name(out_path.name + ".bboxes")

    paths, excluded = scan_documents(input_dir)
    skip = baseline_pipeline.existing_doc_ids(out_path) if resume else set()
    stats = {"found": len(paths), "skipped": 0, "written": 0, "failed": 0, "fallback": 0,
             "excluded": {reason: len(items) for reason, items in sorted(excluded.items())}}
    mode = "a" if resume else "w"

    with open(out_path, mode, encoding="utf-8", newline="\n") as fh, \
         open(prov_path, mode, encoding="utf-8", newline="\n") as ph:
        for p in paths:
            try:
                if compute_doc_id(p) in skip:
                    stats["skipped"] += 1
                    continue
                result = process_document(p, cfg, ccfg, backend, models)
            except Exception as exc:
                log.error("document failed", extra={"path": str(p), "error": str(exc)})
                stats["failed"] += 1
                continue
            baseline_pipeline.write_sidecar(result.sidecar, bdir)
            fh.write(baseline_pipeline.dumps(result.record) + "\n")
            ph.write(json.dumps(result.provenance, ensure_ascii=False) + "\n")
            fh.flush()
            ph.flush()
            stats["written"] += 1
            stats["fallback"] += int(result.provenance["fallback"])

    log.info("cascade batch done", extra=stats)
    return stats
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_cascade_runner.py -v`
Expected: PASS, 5 tests.

If `test_invoice_end_to_end` fails on `vendor_name`, print `base.metadata` for the invoice and the `vendor_name` provenance item. The name is printed twice, and the agreement step must pick the occurrence the highest ranked rule candidate produced (offset 50 on this document). A mismatch there is a grounding bug, so fix it in `cascade/ground.py`, not in the test.

`broken.pdf` in the corpus cannot be opened. The baseline returns a record with empty text and an extraction error; `render_pages` then raises, so cascade falls back and records it. That is the intended path.

- [ ] **Step 5: Commit**

```bash
git add cascade/runner.py tests/test_cascade_runner.py
git commit -m "Run cascade per document and per batch, falling back to the baseline"
```

---

### Task 10: CLI, documentation, and final verification

**Files:**
- Create: `cascade/cli.py`, `cascade/__main__.py`
- Modify: `pyproject.toml` (`[project.scripts]`), `README.md`, `HANDOFF.md`, `docs/superpowers/specs/2026-09-17-cascade-architecture-design.md` (status line)
- Test: `tests/test_cascade_cli.py`

**Interfaces:**
- Consumes: `run_batch` (Task 9), `VLMClient` (Task 7), `load_cascade_config` (Task 3), `baseline.config.load_config_cached`, `baseline.pipeline.Models`.
- Produces: `cascade.cli.main(argv: list[str] | None = None) -> int`; the `python -m cascade run` and `cascade run` commands.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cascade_cli.py`:

```python
import json

from conftest import ROOT
from vlm_stub import chat, serve


def test_run_against_a_local_server_writes_records_and_provenance(corpus, tmp_path):
    from cascade.cli import main

    out = tmp_path / "out.jsonl"
    answer = chat(json.dumps({"class": "unknown", "fields": {}}))
    with serve([(200, answer, 0, None)]) as (url, reqs):
        rc = main(["--config", str(ROOT / "config"), "run", "--input", str(corpus),
                   "--output", str(out), "--model", "m", "--base-url", url])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    provs = (tmp_path / "out.jsonl.provenance").read_text(encoding="utf-8").splitlines()
    assert lines and len(lines) == len(provs)
    assert reqs and all(r["body"]["model"] == "m" for r in reqs)


def test_a_remote_server_is_refused_before_any_document_is_read(corpus, tmp_path):
    from cascade.cli import main

    out = tmp_path / "out.jsonl"
    rc = main(["--config", str(ROOT / "config"), "run", "--input", str(corpus),
               "--output", str(out), "--model", "m", "--base-url", "http://192.0.2.1:8000/v1"])
    assert rc == 1
    assert not out.exists()


def test_no_model_configured_is_refused(corpus, tmp_path):
    from cascade.cli import main

    rc = main(["--config", str(ROOT / "config"), "run", "--input", str(corpus),
               "--output", str(tmp_path / "out.jsonl")])
    assert rc == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_cascade_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cascade.cli'`.

- [ ] **Step 3: Implement the CLI**

Create `cascade/cli.py`:

```python
"""Command line entry point.

  cascade run --input DIR --output results.jsonl [--model NAME] [--base-url URL]

Writes results.jsonl, results.jsonl.provenance and results.jsonl.bboxes/.
Score with `baseline eval` and gate with `baseline compare`, as for any run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baseline.config import load_config_cached
from baseline.logging_setup import get_logger, setup_logging
from baseline.pipeline import Models

from .config import load_cascade_config
from .runner import run_batch
from .vlm import VLMClient

log = get_logger(__name__)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config_cached(args.config)
    ccfg = load_cascade_config(args.config)
    overrides = {k: v for k, v in (("model", args.model), ("base_url", args.base_url)) if v}
    settings = ccfg.vlm.model_copy(update=overrides)
    backend = VLMClient(settings)  # refuses a missing model or a non local host
    log.info("cascade server", extra={"base_url": settings.base_url, "model": settings.model})
    stats = run_batch(
        args.input, args.output, cfg, ccfg, backend,
        models=Models.load(cfg, Path(args.root)),
        bbox_dir=args.bbox_dir,
        resume=not args.no_resume,
    )
    log.info("run complete", extra=stats)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade", description="Cascade v2: VLM extraction with grounding")
    p.add_argument("--config", default=None, help="config directory, defaults to ./config")
    p.add_argument("--root", default=".", help="root for relative model paths")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="process a directory into JSONL")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--model", default=None, help="overrides vlm.model in cascade.yaml")
    r.add_argument("--base-url", default=None, help="overrides vlm.base_url in cascade.yaml")
    r.add_argument("--bbox-dir", default=None, help="defaults to <output>.bboxes")
    r.add_argument("--no-resume", action="store_true", help="reprocess everything")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except Exception:
        log.exception("command failed", extra={"command": args.command})
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

Create `cascade/__main__.py`:

```python
import sys

from .cli import main

sys.exit(main())
```

In `pyproject.toml`, under `[project.scripts]`, add:

```toml
cascade = "cascade.cli:main"
```

- [ ] **Step 4: Run the CLI tests**

Run: `python -m pytest tests/test_cascade_cli.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Document it**

In `README.md`, add a section after the baseline quick start:

````markdown
## Cascade v2

A local vision language model reads each page, and every value it returns is
located in the canonical text before it is emitted. Design:
`docs/superpowers/specs/2026-10-05-cascade-v2-vlm-grounding-design.md`.

Serve a model on an OpenAI compatible endpoint on this machine (vLLM or
Ollama), then:

```bash
python -m cascade run --input corpus --output cascade.jsonl --model <model-name> --base-url http://127.0.0.1:8000/v1
python -m baseline compare --gold gold.jsonl --a results.jsonl --b cascade.jsonl
```

`cascade.jsonl.provenance` records, per document, the model's raw answer and
what happened to every value: grounded exactly, loosely or through its
normalizer, ungrounded, failed checksum, or dropped as rule only. It holds
document values, so on client data it is client material, and `*.provenance`
is gitignored.
````

In `HANDOFF.md`, under "## Where things stand", add a bullet after the "Designed, not built" bullet:

```markdown
- **Cascade v2 is built, unscored on a real model.** On 2026-10-05 the
  cascade design was replaced for the GPU client by VLM extraction with
  grounding: `docs/superpowers/specs/2026-10-05-cascade-v2-vlm-grounding-design.md`,
  plan in `docs/superpowers/plans/2026-10-05-cascade-v2-vlm-grounding.md`,
  code in `cascade/`, on branch `cascade-v2`. Every test runs against a fake
  backend. Next: serve a permissively licensed VLM locally, run it over the
  synthetic corpus and the labelled set, and gate it with `baseline compare`.
```

In `docs/superpowers/specs/2026-09-17-cascade-architecture-design.md`, change line 4 to:

```
Status: superseded in part on 2026-10-05 by 2026-10-05-cascade-v2-vlm-grounding-design.md
```

- [ ] **Step 6: Final verification**

Run: `python -m pytest -q`
Expected: all pass. That is 365 from before plus the new cascade tests, with no failures and no errors.

Re-run the baseline identity check from Task 2, Step 5. The field descriptions added in Task 5 must not have changed any record:

```bash
python -m baseline run --input .check/corpus --output .check/after.jsonl --no-resume
python - <<'EOF'
import json

def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            r.pop("schema_version")
            r["diagnostics"].pop("timings_ms")
            out[r["doc_id"]] = r
    return out

a, b = load(".check/before.jsonl"), load(".check/after.jsonl")
assert a.keys() == b.keys() and all(a[k] == b[k] for k in a)
print("identical", len(a))
EOF
```

Expected: `identical N`.

Run: `git status --short`
Expected: no `.check/` or `*.provenance` paths listed.

- [ ] **Step 7: Commit**

```bash
git add cascade/cli.py cascade/__main__.py pyproject.toml README.md HANDOFF.md docs/superpowers/specs/2026-09-17-cascade-architecture-design.md tests/test_cascade_cli.py
git commit -m "Add the cascade run command and document cascade v2"
```

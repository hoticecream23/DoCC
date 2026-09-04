"""Is this text layer trustworthy, or is it somebody else's bad OCR?

A PDF can carry an embedded text layer that was itself produced by a scanner's
OCR and saved into the file. PyMuPDF hands it over as native text, so the
extractor never routes the page to OCR and nothing downstream learns the text
is corrupt. On the client invoice set this is the single largest source of
classification failure: `lnvoice` with a lowercase L never matches an
`invoice` pattern, and the document falls to `unknown`.

This module scores that corruption so the extractor can decide to re-render
the page and OCR it properly. It holds no vocabulary: no field name, class
name or domain word appears here, only facts about how Latin script behaves.

The three signals, all measured on the client corpus before being chosen:

`l_for_i`    A lowercase L standing in for a capital I. English has almost no
             words starting `l` followed by another consonant, so `lnvoice`,
             `lnvolce` and `lAm` are near proof of substitution.
`digit_letter` A lowercase letter touching a digit inside one token: `lnvoic6`,
             `12122t2025`, `1t16t2026`. Identifiers do this too, but they
             normally do it in upper case, so restricting to lower case keeps
             `INV-2024-0042` and `IN3308526` out of the count.
`case_shape` A token whose letters are neither all lower, nor all upper, nor
             capitalised: `NuDb`, `Typ€`, `lAm`. Tokens containing `.` or `/`
             are skipped, because initials and reference numbers legitimately
             look like this: `G.Jayachandran`, `W.P.No.26378/2023`.

Non Latin script is counted separately and deliberately kept out of the score.
A Karnataka judgment carrying Kannada in a legacy 8 bit font reads as accented
Latin-1 and scored 0.233 on an earlier version of this metric, which was the
highest score in the entire corpus and completely wrong: the text is not
corrupt, it is another language, and re-running English OCR would not improve
it. `non_latin_rate` surfaces that as its own finding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A lowercase l followed by a consonant. Vowels are excluded because "like",
# "lost" and "clean" are ordinary words; consonant clusters after l are not.
_L_FOR_I = re.compile(r"\bl[bcdfgjklmnpqrstvwxz]")
_DIGIT_LETTER = re.compile(r"[a-z]\d|\d[a-z]")
_LATIN_ONLY = re.compile(r"^[\x00-\x7F]+$")
_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class TextQuality:
    """Corruption evidence for one block of text. Higher score is worse."""

    score: float
    l_for_i: float
    digit_letter: float
    case_shape: float
    non_latin_rate: float
    tokens: int

    def is_suspect(self, threshold: float) -> bool:
        return self.tokens > 0 and self.score >= threshold


def _has_bad_case_shape(token: str) -> bool:
    # Initials and reference numbers are legitimately mixed case.
    if "." in token or "/" in token:
        return False
    letters = "".join(c for c in token if c.isalpha())
    if len(letters) < 3:
        return False
    return not (
        letters.islower()
        or letters.isupper()
        or (letters[0].isupper() and letters[1:].islower())
    )


def score_text(text: str) -> TextQuality:
    """Rate one block of text. Empty text is not suspect, it is simply empty."""
    tokens = [t for t in _TOKEN.findall(text) if any(c.isalpha() for c in t)]
    if not tokens:
        return TextQuality(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    latin = [t for t in tokens if _LATIN_ONLY.match(t)]
    non_latin_rate = (len(tokens) - len(latin)) / len(tokens)

    # Every rate below is measured over Latin tokens only, so a document in
    # another script is not scored as corrupt for being in another script.
    denom = max(1, len(latin))
    l_for_i = len(_L_FOR_I.findall(" ".join(latin))) / denom
    digit_letter = sum(1 for t in latin if _DIGIT_LETTER.search(t)) / denom
    case_shape = sum(1 for t in latin if _has_bad_case_shape(t)) / denom

    return TextQuality(
        score=round(l_for_i + digit_letter + case_shape, 6),
        l_for_i=round(l_for_i, 6),
        digit_letter=round(digit_letter, 6),
        case_shape=round(case_shape, 6),
        non_latin_rate=round(non_latin_rate, 6),
        tokens=len(tokens),
    )


def prefer(native: str, ocr: str) -> tuple[str, bool]:
    """Pick the cleaner of two readings of the same page.

    Returns (chosen_text, used_ocr). The comparison is the whole point of the
    gate: a low score does not prove OCR is right, it only proves it is less
    obviously wrong, and a re-OCR that scores worse is discarded. This is what
    keeps the gate from being able to make the corpus worse by its own metric.
    """
    if not ocr.strip():
        return native, False
    if not native.strip():
        return ocr, True
    return (ocr, True) if score_text(ocr).score < score_text(native).score else (native, False)

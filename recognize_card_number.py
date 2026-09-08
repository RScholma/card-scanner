"""
Set-number recognition for Yu-Gi-Oh! cards.

The printed set number (e.g. `RA01-EN057`) is only ~20-35 px tall in a typical
crop, which is why raw pytesseract/EasyOCR both score ~2-5% exact match on it.
This module attacks that from three directions instead of one:

  1. Scale normalisation + a spread of preprocessing variants, so at least one
     rendering of the crop lands in the range the OCR engines were trained for.
  2. Multi-engine, multi-config candidate generation (Tesseract psm 7/13 and
     EasyOCR), producing a pool of noisy guesses rather than one.
  3. Snapping the pool to a closed vocabulary of ~200k real set numbers using a
     distance that knows which character pairs OCR actually confuses
     (0/O, 1/I, 5/S, 2/Z, 8/B, ...).

Step 3 is what does the heavy lifting: it turns "read 10 tiny glyphs correctly"
into "pick the nearest real code", so a candidate only has to be *close*.

Public API:
    recognize(bgr_image)          -> Result
    recognize_crop(bgr_crop)      -> Result   (alias, crop already isolated)
"""

from __future__ import annotations

import functools
import gzip
import os
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

LEXICON_PATH = os.path.join(os.path.dirname(__file__), "data", "card_numbers.txt.gz")
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

# Character pairs OCR engines routinely swap. Substituting within a pair is
# cheap; any other substitution costs a full point.
_CONFUSABLE = [
    "0OQDU", "1IL|T", "2Z", "5S", "8B", "6G", "9g", "4A", "7T",
    "MN", "EF", "CG", "KX", "VY", "PR", "3B",
]
_CONFUSION_COST = 0.25
_SUB_COST = 1.0


def _build_confusion_table():
    table = {}
    for group in _CONFUSABLE:
        for a in group:
            for b in group:
                if a != b:
                    table[(a, b)] = _CONFUSION_COST
    return table


_CONFUSION = _build_confusion_table()


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return _CONFUSION.get((a, b), _SUB_COST)


def _weighted_distance(a: str, b: str, cutoff: float = float("inf")) -> float:
    """Levenshtein distance with cheap substitutions for confusable glyphs."""
    if a == b:
        return 0.0
    if abs(len(a) - len(b)) > cutoff:
        return cutoff + 1.0

    previous = [float(i) for i in range(len(b) + 1)]
    for i, ca in enumerate(a, start=1):
        current = [float(i)]
        best_in_row = current[0]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1.0,                      # deletion
                current[j - 1] + 1.0,                   # insertion
                previous[j - 1] + _sub_cost(ca, cb),    # substitution
            ))
            best_in_row = min(best_in_row, current[-1])
        if best_in_row > cutoff:
            return cutoff + 1.0
        previous = current
    return previous[-1]


# --------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_lexicon():
    """Return (all_numbers, suffixes_by_prefix, prefix_list)."""
    if not os.path.exists(LEXICON_PATH):
        raise FileNotFoundError(
            f"Lexicon not found at {LEXICON_PATH}. Run: python build_lexicon.py"
        )
    with gzip.open(LEXICON_PATH, "rt", encoding="utf-8") as fh:
        numbers = {line.strip() for line in fh if line.strip()}

    by_prefix = defaultdict(set)
    for number in numbers:
        prefix, _, suffix = number.partition("-")
        by_prefix[prefix].add(suffix)
    return numbers, dict(by_prefix), sorted(by_prefix)


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _scale_to_height(gray, target_h):
    h, w = gray.shape[:2]
    if h == 0:
        return gray
    scale = target_h / h
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(gray, (max(1, int(round(w * scale))), target_h), interpolation=interp)


def _unsharp(gray, amount=1.5, radius=3):
    blur = cv2.GaussianBlur(gray, (0, 0), radius)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


def _pad(img, border=20, value=255):
    return cv2.copyMakeBorder(img, border, border, border, border,
                              cv2.BORDER_CONSTANT, value=value)


def _ensure_dark_text(binary):
    """Return the image with dark text on a white ground."""
    # Text is the minority ink; if the image is mostly dark, flip it.
    if np.mean(binary) < 127:
        return cv2.bitwise_not(binary)
    return binary


def preprocess_variants(crop, fast=False):
    """Yield (name, image) renderings of a number crop for OCR.

    Small text fails mostly because the glyphs fall below the resolution the
    engines expect, so every variant is first scaled to a healthy cap height
    and padded (Tesseract needs quiet space around a text line).

    `fast=True` keeps only the single most productive scale, for interactive
    use where a few seconds per frame is not acceptable.
    """
    gray = _to_gray(crop)
    variants = []

    for target_h in ((96,) if fast else (64, 96, 128)):
        scaled = _scale_to_height(gray, target_h)
        sharp = _unsharp(scaled)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(scaled)
        clahe_sharp = _unsharp(clahe)

        _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, otsu_clahe = cv2.threshold(clahe_sharp, 0, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            clahe_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, max(11, (target_h // 8) * 2 + 1), 8,
        )

        variants.extend([
            (f"gray{target_h}", _pad(sharp)),
            (f"clahe{target_h}", _pad(clahe_sharp)),
            (f"otsu{target_h}", _pad(_ensure_dark_text(otsu))),
            (f"otsuclahe{target_h}", _pad(_ensure_dark_text(otsu_clahe))),
            (f"adaptive{target_h}", _pad(_ensure_dark_text(adaptive))),
        ])

    return variants


# --------------------------------------------------------------------------
# OCR engines
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _easyocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _tesseract_candidates(variants, fast=False):
    try:
        import pytesseract
    except ImportError:
        return []

    out = []
    for psm in ((7,) if fast else (7, 13)):
        config = (f"--oem 1 --psm {psm} "
                  f"-c tessedit_char_whitelist={CHARSET}")
        for name, img in variants:
            try:
                text = pytesseract.image_to_string(img, config=config)
            except Exception:
                return out  # engine missing / broken: stop trying
            out.append((f"tess-psm{psm}-{name}", text))
    return out


def _easyocr_candidates(variants):
    try:
        reader = _easyocr_reader()
    except Exception:
        return []

    out = []
    for name, img in variants:
        try:
            parts = reader.readtext(img, detail=0, allowlist=CHARSET, paragraph=False)
        except Exception:
            return out
        out.append((f"easy-{name}", "".join(parts)))
    return out


# --------------------------------------------------------------------------
# Candidate cleanup
# --------------------------------------------------------------------------

def _clean(raw: str) -> str:
    text = "".join(ch for ch in raw.upper() if ch in CHARSET)
    return text.strip("-")


def _split(text: str):
    """Split a candidate into (prefix, suffix) around the set-code dash."""
    if "-" in text:
        prefix, _, suffix = text.partition("-")
        return prefix, suffix.replace("-", "")
    # No dash survived OCR. Real codes are PREFIX + 2-letter region + 3 digits,
    # so peel the trailing digits and the region letters back off.
    i = len(text)
    while i > 0 and text[i - 1].isdigit():
        i -= 1
    digits = text[i:]
    head = text[:i]
    if len(digits) >= 2 and len(head) >= 2:
        return head[:-2], head[-2:] + digits
    return head, digits


# Position-aware repairs. A set number is PREFIX-REGION+SERIAL, where the
# serial is always digits and the region is always letters, so a glyph read in
# the wrong class can be corrected from shape alone - no lexicon needed. This
# is what turns `PHNI-ENO45` into `PHNI-EN045`.
_AS_DIGIT = str.maketrans({
    "O": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "L": "1", "J": "1", "T": "7",
    "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4",
})
_AS_LETTER = str.maketrans({
    "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A",
})


def _shape_fix(prefix: str, suffix: str):
    """Coerce each part of a candidate into the character class it must hold."""
    if len(suffix) >= 3:
        serial = suffix[-3:].translate(_AS_DIGIT)
        region = suffix[:-3].translate(_AS_LETTER)
        suffix = region + serial
    return prefix, suffix


def _plausible(prefix: str, suffix: str) -> bool:
    """Reject shapes no real set number takes, so junk cannot accrue votes."""
    return (
        2 <= len(prefix) <= 5
        and 3 <= len(suffix) <= 6
        and prefix.isalnum()
        and suffix[-3:].isdigit()
    )


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass
class Result:
    text: str | None
    confidence: float
    raw_candidates: list = field(default_factory=list)
    scores: list = field(default_factory=list)

    def __bool__(self):
        return self.text is not None


# --------------------------------------------------------------------------
# Lexicon matching
# --------------------------------------------------------------------------

#: Beyond this weighted distance a "match" is speculative enough that trusting
#: the OCR consensus is safer. The lexicon is a corrector, not an authority -
#: it does not cover every set, so it must never overrule a confident read.
MAX_SNAP_COST = 2.0

#: A lexicon hit at distance `c` is worth LEXICON_WEIGHT / (1 + c) against the
#: 1.0 a literal reading scores. At 2.5 the crossover sits near c = 1.5: close
#: matches win, speculative ones lose.
LEXICON_WEIGHT = 2.5

#: How many independent candidates must back a reading before its confidence is
#: taken at face value. Below this the score is scaled down proportionally.
MIN_SUPPORT = 4


def _match_to_lexicon(prefix, suffix, top_prefixes=6):
    """Return (full_code, cost) for the best real card number, or None."""
    numbers, by_prefix, prefix_list = load_lexicon()

    if not prefix and not suffix:
        return None

    # Exact hit short-circuits everything.
    exact = f"{prefix}-{suffix}"
    if exact in numbers:
        return exact, 0.0

    scored_prefixes = []
    for cand in prefix_list:
        if abs(len(cand) - len(prefix)) > 1:
            continue
        cost = _weighted_distance(prefix, cand, cutoff=2.5)
        if cost <= 2.5:
            scored_prefixes.append((cost, cand))
    if not scored_prefixes:
        return None
    scored_prefixes.sort()

    best = None
    for prefix_cost, cand_prefix in scored_prefixes[:top_prefixes]:
        for cand_suffix in by_prefix[cand_prefix]:
            if abs(len(cand_suffix) - len(suffix)) > 1:
                continue
            suffix_cost = _weighted_distance(suffix, cand_suffix, cutoff=3.0)
            if suffix_cost > 3.0:
                continue
            total = prefix_cost + suffix_cost
            if best is None or total < best[1]:
                best = (f"{cand_prefix}-{cand_suffix}", total)
    return best


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def recognize(crop, fast=False, return_debug=False) -> Result:
    """Read the set number from an already-cropped number region (BGR or gray).

    Runs every preprocessing variant through every available engine, snaps each
    guess to the nearest real card number, and returns the code that the pool
    of guesses agrees on most strongly.

    `fast=True` trades some accuracy for roughly a 5x speedup (one scale,
    one Tesseract mode) - use it for live/webcam frames.
    """
    if crop is None or crop.size == 0:
        return Result(None, 0.0)

    variants = preprocess_variants(crop, fast=fast)
    raw = _tesseract_candidates(variants, fast=fast) + _easyocr_candidates(variants)

    votes = defaultdict(float)
    supporters = defaultdict(int)
    seen = []
    for source, text in raw:
        cleaned = _clean(text)
        if len(cleaned) < 4:
            continue
        seen.append((source, cleaned))

        prefix, suffix = _shape_fix(*_split(cleaned))
        if not _plausible(prefix, suffix):
            continue

        # Every engine/variant that agrees on a shape-corrected reading gets an
        # equal say. This stands on its own, so a set missing from the lexicon
        # can still be read correctly.
        votes[f"{prefix}-{suffix}"] += 1.0
        supporters[f"{prefix}-{suffix}"] += 1

        # The lexicon then adds weight to readings that are (or are near) a real
        # code. The weight is tuned so a near match (cost <= ~1.4, i.e. a couple
        # of confusable glyphs like MPIS -> MP15) outranks the literal reading,
        # while a distant one does not - past that, consensus is the better bet.
        match = _match_to_lexicon(prefix, suffix)
        if match is not None:
            code, cost = match
            if cost <= MAX_SNAP_COST:
                votes[code] += LEXICON_WEIGHT / (1.0 + cost)
                supporters[code] += 1

    if not votes:
        return Result(None, 0.0, raw_candidates=seen)

    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    total = sum(votes.values())
    best_code, best_score = ranked[0]

    # Confidence has to answer two questions, not one: did the candidates agree,
    # and were there enough of them to mean anything? Agreement alone rates a
    # lone surviving guess as unanimous, which is how a garbage read once came
    # back at 1.00. Scaling by evidence keeps a single voter capped low.
    agreement = best_score / total if total else 0.0
    evidence = min(1.0, supporters[best_code] / MIN_SUPPORT)

    return Result(
        text=best_code,
        confidence=agreement * evidence,
        raw_candidates=seen if return_debug else [],
        scores=ranked[:5] if return_debug else [],
    )


recognize_crop = recognize

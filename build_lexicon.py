"""
Build the card-number lexicon used by `recognize_card_number.py`.

Source: the set-list JSON dumps that seed the Laravel backend
(`Yu-Gi-Oh-Scanner-Backend/.../database/seeds/data/set_versions/*.json`).

Every printed Yu-Gi-Oh! set number is a member of a small closed vocabulary,
so OCR output can be snapped to the nearest real code instead of being
trusted character-by-character. That is the single biggest accuracy win
available for the small-text problem.

Usage:
    python build_lexicon.py [path/to/set_versions]
"""

import glob
import gzip
import json
import os
import re
import sys

_SEEDS = os.path.join(
    "..", "Yu-Gi-Oh-Scanner-Backend", "Yu-Gi-Oh-Scanner-Backend",
    "database", "seeds", "data",
)
DEFAULT_SRC = os.path.join(_SEEDS, "set_versions")
# cardinfo.json is the newer of the two dumps and is the only one carrying
# recent sets (MP25, RA02, BLMM, ALIN, ...). The set_versions files go back
# further. Merging both gives the widest coverage.
DEFAULT_CARDINFO = os.path.join(_SEEDS, "cards", "cardinfo.json")
OUT_PATH = os.path.join("data", "card_numbers.txt.gz")

# Set number as printed on a card: PREFIX-REGION+SERIAL, e.g. RA01-EN057.
CARD_NUMBER_RE = re.compile(r"[A-Z0-9]{2,5}-[A-Z]{0,2}[0-9]{1,4}")


def collect_numbers(src_dir):
    numbers = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("Card number", "Set number") and isinstance(value, str):
                    candidate = value.strip().upper()
                    if CARD_NUMBER_RE.fullmatch(candidate):
                        numbers.add(candidate)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    files = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    if not files:
        raise SystemExit(f"No JSON files found in {src_dir}")

    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                walk(json.load(fh))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipped {os.path.basename(path)}: {exc}")

    return numbers, len(files)


def collect_from_cardinfo(path):
    """Pull `card_sets[].set_code` out of the ygoprodeck cardinfo dump."""
    if not os.path.exists(path):
        print(f"  cardinfo not found at {path}, skipping")
        return set()

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    numbers = set()
    for card in payload.get("data", payload):
        for entry in card.get("card_sets") or []:
            code = (entry.get("set_code") or "").strip().upper()
            if CARD_NUMBER_RE.fullmatch(code):
                numbers.add(code)
    return numbers


def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    cardinfo = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CARDINFO

    from_sets, n_files = collect_numbers(src_dir)
    from_cardinfo = collect_from_cardinfo(cardinfo)
    numbers = from_sets | from_cardinfo

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(numbers)))

    prefixes = {n.split("-")[0] for n in numbers}
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"set_versions : {len(from_sets)} numbers from {n_files} files")
    print(f"cardinfo.json: {len(from_cardinfo)} numbers "
          f"(+{len(from_cardinfo - from_sets)} new)")
    print(f"Wrote {len(numbers)} card numbers ({len(prefixes)} prefixes) "
          f"-> {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()

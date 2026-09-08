"""
End-to-end scan: photo -> card -> set-number region -> recognised set number.

Uses the two dedicated single-class detectors (which are far stronger than the
combined 2-class model) and hands the crop to `recognize_card_number`.

    from scan_pipeline import scan
    results = scan(cv2.imread("photo.jpg"))
    # [{'card_box': (...), 'number_box': (...), 'number': 'RA01-EN057', ...}]
"""

from __future__ import annotations

import functools
import os

import cv2

from recognize_card_number import recognize

CARD_WEIGHTS = "runs/detect/card_detector_v13/weights/best.pt"
NUMBER_WEIGHTS = "runs/detect/card_number_detector/weights/best.pt"

#: The number detector draws a tight box. OCR does better with a little context
#: around the glyphs, so the crop is padded by this fraction of the box height.
CROP_MARGIN = 0.35


@functools.lru_cache(maxsize=2)
def _model(weights):
    from ultralytics import YOLO
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Missing weights: {weights}")
    return YOLO(weights)


def _expand(box, margin, bounds):
    x1, y1, x2, y2 = box
    pad = int(round((y2 - y1) * margin))
    max_w, max_h = bounds
    return (
        max(0, x1 - pad), max(0, y1 - pad),
        min(max_w, x2 + pad), min(max_h, y2 + pad),
    )


def _fallback_number_region(card):
    """Where the set number sits on a standard card, when detection misses."""
    h, w = card.shape[:2]
    return (int(w * 0.55), int(h * 0.88), w, h)


def scan(img, conf=0.25, fast=False):
    """Detect every card in `img` and read its set number.

    Returns a list of dicts with `card_box`, `number_box`, `number`,
    `confidence` and `source` ('detector' or 'fallback').

    Pass `fast=True` for live frames; see `recognize_card_number.recognize`.
    """
    if img is None or img.size == 0:
        return []

    card_model = _model(CARD_WEIGHTS)
    number_model = _model(NUMBER_WEIGHTS)
    img_h, img_w = img.shape[:2]

    results = []
    for card_box in card_model(img, conf=conf, verbose=False)[0].boxes:
        cx1, cy1, cx2, cy2 = map(int, card_box.xyxy[0])
        cx1, cy1, cx2, cy2 = max(0, cx1), max(0, cy1), min(img_w, cx2), min(img_h, cy2)
        card = img[cy1:cy2, cx1:cx2]
        if card.size == 0:
            continue

        card_h, card_w = card.shape[:2]
        number_boxes = number_model(card, conf=conf, verbose=False)[0].boxes

        if len(number_boxes):
            best = max(number_boxes, key=lambda b: float(b.conf[0]))
            region = tuple(map(int, best.xyxy[0]))
            source = "detector"
        else:
            region = _fallback_number_region(card)
            source = "fallback"

        nx1, ny1, nx2, ny2 = _expand(region, CROP_MARGIN, (card_w, card_h))
        number_crop = card[ny1:ny2, nx1:nx2]

        outcome = recognize(number_crop, fast=fast)
        results.append({
            "card_box": (cx1, cy1, cx2, cy2),
            "number_box": (cx1 + nx1, cy1 + ny1, cx1 + nx2, cy1 + ny2),
            "number": outcome.text,
            "confidence": round(outcome.confidence, 3),
            "source": source,
        })

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python scan_pipeline.py <image> [image ...]")
    for path in sys.argv[1:]:
        for hit in scan(cv2.imread(path)) or [None]:
            if hit is None:
                print(f"{os.path.basename(path)}: no card detected")
            else:
                print(f"{os.path.basename(path)}: {hit['number']} "
                      f"(conf {hit['confidence']}, via {hit['source']})")

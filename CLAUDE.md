# CLAUDE.md — card-scanner

## Goal

Detect a Yu-Gi-Oh! card in an image (or webcam frame), locate the **set number**
(the code printed in the card's bottom-right corner, e.g. `RA01-EN057`,
`LDS3-EN127`), and read it with OCR. The set number is the key used to look the
card up in the [Yu-Gi-Oh-Scanner-Backend](../Yu-Gi-Oh-Scanner-Backend) metadata
API.

This repo is the **computer-vision / OCR pipeline only**. It runs as a
standalone FastAPI service. It is not yet wired to the Vue frontend (that talks
only to the Laravel backend).

## Architecture

Two-stage detection, then OCR.

### 1. Detection — YOLOv8 (Ultralytics 8.3.x)

All models fine-tuned from `yolov8n.pt` (nano), CPU, 50 epochs, `imgsz=640`.

| Weights | Classes | Quality (mAP50) | Used by |
|---|---|---|---|
| `runs/train/yu_gi_oh_card_numbers2/weights/best.pt` | `0=card`, `1=card_number` | ~0.69 | `detect_card.py` → `app.py` `/annotate-card` |
| `runs/detect/card_detector_v13/weights/best.pt` | `card` | ~0.99 | live scripts |
| `runs/detect/card_number_detector/weights/best.pt` | `card_number` | ~0.95 | live scripts, `crop_card_numbers.py` |

The dedicated single-class detectors (`card_detector_v13`,
`card_number_detector`) are stronger than the combined 2-class model and are
what the live scripts use in series: detect card → crop → detect number → crop.

### 2. OCR — four approaches tried, none settled

1. **pytesseract** — `scan_card_number.py`, used by the `/scan-card` API route.
   Fixed heuristic crop of the bottom-right corner (no YOLO), grayscale → Otsu →
   2× upscale → `--psm 7`.
2. **EasyOCR, light preprocessing** — `live_card_detection.py` (webcam,
   continuous).
3. **EasyOCR, heavy preprocessing** — `live_card_detection_v2.py` (webcam,
   press `s` to scan). 4× upscale, CLAHE, adaptive threshold, morphological
   close, conditional inversion; EasyOCR with charset allowlist
   `A-Z0-9-` and `text_threshold=0.7`.
4. **Custom CRNN** — `ocr_card_number_recognizer.py`. PyTorch CNN + 2-layer
   BiLSTM + CTC loss, charset `A-Z0-9-`, trained on the number crops.
   `format_card_number()` normalizes `PREFIX + digits` and zero-pads the number
   to 3 digits. Training script only — **no checked-in weights**.

### API (`app.py`, FastAPI + Uvicorn)

- `POST /scan-card` — multipart image → `{ "number": "<ocr text>" }` (pytesseract path).
- `POST /annotate-card` — multipart image → `{ "detections": [[cls, xc, yc, w, h], ...] }`, normalized YOLO format, `0=card` / `1=card number`.
- `GET /` — health string.

### Key files

- `app.py` — FastAPI entrypoint.
- `detect_card.py` — YOLO inference wrapper (combined 2-class model).
- `scan_card_number.py` — pytesseract OCR (heuristic crop).
- `ocr_card_number_recognizer.py` — CRNN training script.
- `crop_card_numbers.py` — runs the number detector over a folder, saves number
  crops (+5 px margin) as PNG + placeholder label files for manual transcription.
- `live_card_detection.py`, `live_card_detection_v2.py` — webcam demos.
- `helper_functions/` — dataset tooling:
  - `renamer.py` — rename incoming images to `card_NNN`, 80/20 train/val split.
  - `split_yolo_dataset.py` — split a combined 2-class label set into separate
    `cards/` and `numbers/` YOLO datasets.
  - `batch_annotate.py` — auto-label images with a trained model.
  - `label_card_numbers.py` — manual number-string labeler.
  - `fix_number_labels.py`, `train_val_split.py`.

### Datasets (`dataset/`)

- `cards/` — 372 JPG + YOLO `.txt` labels + `data.yaml` (`nc: 1`, `['card']`), train/val.
- `numbers/` — 314 JPG + labels + `data.yaml` (`nc: 1`, `['card_number']`), train/val.
- `card_number_crops/` — 206 PNG number crops + 206 `.txt` files holding the
  ground-truth strings; the OCR train/test set. A few labels are dirty/truncated
  (`RA01-EN05`, `PHRA-EN051`).

## Set-number recognition (current approach)

`recognize_card_number.recognize()` is the working recogniser; `scan_pipeline.scan()`
is the end-to-end entry point (photo -> card -> number region -> code) and is what
`POST /scan-card` serves. **69% exact match on the 42 labelled val crops**, against
2% for bare pytesseract and 5% for bare EasyOCR.

It works because the problem has three independent handles, and the old code used
none of them:

1. **Scale** — crops are 20-35 px tall; the engines want 64-128 px plus padding.
   `preprocess_variants()` emits 15 renderings (3 target heights x gray / CLAHE /
   Otsu / Otsu+CLAHE / adaptive).
2. **Character class** — the serial is always digits and the region always
   letters, so `PHNI-ENO45` -> `PHNI-EN045` is fixable from shape alone, with no
   lookup. That is `_shape_fix()`.
3. **Closed vocabulary** — every real set number is one of ~205k codes in
   `data/card_numbers.txt.gz`. `_match_to_lexicon()` snaps a reading to the
   nearest real code using an edit distance where OCR-confusable pairs
   (`0/O`, `1/I`, `5/S`, `2/Z`, `8/B`) cost 0.25 instead of 1.0.

Every variant x engine pair votes; the lexicon adds weight to readings that are
near a real code. Two tuning constants matter and are easy to get wrong:

- `LEXICON_WEIGHT` (2.5) sets the crossover at ~1.5 edit-cost. Too low and
  `MPIS-EN201` beats the real `MP15-EN201`; too high and the lexicon overrules
  correct reads of sets it does not contain.
- `MIN_SUPPORT` (4) scales confidence by how many candidates backed the winner.
  Without it a lone surviving guess scores 1.00 by construction — that bug had
  two garbage reads rated maximally confident.

Rebuild the lexicon with `python build_lexicon.py`; score changes with
`python eval_recognizer.py --show`.

**The lexicon must stay fresh.** Built from `set_versions/*.json` alone it covered
only 49% of the labelled crops (it predates MP25, RA02, BLMM, ALIN, ...), which
actively corrupted correct reads. Merging `cards/cardinfo.json` took that to 99%.
New sets will need a newer dump.

`recognize(fast=True)` uses one scale and one Tesseract mode for roughly a 5x
speedup — for webcam frames. Full mode is ~9s per crop on CPU, which is fine for
the API but not for a live loop.

## Known problem — set number text too small to OCR reliably

**Substantially mitigated** by the above, but not solved. Remaining failures on
the val split: 9 declined (returns `None`), 4 wrong.

The set number occupies only a tiny fraction of a full card image. After the
card and number-region detectors crop it out, the text is frequently just a few
pixels tall, so:

- **pytesseract** returns garbled or empty strings on real photos.
- **EasyOCR** improves with the heavy `live_card_detection_v2.py` preprocessing
  (4× upscale + CLAHE + adaptive threshold), but is still unreliable and
  sensitive to lighting, focus, glare, and card rotation.
- The **CRNN** is the intended fix (purpose-trained on the small crops) but has
  no trained weights checked in and only ~206 labeled crops to learn from.

Contributing factors:
- Low-res / low-DPI source images; nano detector produces loose crops.
- No perspective correction — angled cards distort the glyphs.
- Combined 2-class detector's number recall is weak (~0.69 mAP50); the
  dedicated `card_number_detector` (~0.95) should be preferred everywhere.
- Very small labeled OCR set (206 crops), some labels wrong.

Directions not yet tried: higher-resolution capture, deskew / perspective
warp before cropping, super-resolution on the crop, a larger YOLO backbone for
tighter number boxes, and expanding + cleaning the CRNN training set.

## Environment

- Python, `venv/` is committed. `pip install -r requirements.txt`.
- `requirements.txt` covers torch 2.8, ultralytics 8.3.213, opencv, pytesseract,
  easyocr 1.7.2, fastapi/uvicorn.
- **`pytesseract` is only a wrapper** — the Tesseract engine must be installed
  separately. On Windows: `scoop install tesseract tesseract-languages` then
  `scoop reset tesseract-languages` (sets `TESSDATA_PREFIX`). The `tesseract`
  binary must be on `PATH`. See the note at the bottom of `requirements.txt`.
- Run the API: `uvicorn app:app --reload`.

## Ground truth available

`dataset/card_number_crops/{train,val}/labels/*.txt` hold the **actual set-number
string** for 206 cards (164 train / 42 val), each paired with a pre-cropped PNG
in the sibling `images/` folder. This is the only ground truth — there are no
labels for arbitrary images outside this set.

Exact match on the 42 val crops:

| approach | exact |
|---|---|
| pytesseract, Otsu + 3x upscale, `--psm 7` | 1/42 (2%) |
| EasyOCR, same preprocessing | 2/42 (5%) |
| `recognize_card_number.recognize()` | **29/42 (69%)** |

Two of the 206 labels are not real codes: `RA01-EN05` is truncated and
`DUAD-EN054` is absent from every dump. Fixing those is worth doing before
reading much into small score changes.

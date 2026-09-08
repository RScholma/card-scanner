from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

from detect_card import detect_card_and_number
from scan_pipeline import scan

app = FastAPI(title="Card Scanner API")


@app.post("/scan-card")
async def scan_card(file: UploadFile = File(...)):
    """
    Detects each card in the uploaded image, locates its set number and reads it.

    Returns the best result plus every card found:
        {"number": "RA01-EN057", "confidence": 0.81, "cards": [...]}
    """
    # Read uploaded file into OpenCV image
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"number": None, "confidence": 0.0, "cards": [], "error": "unreadable image"}

    cards = scan(img)
    best = max((c for c in cards if c["number"]), key=lambda c: c["confidence"], default=None)
    return {
        "number": best["number"] if best else None,
        "confidence": best["confidence"] if best else 0.0,
        "cards": cards,
    }


@app.post("/annotate-card")
async def annotate_card(file: UploadFile = File(...)):
    """
    Runs the trained YOLO model on the uploaded image and returns
    bounding boxes in YOLO format (class_id, x_center, y_center, width, height).
    
    Class IDs:
        0 = full card
        1 = card number
    """
    # Read uploaded file into OpenCV image
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Run detection
    detections = detect_card_and_number(img)
    return {"detections": detections or []}


@app.get("/")
async def root():
    """
    Root endpoint for quick server check.
    """
    return {"message": "Card Scanner API is running. Use /scan-card or /annotate-card endpoints."}

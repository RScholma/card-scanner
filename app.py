from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

from detect_card import detect_card_and_number
from scan_card_number import extract_card_number

app = FastAPI(title="Card Scanner API")


@app.post("/scan-card")
async def scan_card(file: UploadFile = File(...)):
    """
    Extracts the card number text from an uploaded image using OCR.
    """
    # Read uploaded file into OpenCV image
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Extract card number
    card_number = extract_card_number(img)
    return {"number": card_number}


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

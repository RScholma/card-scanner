from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import pytesseract

from scan_card_number import extract_card_number

app = FastAPI()

@app.post("/scan-card")
async def scan_card(file: UploadFile = File(...)):
    # Read file bytes and decode into OpenCV image
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Run your extraction logic
    card_number = extract_card_number(img)

    return {"number": card_number}
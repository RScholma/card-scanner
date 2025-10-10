import cv2
import pytesseract

def extract_card_number(img):
    """
    Example placeholder logic:
    - Crop region where card number is located
    - Apply OCR to extract text
    """

    # Assuming number is near bottom-right (40% height, 90% width)
    h, w, _ = img.shape
    crop = img[int(h*0.35):int(h*0.45), int(w*0.85):w]  # tweak as needed

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(thresh, config="--psm 7")
    return text.strip()

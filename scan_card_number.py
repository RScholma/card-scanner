import cv2
import pytesseract


def extract_card_number(img):
    """
    Extracts the card number text from a given image using OCR (pytesseract).
    You can adjust the crop region depending on your card layout.
    """

    # Get image dimensions
    h, w, _ = img.shape

    # Crop area where the card number usually appears
    # (Here we assume it's near the bottom-right of the card)
    y_start = int(h * 0.55)
    y_end = int(h * 0.95)
    x_start = int(w * 0.70)
    x_end = w
    crop = img[y_start:y_end, x_start:x_end]

    # Convert to grayscale and threshold for better OCR accuracy
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Optionally, slightly enlarge text for better detection
    scale_factor = 2
    resized = cv2.resize(thresh, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # OCR: psm 7 = treat the image as a single line of text
    text = pytesseract.image_to_string(resized, config="--psm 7")

    # Clean up OCR output
    text = text.strip().replace("\n", "").replace("\r", "")

    return text

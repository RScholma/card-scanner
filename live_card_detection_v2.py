import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# --- OCR Configuration ---
CARD_ID_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
# -------------------------------

# Load models
card_model = YOLO("runs/detect/card_detector_v13/weights/best.pt")
number_model = YOLO("runs/detect/card_number_detector/weights/best.pt")

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# --- FINAL ROBUST PREPROCESSING FUNCTION: CLAHE + Binarization ---
def preprocess_card_number_region(crop_image):
    
    h, w = crop_image.shape[:2]
    
    # 1. Upscale the image (Using your preferred scale factor of 4)
    scale_factor = 4
    width = int(w * scale_factor)
    height = int(h * scale_factor)
    if width > 0 and height > 0:
        crop_image = cv2.resize(crop_image, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

    # 3. *** CRITICAL TUNE: CLAHE for Local Contrast Enhancement ***
    #    Changed tileGridSize from (8,8) to (16,16) to smooth the local contrast 
    #    adjustment over a wider area, preserving fine lines.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    clahe_enhanced = clahe.apply(gray)
    
    # 4. Minimal Gaussian Blur (Kinder than Median Blur)
    blurred = cv2.GaussianBlur(clahe_enhanced, (1, 1), 0) # Back to (1,1)
    
    # 5. Adaptive Threshold
    # Adaptive threshold is more stable than Otsu for lighting variations
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 1
    )
    
    # 6. Morphological DILATION (To connect fragments)
    kernel = np.ones((2, 2), np.uint8) 
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 7. *** CRITICAL FIX: CONDITIONAL INVERSION ***
    #    If the average pixel value is low (meaning the image is mostly black/dark text), 
    #    it's already in the correct format. If it's high (mostly white/light text on black), 
    #    we need to flip it to ensure BLACK text on WHITE background.
    
    # Conditional inversion check (simplified)
    if np.mean(closed) < 128:
        final = 255 - closed
    else:
        final = closed

    # Optional slight erosion to reduce merged blobs
    final = cv2.erode(closed, np.ones((1,1), np.uint8), iterations=1)
    
    # Optional: Display the intermediate CLAHE image for inspection
    cv2.imshow("CLAHE Enhanced (Tuned)", clahe_enhanced)
    return final
# ------------------------------------

# Open webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("Press 's' to scan a frame or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # We only draw the live YOLO boxes in this loop.
    # No OCR is run here, to keep things fast.
    
    display_frame = frame.copy() # We'll draw boxes on this
    
    card_results = card_model(frame, verbose=False)
    for card_box in card_results[0].boxes:
        x1, y1, x2, y2 = map(int, card_box.xyxy[0])
        # Draw card box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the live feed with boxes
    cv2.putText(display_frame, "Press 's' to Scan, 'q' to Quit", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Live Feed", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
        
    if key == ord('s'):
        # --- SCAN KEY PRESSED: Run the full pipeline ONCE ---
        print("\n--- Scanning Frame ---")
        
        card_results = card_model(frame, verbose=False)
        
        # We need to copy the frame to draw on, otherwise we draw on the
        # main feed and it flickers.
        scan_display = frame.copy()

        for card_box in card_results[0].boxes:
            x1, y1, x2, y2 = map(int, card_box.xyxy[0])
            card_crop = frame[y1:y2, x1:x2]
            
            if card_crop.size == 0: continue

            number_results = number_model(card_crop, verbose=False)

            for num_box in number_results[0].boxes:
                nx1, ny1, nx2, ny2 = map(int, num_box.xyxy[0])
                number_crop = card_crop[ny1:ny2, nx1:nx2]
                
                if number_crop.size == 0: continue
                
                # 1. Preprocess the number crop
                preprocessed_crop = preprocess_card_number_region(number_crop)

                # 2. Run OCR with tuned parameters
                ocr_result = reader.readtext(
                    preprocessed_crop,
                    detail=0,
                    allowlist=CARD_ID_CHARS,
                    paragraph=False,
                    # *** NEW: Tell EasyOCR to ignore low-confidence "guesses"
                    text_threshold=0.7 
                )

                number_text = " ".join(ocr_result)
                print(f"OCR Result: {number_text}")

                # 3. *** NEW: Show the debug windows ***
                # This is the MOST important step.
                # You can see what the OCR is "seeing".
                cv2.imshow("Original Number Crop", number_crop)
                cv2.imshow("Preprocessed Input to OCR", preprocessed_crop)

                # Draw results on the scan_display frame
                cv2.rectangle(scan_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # We need to map the number box coords back to the full frame
                num_box_full_x1 = x1 + nx1
                num_box_full_y1 = y1 + ny1
                cv2.putText(scan_display, f"{number_text}", (num_box_full_x1, max(20, num_box_full_y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show the single, analyzed frame
        cv2.imshow("Last Scan Result", scan_display)

cap.release()
cv2.destroyAllWindows()
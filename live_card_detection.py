import cv2
from ultralytics import YOLO
import easyocr

# Load models
card_model = YOLO("runs/detect/card_detector_v13/weights/best.pt")
number_model = YOLO("runs/detect/card_number_detector/weights/best.pt")

# Initialize OCR reader (English numbers)
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if using CUDA

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect cards
    card_results = card_model(frame, verbose=False)

    for card_box in card_results[0].boxes:
        x1, y1, x2, y2 = map(int, card_box.xyxy[0])
        conf_card = float(card_box.conf[0])

        # Crop card
        card_crop = frame[y1:y2, x1:x2]

        # Detect number region inside card
        number_results = number_model(card_crop, verbose=False)

        for num_box in number_results[0].boxes:
            nx1, ny1, nx2, ny2 = map(int, num_box.xyxy[0])
            conf_num = float(num_box.conf[0])

            # Crop the number region for OCR
            number_crop = card_crop[ny1:ny2, nx1:nx2]
            
            #upsize the number region for OCR
            number_crop = cv2.resize(number_crop, (number_crop.shape[1]*2, number_crop.shape[0]*2))

            gray = cv2.cvtColor(number_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            ocr_result = reader.readtext(thresh)

            # OCR
            ocr_result = reader.readtext(number_crop)
            number_text = ""
            if ocr_result:
                # Take the first detected text
                number_text = ocr_result[0][1]

            # Draw number box + OCR text
            cv2.rectangle(card_crop, (nx1, ny1), (nx2, ny2), (0, 0, 255), 2)
            cv2.putText(card_crop, f"{number_text}", (nx1, max(20, ny1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw card box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Card {conf_card:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Card + Number + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

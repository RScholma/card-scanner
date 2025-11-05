import cv2
from ultralytics import YOLO
import numpy as np

# Path to your trained weights
WEIGHTS_PATH = "runs/train/yu_gi_oh_card_numbers2/weights/best.pt"

# Load YOLO model
model = YOLO(WEIGHTS_PATH)


def detect_card_and_number(img):
    """
    Run YOLO detection on an image and return YOLO-format bounding boxes:
    (class_id, x_center, y_center, width, height)
    
    class_id: 0 = full card, 1 = card number
    All coordinates are normalized (0-1).
    """
    results = model(img)  # returns a Results object
    detections = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        classes = r.boxes.cls.cpu().numpy()
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            x_center = x1 + w / 2
            y_center = y1 + h / 2

            # Normalize coordinates by image width/height
            img_h, img_w = img.shape[:2]
            x_center /= img_w
            y_center /= img_h
            w /= img_w
            h /= img_h

            detections.append((int(cls), float(x_center), float(y_center), float(w), float(h)))

    return detections


def test_single_image(image_path):
    """Utility to test one image and print detections."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return
    detections = detect_card_and_number(img)
    print(f"Detections for {image_path}:")
    for det in detections:
        print(det)


if __name__ == "__main__":
    # Example usage: test one image
    test_single_image("dataset/images/train/card_015.jpg")

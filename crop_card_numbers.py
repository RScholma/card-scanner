import os
import cv2
from ultralytics import YOLO

# --------------------------
# Configuration
# --------------------------
yolo_model_path = "runs/detect/card_number_detector/weights/best.pt"  # Your trained number detector
source_images = "dataset/numbers/images/train"  # Folder with card images
output_dir = "dataset/card_number_crops/images"  # Folder to save cropped numbers
os.makedirs(output_dir, exist_ok=True)

# Optional: save labels for OCR training
save_labels = True
labels_dir = "dataset/card_number_crops/labels"
if save_labels and not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

# --------------------------
# Load YOLO model
# --------------------------
model = YOLO(yolo_model_path)

# --------------------------
# Process each image
# --------------------------
image_files = [f for f in os.listdir(source_images) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for img_file in image_files:
    img_path = os.path.join(source_images, img_file)
    img = cv2.imread(img_path)
    
    # Run YOLO detection
    results = model(img)

    # Iterate over detected boxes
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        
        # Only save number crops (class 0 = 'card_number' based on your YAML)
        if cls == 0:
            crop = img[y1:y2, x1:x2]
            
            # Save crop
            base_name = os.path.splitext(img_file)[0]
            crop_name = f"{base_name}_{i}.png"
            cv2.imwrite(os.path.join(output_dir, crop_name), crop)
            
            # Optionally, save label placeholder for OCR training
            if save_labels:
                label_file = os.path.join(labels_dir, f"{base_name}_{i}.txt")
                # You can manually fill in the card number later
                with open(label_file, "w") as f:
                    f.write("PLACEHOLDER")  # Replace with real number later
    print(f"Processed {img_file} ({len(results[0].boxes)} boxes)")

print("Number cropping complete!")

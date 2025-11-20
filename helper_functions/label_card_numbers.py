import cv2
import os

# === CONFIG ===
image_folder = "dataset/card_number_crops/images"   # Folder with cropped images
label_folder = "dataset/card_number_crops/labels"  # Folder to save labels
os.makedirs(label_folder, exist_ok=True)

# === Load images ===
images = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
images.sort()

if not images:
    print("⚠️ No images found in folder:", image_folder)
    exit()

print(f"🖼️ Found {len(images)} images to label.\n")

for i, img_name in enumerate(images):
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Could not open {img_name}")
        continue

    # Resize for visibility if needed
    scale = 3
    h, w = img.shape[:2]
    resized = cv2.resize(img, (w * scale, h * scale))

    # Display image
    cv2.imshow("Label OCR", resized)
    print(f"\n[{i+1}/{len(images)}] File: {img_name}")
    print("💡 Type label in the terminal (e.g. 'ABCD-EN123') or press Enter to skip.")

    # Wait a bit to ensure window refreshes properly
    cv2.waitKey(1)

    # Get label from terminal
    label = input("Enter card number: ").strip()

    if label == "":
        print("⏭️ Skipped.")
    else:
        # Save label
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_folder, base_name + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(label + "\n")
        print(f"✅ Saved label to {label_path}")

    # Wait for small key press before next (press ESC to exit)
    key = cv2.waitKey(500)
    if key == 27:
        print("🚪 Exiting labeling session early.")
        break

    # Close current window before showing next
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
print("\n🎉 All images processed!")
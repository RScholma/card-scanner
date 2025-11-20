import os
import random
import shutil

# === CONFIG ===
base_dir = "dataset/card_number_crops"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

train_images_dir = os.path.join(base_dir, "train", "images")
train_labels_dir = os.path.join(base_dir, "train", "labels")
val_images_dir = os.path.join(base_dir, "val", "images")
val_labels_dir = os.path.join(base_dir, "val", "labels")

split_ratio = 0.8  # 80% train, 20% val

# === SETUP ===
for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
    os.makedirs(d, exist_ok=True)

# === GET FILES ===
images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
images.sort()
random.shuffle(images)

split_index = int(len(images) * split_ratio)
train_files = images[:split_index]
val_files = images[split_index:]

# === HELPER FUNCTION ===
def copy_pair(image_name, dest_img_dir, dest_lbl_dir):
    base_name, _ = os.path.splitext(image_name)
    src_img = os.path.join(images_dir, image_name)
    src_lbl = os.path.join(labels_dir, base_name + ".txt")

    dst_img = os.path.join(dest_img_dir, image_name)
    dst_lbl = os.path.join(dest_lbl_dir, base_name + ".txt")

    # Copy image
    shutil.copy2(src_img, dst_img)
    # Copy label if exists
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)
    else:
        print(f"⚠️ Missing label for {image_name}")

# === SPLIT ===
for img_name in train_files:
    copy_pair(img_name, train_images_dir, train_labels_dir)

for img_name in val_files:
    copy_pair(img_name, val_images_dir, val_labels_dir)

print(f"✅ Done! {len(train_files)} train and {len(val_files)} val samples created.")

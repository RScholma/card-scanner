import os
import re
import shutil
import random

# Base dataset folder
BASE_DIR = "dataset"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

# Subfolders
TRAIN_IMG_DIR = os.path.join(IMAGE_DIR, "train")
VAL_IMG_DIR = os.path.join(IMAGE_DIR, "val")
TRAIN_LBL_DIR = os.path.join(LABEL_DIR, "train")
VAL_LBL_DIR = os.path.join(LABEL_DIR, "val")

# Ensure directories exist
for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
    os.makedirs(d, exist_ok=True)

# Regex to match card filenames like card_001.jpg
card_pattern = re.compile(r"card_(\d+)", re.IGNORECASE)

def get_highest_card_number(*folders):
    """Return the highest numeric index among all card files in the given folders."""
    max_num = 0
    for folder in folders:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            match = card_pattern.search(fname)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
    return max_num

# 1️⃣ Find the highest card number
highest_existing = get_highest_card_number(TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR)
print(f"🔎 Highest existing card number: {highest_existing}")

# 2️⃣ Get new images in dataset/images (not in train/val)
new_images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not new_images:
    print("⚠️ No new images found in dataset/images.")
    exit()

# 3️⃣ Shuffle and split (80% train, 20% val)
random.shuffle(new_images)
split_idx = int(len(new_images) * 0.8)
train_new = new_images[:split_idx]
val_new = new_images[split_idx:]

def process_files(file_list, img_dest, lbl_dest, start_idx):
    idx = start_idx
    for image_file in sorted(file_list):
        idx += 1
        new_name = f"card_{idx:03d}"
        ext = os.path.splitext(image_file)[1]

        old_img = os.path.join(IMAGE_DIR, image_file)
        new_img = os.path.join(img_dest, new_name + ext)

        old_lbl = os.path.join(LABEL_DIR, os.path.splitext(image_file)[0] + ".txt")
        new_lbl = os.path.join(lbl_dest, new_name + ".txt")

        shutil.move(old_img, new_img)
        if os.path.exists(old_lbl):
            shutil.move(old_lbl, new_lbl)
        else:
            open(new_lbl, "w").close()
    return idx

# 4️⃣ Process train and val sets
current_idx = highest_existing
current_idx = process_files(train_new, TRAIN_IMG_DIR, TRAIN_LBL_DIR, current_idx)
current_idx = process_files(val_new, VAL_IMG_DIR, VAL_LBL_DIR, current_idx)

print(f"✅ Finished renaming! Next available card number will be: {current_idx + 1}")
print(f"📁 Train: {len(train_new)} images")
print(f"📁 Val: {len(val_new)} images")

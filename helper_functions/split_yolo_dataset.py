import os
import shutil

# Paths to your original dataset
BASE_DIR = "dataset"   # <-- change this if needed
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")

# New dataset destinations
OUT_DIR_CARD = "dataset/cards"
OUT_DIR_NUMBER = "dataset/numbers"

# Split subsets (train, val)
SUBSETS = ["train", "val"]

def ensure_dirs(base_out):
    for subset in SUBSETS:
        os.makedirs(os.path.join(base_out, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(base_out, "labels", subset), exist_ok=True)

def process_label_file(src_label_path, dst_card_label, dst_number_label):
    with open(src_label_path, "r") as f:
        lines = f.readlines()

    card_lines = []
    number_lines = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(parts[0])
        if class_id == 0:
            card_lines.append(line)
        elif class_id == 1:
            number_lines.append(line)

    # Write separate label files if they have data
    if card_lines:
        with open(dst_card_label, "w") as f:
            f.writelines(card_lines)
    if number_lines:
        with open(dst_number_label, "w") as f:
            f.writelines(number_lines)

def copy_image(src, dst):
    shutil.copy2(src, dst)

# Make output folders
ensure_dirs(OUT_DIR_CARD)
ensure_dirs(OUT_DIR_NUMBER)

for subset in SUBSETS:
    img_subset_dir = os.path.join(IMG_DIR, subset)
    lbl_subset_dir = os.path.join(LBL_DIR, subset)

    for lbl_file in os.listdir(lbl_subset_dir):
        if not lbl_file.endswith(".txt"):
            continue

        img_name = os.path.splitext(lbl_file)[0] + ".jpg"
        img_src_path = os.path.join(img_subset_dir, img_name)
        lbl_src_path = os.path.join(lbl_subset_dir, lbl_file)

        # Output paths
        card_lbl_out = os.path.join(OUT_DIR_CARD, "labels", subset, lbl_file)
        number_lbl_out = os.path.join(OUT_DIR_NUMBER, "labels", subset, lbl_file)
        card_img_out = os.path.join(OUT_DIR_CARD, "images", subset, img_name)
        number_img_out = os.path.join(OUT_DIR_NUMBER, "images", subset, img_name)

        # Split and write labels
        process_label_file(lbl_src_path, card_lbl_out, number_lbl_out)

        # Copy images only if label file had that class
        if os.path.exists(card_lbl_out):
            copy_image(img_src_path, card_img_out)
        if os.path.exists(number_lbl_out):
            copy_image(img_src_path, number_img_out)

print("✅ Done! Split datasets created at:")
print(f"   {OUT_DIR_CARD}")
print(f"   {OUT_DIR_NUMBER}")

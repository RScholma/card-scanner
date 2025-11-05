import os

# Path to your numbers labels folder
LABELS_DIR = "dataset/numbers/labels"

# Loop through train and val
for subset in ["train", "val"]:
    subset_dir = os.path.join(LABELS_DIR, subset)
    if not os.path.exists(subset_dir):
        continue

    for filename in os.listdir(subset_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(subset_dir, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()

        # Replace first number '1' (class ID) with '0'
        new_lines = []
        for line in lines:
            if line.strip():  # skip empty lines
                new_line = line.replace("1", "0", 1)
                new_lines.append(new_line)

        # Overwrite the label file with fixed class IDs
        with open(file_path, "w") as f:
            f.writelines(new_lines)

print("✅ All number labels updated: class 1 → 0")
import requests
from pathlib import Path

BASE_DIR = Path("dataset/images")
SUBFOLDERS = ["train", "val"]
API_URL = "http://127.0.0.1:8000/annotate-card"


def annotate_image(img_path):
    """Send one image to the FastAPI annotation endpoint."""
    with open(img_path, "rb") as f:
        response = requests.post(API_URL, files={"file": f})
    response.raise_for_status()
    data = response.json()
    return data.get("detections", [])


def save_annotations(img_path, detections):
    """Save YOLO-format bounding boxes to .txt next to the image."""
    if not detections:
        print(f"⚠️ No detections for {img_path.name}, skipping...")
        return

    txt_path = img_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        for item in detections:
            try:
                # Handle dict format
                if isinstance(item, dict):
                    cls = float(item.get("cls", 0))
                    x = float(item.get("x", 0))
                    y = float(item.get("y", 0))
                    w = float(item.get("w", 0))
                    h = float(item.get("h", 0))
                # Handle tuple/list format
                else:
                    cls, x, y, w, h = map(float, item)

                f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            except Exception:
                print(f"⚠️ Skipping invalid detection for {img_path.name}: {item}")
                continue

    # Safe string conversion for relative path
    try:
        relative_path = str(txt_path.relative_to(BASE_DIR))
    except ValueError:
        relative_path = str(txt_path)

    print("✅ Saved: " + relative_path)


def main():
    print("🚀 Starting batch annotation...\n")

    for subfolder in SUBFOLDERS:
        folder_path = BASE_DIR / subfolder
        if not folder_path.exists():
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
        if not image_files:
            print(f"⚠️ No images found in {subfolder}/")
            continue

        print(f"📁 Annotating {len(image_files)} images in '{subfolder}/'...")

        for img_path in image_files:
            try:
                detections = annotate_image(img_path)
                save_annotations(img_path, detections)
            except requests.exceptions.RequestException as e:
                print(f"❌ Request error for {img_path.name}: {e}")
            except Exception as e:
                print(f"❌ Unexpected error for {img_path.name}: {str(e)}")

    print("\n✅ Batch annotation complete!")


if __name__ == "__main__":
    main()

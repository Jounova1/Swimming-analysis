import os
import cv2
import random
from pathlib import Path
from ultralytics import YOLO

# =========================
# ⚙️ CONFIG
# =========================
VIDEOS_DIR = r"C:\Swimming-analysis\ai\videos"
OUTPUT_DIR = r"C:\Swimming-analysis\ai\fins_dataset"
MODEL_PATH = r"C:\Swimming-analysis\ai\training\runs\train\yolo11m_swimmer_finetune_v2\weights\best.pt"

FRAME_SKIP = 5
IMG_SIZE = 640

SPLIT = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

# class ids
# 0 = swimmer
# 1 = fins_swimmer

# =========================
# 📁 CREATE STRUCTURE
# =========================
def create_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/temp", exist_ok=True)

# =========================
# 🎥 EXTRACT FRAMES
# =========================
def extract_frames():
    frames = []
    for video_file in os.listdir(VIDEOS_DIR):
        path = os.path.join(VIDEOS_DIR, video_file)
        cap = cv2.VideoCapture(path)

        frame_id = 0
        saved_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % FRAME_SKIP == 0:
                filename = f"{video_file}_{saved_id}.jpg"
                save_path = f"{OUTPUT_DIR}/temp/{filename}"

                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(save_path, frame)

                frames.append(save_path)
                saved_id += 1

            frame_id += 1

        cap.release()

    print(f"🎥 Extracted {len(frames)} frames")
    return frames

# =========================
# 🧠 AUTO LABEL (DOUBLE BOX)
# =========================
def auto_label(frames):
    model = YOLO(MODEL_PATH)
    labeled = []

    for img_path in frames:
        results = model(img_path, conf=0.4)

        label_path = img_path.replace(".jpg", ".txt")

        with open(label_path, "w") as f:
            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.4:
                        continue

                    xywh = box.xywhn[0].tolist()
                    x, y, w, h = xywh

                    # =========================
                    # 🟢 swimmer (body only)
                    # =========================
                    h_swimmer = h * 0.8
                    y_swimmer = y - (h - h_swimmer) / 2

                    f.write(f"0 {x} {y_swimmer} {w} {h_swimmer}\n")

                    # =========================
                    # 🔵 fins_swimmer (extended)
                    # =========================
                    h_fins = h * 1.2
                    y_fins = y + (h_fins - h) / 2

                    f.write(f"1 {x} {y_fins} {w} {h_fins}\n")

        labeled.append((img_path, label_path))

    print("🧠 Auto labeling done")
    return labeled

# =========================
# 🔀 SPLIT DATA
# =========================
def split_data(data):
    random.shuffle(data)

    n = len(data)
    train_end = int(n * SPLIT["train"])
    val_end = int(n * (SPLIT["train"] + SPLIT["val"]))

    splits = {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:]
    }

    for split, items in splits.items():
        for img_path, label_path in items:
            img_name = os.path.basename(img_path)
            lbl_name = os.path.basename(label_path)

            new_img = f"{OUTPUT_DIR}/{split}/images/{img_name}"
            new_lbl = f"{OUTPUT_DIR}/{split}/labels/{lbl_name}"

            os.rename(img_path, new_img)
            os.rename(label_path, new_lbl)

    print("🔀 Dataset split done")

# =========================
# 📄 CREATE YAML
# =========================
def create_yaml():
    import yaml

    data = {
        "path": OUTPUT_DIR.replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": ["swimmer", "fins_swimmer"]
    }

    with open(f"{OUTPUT_DIR}/dataset.yaml", "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print("📄 dataset.yaml created")

# =========================
# 🚀 MAIN
# =========================
def main():
    print("🚀 Creating fins dataset...")

    create_dirs()
    frames = extract_frames()
    labeled = auto_label(frames)
    split_data(labeled)
    create_yaml()

    print("\n✅ fins_dataset جاهز بالكامل!")

if __name__ == "__main__":
    main()
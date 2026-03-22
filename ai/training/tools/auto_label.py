from ultralytics import YOLO
import cv2
import os

# موديل جاهز
model = YOLO("yolov8l.pt")  # pretrained

images_path = "dataset/images/train"
labels_path = "dataset/labels/train"

os.makedirs(labels_path, exist_ok=True)

for img_name in os.listdir(images_path):
    img_path = os.path.join(images_path, img_name)

    results = model.predict(img_path, conf=0.3)

    h, w = cv2.imread(img_path).shape[:2]

    label_file = os.path.join(labels_path, img_name.replace(".jpg", ".txt"))

    with open(label_file, "w") as f:
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]

                # convert to YOLO format
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                f.write(f"0 {x_center} {y_center} {bw} {bh}\n")
import os
import cv2

# عدل ده حسب مكان الداتا عندك
base_path = r"C:\Swimming-analysis\ai\training\dataset"
split = "train"  # train / valid / test

images_path = os.path.join(base_path, split, "images")
labels_path = os.path.join(base_path, split, "labels")

for file in os.listdir(images_path):
    if file.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(images_path, file)
        label_path = os.path.join(labels_path, file.replace(".jpg", ".txt").replace(".png", ".txt"))

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                cls = int(parts[0])
                x, y, bw, bh = map(float, parts[1:])

                # convert YOLO → pixel
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                # colors
                if cls == 0:
                    color = (0, 0, 255)  # 🔴 Red
                    label = "Class 0"
                elif cls == 1:
                    color = (0, 255, 0)  # 🟢 Green
                    label = "Class 1"
                else:
                    color = (255, 255, 255)
                    label = f"Class {cls}"

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Check Labels", img)

        key = cv2.waitKey(0)
        if key == 27:  # ESC عشان تخرج
            break

cv2.destroyAllWindows()
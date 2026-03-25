from ultralytics import YOLO

model = YOLO(r"C:\Swimming-analysis\ai\training\runs\train\yolo11m_swimmer_finetune_v2\weights\best.pt")
results = model.predict(
    source=r"C:\Swimming-analysis\ai\fins_dataset\train\images\flip_turn.mp4_13.jpg",
    conf=0.5,
    save=True,
    imgsz=640
)

print(results[0].boxes)
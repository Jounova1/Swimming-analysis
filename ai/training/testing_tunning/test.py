from ultralytics import YOLO

model = YOLO("last.pt")
results = model.predict(
    source="T100000.jpg",
    conf=0.05,
    save=True,
    imgsz=640
)

print(results[0].boxes)
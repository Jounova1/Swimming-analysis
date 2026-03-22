from ultralytics import YOLO

model = YOLO("last.pt")

results = model.predict(
    source="T100000.jpg",
    conf=0.25,
    save=True
)
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="bf_0012.jpg",
    conf=0.1,
    save=True
)
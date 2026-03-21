from ultralytics import YOLO

model = YOLO("best copy.pt")

results = model.predict(
    source="flip_turn_0000.jpg",
    conf=0.1,
    save=True
)
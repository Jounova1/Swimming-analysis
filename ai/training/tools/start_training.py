import ultralytics

model = YOLO("yolpv8m.pt")

model.train(
    data= "data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)
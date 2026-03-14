from ultralytics import YOLO

# load base model
model = YOLO("yolov8n.pt")

# start training
model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)

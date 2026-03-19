<<<<<<< HEAD
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
=======
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
>>>>>>> c2eba27109d5ce886fe725621475426ae1166196

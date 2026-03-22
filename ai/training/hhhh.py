from ultralytics import YOLO

model = YOLO("last.pt")

results = model.predict(
    source="T100000.jpg",
    conf=0.25,# confidence threshold for detections, adjust as needed (e.g. 0.25 or 0.45)   
    save=True # set to True to save the annotated image with detections, adjust as needed
)
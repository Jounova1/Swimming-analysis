from ultralytics import YOLO

model = YOLO(r"D:\Swimming-analysis\runs\train\yolo11m_swimmer_clean\weights\best.pt")

results = model.predict(
    source="flip_turn_0000.jpg",
    conf=0.25,# confidence threshold for detections, adjust as needed (e.g. 0.25 or 0.45)   
    save=True # set to True to save the annotated image with detections, adjust as needed
)
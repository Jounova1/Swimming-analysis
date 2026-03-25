from ultralytics import YOLO

model = YOLO(r"C:\Swimming-analysis\ai\training\runs\train\yolo11m_swimmer_finetune_v2\weights\best.pt")

results = model.predict(
    source=r"C:\Swimming-analysis\ai\fins_dataset\train\images\flip_turn.mp4_5.jpg",
    conf=0.25,# confidence threshold for detections, adjust as needed (e.g. 0.25 or 0.45)   
    save=True # set to True to save the annotated image with detections, adjust as needed
)
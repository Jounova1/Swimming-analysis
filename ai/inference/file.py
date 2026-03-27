fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    r"C:\Users\original\Desktop\output.mp4",
    fourcc,
    fps,
    (INPUT_WIDTH, INPUT_HEIGHT)
)

if not writer.isOpened():
    print("❌ writer failed")
    exit()
import cv2

path = r"C:\Swimming-analysis\Swimming-analysis\ai\training\videoplayback (1).mp4"

cap = cv2.VideoCapture(path)
print("Opened =", cap.isOpened())

while True:
    ret, frame = cap.read()
    print("ret =", ret)
    if not ret:
        break
    cv2.imshow("test", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
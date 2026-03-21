import cv2

img = cv2.imread("flip_turn_0000.jpg")
h, w, _ = img.shape

with open("flip_turn_0000.txt") as f:
    for line in f:
        c, x, y, bw, bh = map(float, line.split())
        
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
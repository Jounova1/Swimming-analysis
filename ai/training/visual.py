import cv2 

img = cv2.imread("T100000.jpg")
h, w, _ = img.shape # get image dimensions for converting YOLO format to pixel coordinates

with open("T100000.txt") as f:
    for line in f:
        c, x, y, bw, bh = map(float, line.split()) # read class and bounding box coordinates from YOLO format (class, x_center, y_center, width, height)
        
        x1 = int((x - bw/2) * w) # convert from YOLO format to pixel coordinates
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2) # draw bounding box on image

cv2.imshow("img", img) # display the image with bounding boxes
cv2.waitKey(0)
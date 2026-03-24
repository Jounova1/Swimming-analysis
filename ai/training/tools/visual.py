import cv2 

img = cv2.imread(r"C:\Swimming-analysis\ai\training\Swimmer Detection.v1i.yolov11\train\images\Booker-Start-Breast-11_01_2024-08_36_20_5_Edited_swimmer_detected_frame_000004_PNG.rf.50943088d00ec0fc1a4fb588baa412a2.jpg")
h, w, _ = img.shape # get image dimensions for converting YOLO format to pixel coordinates

with open(r"C:\Swimming-analysis\ai\training\Swimmer Detection.v1i.yolov11\train\labels\Booker-Start-Breast-11_01_2024-08_36_20_5_Edited_swimmer_detected_frame_000004_PNG.rf.50943088d00ec0fc1a4fb588baa412a2.txt") as f:
    for line in f:
        c, x, y, bw, bh = map(float, line.split()) # read class and bounding box coordinates from YOLO format (class, x_center, y_center, width, height)
        
        x1 = int((x - bw/2) * w) # convert from YOLO format to pixel coordinates
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2) # draw bounding box on image

cv2.imshow("img", img) # display the image with bounding boxes
cv2.waitKey(0)
import numpy as np
from ultralytics import YOLO
import cv2
import support
from sort import *

capture = cv2.VideoCapture("../dataset/cars.mp4")  # Video
mask = cv2.imread("./mask.png")  # Mask

model = YOLO("../weight/yolov8l.pt")

line = [400, 297, 673, 297]  # size in pixel, line that will be consider to count up if the car pass it

# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCount = []

while True:
    success, img = capture.read()
    masked_capture = cv2.bitwise_and(img, mask)  # Mask Video to ignore the area where we are not interested in before send to network
    result = model(masked_capture, stream=True)

    detections = np.empty((0, 4))

    for r in result:
        boxes = r.boxes
        for eachbox in boxes:
            x1, y1, x2, y2 = eachbox.xyxy[0]
            confident = eachbox.conf[0]
            classes_number = int(eachbox.cls[0].item())
            classes_name = model.names[classes_number]

            if classes_name == ("car" or "truck" or "bus") and confident > 0.3:  # check only car to detect
                currentArray = np.array([int(x1), int(y1), int(x2), int(y2)])
                detections = np.vstack((detections, currentArray))
                support.draw_rectangle(masked_capture, x1, y1, x2, y2)  # rectangle over car
                support.draw_text(masked_capture, classes_name + str(round(confident.item(), 2)), x1, y1)  # name of class

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        print(result)
        boxh = y2 - y1  # find box height
        boxw = x2 - x1  # find box width
        cross_line = support.linerectangle_collsion(line[0], line[1], line[2], line[3], x1, y1, boxh, boxw)  # check if the car is cross touch the line
        if cross_line:
            if totalCount.count(id) == 0:
                totalCount.append(id)
            crossed_status = "Crossed"
        else:
            crossed_status = "--"

    support.draw_line(masked_capture, line)
    support.draw_text(masked_capture, "Counter = " + str(len(totalCount)), 0, 40)
    support.draw_text(masked_capture, "Cross Status = " + crossed_status, 0, 70)

    cv2.imshow("Masked", masked_capture)
    cv2.waitKey(1)

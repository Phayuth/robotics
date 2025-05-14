import numpy as np
from ultralytics import YOLO
import cv2
import support
from sort import *

capture = cv2.VideoCapture('../dataset/people.mp4')
mask = cv2.imread('./mask.png')

model = YOLO('../weight/yolov8l.pt')

line_up = [103,161,296,161]
line_down = [527, 489, 735, 489]

# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCount_up = []
totalCount_down = []

while True:
    success, img = capture.read()
    masked_capture = cv2.bitwise_and(img,mask)
    result = model(masked_capture, stream=True)

    detections = np.empty((0, 4))

    for r in result:
        boxes = r.boxes
        for eachbox in boxes:
            x1, y1, x2, y2 = eachbox.xyxy[0]
            confident = eachbox.conf[0]
            classes_number = int(eachbox.cls[0].item())
            classes_name = model.names[classes_number]

            if classes_name == "person" and confident > 0.3:
                currentArray = np.array([int(x1), int(y1), int(x2), int(y2)])
                detections = np.vstack((detections, currentArray))
                support.draw_rectangle(masked_capture,x1,y1,x2,y2)
                support.draw_text(masked_capture,classes_name+str(round(confident.item(),2)),x1,y1)

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        boxh = y2-y1
        boxw = x2-x1
        cross_line_up = support.linerectangle_collsion(line_up[0],line_up[1],line_up[2],line_up[3],x1,y1,boxh,boxw)
        cross_line_down = support.linerectangle_collsion(line_down[0],line_down[1],line_down[2],line_down[3],x1,y1,boxh,boxw)

        if cross_line_up:
            if totalCount_up.count(id) == 0:
                totalCount_up.append(id)

        if cross_line_down:
            if totalCount_down.count(id) == 0:
                totalCount_down.append(id)


    support.draw_line(masked_capture,line_up)
    support.draw_line(masked_capture,line_down)
    support.draw_text(masked_capture,"Up Counter = " +str(len(totalCount_up)),0,40)
    support.draw_text(masked_capture,"Down Counter = " +str(len(totalCount_down)),0,70)

    cv2.imshow("Masked",masked_capture)
    cv2.waitKey(1)
import ultralytics
import cv2

# Camera
capture = cv2.VideoCapture(4)
capture.set(3, 1280)
capture.set(4, 720)

# Video
# capture = cv2.VideoCapture('./dataset/cars.mp4')
model = ultralytics.YOLO('./target_localization/neuralnet/weight/yolov8l.pt')

while True:
    success, img = capture.read()
    result = model(img, stream=True, conf=0.5)
    for r in result:  # loop to find boxes info in result
        boxes = r.boxes  # result give multiple boxes
        for eachbox in boxes:  # for each box in boxes
            x1, y1, x2, y2 = eachbox.xyxy[0]  # give xmin,ymin,xmax,ymax
            confident = eachbox.conf[0]  # give confident of each box
            classes_number = int(eachbox.cls[0].item())  # give class number in tensor -> get item in tensor -> convert to int
            classes_name = model.names[classes_number]  # give class name from model in dictionary
            cv2.putText(img,
                        text=classes_name + str(round(confident.item(), 2)),
                        org=(int(x1), int(y1 + (y2-y1) / 2)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        thickness=2,
                        color=(255, 0, 255))  # put text on the box
            cv2.rectangle(img, 
                          pt1=(int(x1), int(y1)), 
                          pt2=(int(x2), int(y2)), 
                          color=(255, 0, 255), 
                          thickness=3)  # draw box on img
    cv2.imshow("Image", img)
    cv2.waitKey(1)
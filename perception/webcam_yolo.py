import ultralytics
import cv2
import numpy as np

from perception.webcam import CVCamera

# Video
cam = CVCamera()
model = ultralytics.YOLO('./datasave/neural_weight/yolov8x-seg.pt')


# Everything detection
def detect_everything():
    while cam.capture.isOpened():
        success, frame = cam.capture.read()
        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cam.capture.release()
    cv2.destroyAllWindows()


# Everything mask only
def detect_mask():
    while cam.capture.isOpened():
        success, img = cam.capture.read()
        imgShow = np.ones_like(img)
        imgMask = np.zeros((img.shape[0], img.shape[1]))
        result = model(img, stream=True, conf=0.5)
        for r in result:
            boxes = r.boxes
            masks = r.masks
            for bi, box in enumerate(boxes):
                classes_number = int(box.cls[0].item())
                classes_name = model.names[classes_number]
                print(f"class name ={classes_name}, index = {bi}")
                eachMask = masks.masks[bi, :, :].cpu().numpy()
                eachMask = eachMask.reshape(eachMask.shape[0], eachMask.shape[1])
                imgMask = imgMask + eachMask
            imgshow = np.where(imgMask[..., None], img, 0)
            cv2.imshow("Everything mask only", imgshow)
            cv2.waitKey(1)


# Apple mask only
def apple_mask():
    while cam.capture.isOpened():
        success, img = cam.capture.read()
        imgShow = np.ones_like(img)
        imgMask = np.zeros((img.shape[0], img.shape[1]))
        result = model(img, stream=True, conf=0.5)
        for r in result:
            boxes = r.boxes
            masks = r.masks
            for bi, box in enumerate(boxes):
                classes_number = int(box.cls[0].item())
                classes_name = model.names[classes_number]
                if classes_name == "apple":
                    appleMask = masks.masks[bi, :, :].cpu().numpy()
                    appleMask = appleMask.reshape(appleMask.shape[0], appleMask.shape[1])
                    imgMask = imgMask + appleMask
        imgshow = np.where(imgMask[..., None], img, 0)
        cv2.imshow("Apple mask only", imgshow)
        cv2.waitKey(0)


def normal():
    # Camera
    capture = cv2.VideoCapture(4)
    capture.set(3, 1280)
    capture.set(4, 720)

    # Video
    # capture = cv2.VideoCapture('./dataset/cars.mp4')
    model = ultralytics.YOLO('./datasave/neural_weight/yolov8l.pt')

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
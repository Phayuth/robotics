import ultralytics
import cv2
import numpy as np
import time

# Camera
capture = cv2.VideoCapture(4)
h = 480
w = 640
capture.set(3, h)
capture.set(4, w)

# Video
model = ultralytics.YOLO('./target_localization/neuralnet/weight/yolov8x-seg.pt')

# Everythind detection
# while capture.isOpened():
#     success, frame = capture.read()
#     if success:
#         results = model(frame)
#         annotated_frame = results[0].plot()
#         cv2.imshow("YOLOv8 Inference", annotated_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break
# capture.release()
# cv2.destroyAllWindows()


# Everything mask only
# while capture.isOpened():
#     success, img = capture.read()
#     imgShow = np.ones_like(img)
#     imgMask = np.zeros((img.shape[0], img.shape[1]))
#     result = model(img, stream=True, conf=0.5)
#     for r in result:
#         boxes = r.boxes
#         masks = r.masks
#         for bi, box in enumerate(boxes):
#             classes_number = int(box.cls[0].item())
#             classes_name = model.names[classes_number]
#             print(f"class name ={classes_name}, index = {bi}")
#             eachMask = masks.masks[bi, :, :].cpu().numpy()
#             eachMask = eachMask.reshape(eachMask.shape[0], eachMask.shape[1])
#             imgMask = imgMask + eachMask
#         imgshow = np.where(imgMask[..., None], img, 0)
#         cv2.imshow("Everything mask only", imgshow)
#         cv2.waitKey(1)


# Apple mask only
while capture.isOpened():
    success, img = capture.read()
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
    # np.save("./mask.npy", imgMask)
    imgshow = np.where(imgMask[..., None], img, 0)
    cv2.imshow("Apple mask only", imgshow)
    cv2.waitKey(0)
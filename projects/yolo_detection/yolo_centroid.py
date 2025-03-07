import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import ultralytics
import cv2
import numpy as np


class CVYOLOCentroid:

    def __init__(self, torchModelPath, interestNames) -> None:
        self.model = ultralytics.YOLO(torchModelPath)
        self.interestNames = interestNames
        self.erodeKernel = np.ones((7, 7), np.uint8)
        # print(f"Available Class Name are : {self.model.names}")

    def detect_centroid(self, img, edgeErode=False, drawImg=False):
        indvMask = []
        result = self.model(img, stream=True, conf=0.5, verbose=False)
        for r in result:
            for bi, box in enumerate(r.boxes):
                classesNumber = int(box.cls[0].item())
                classesName = self.model.names[classesNumber]
                for interestName in self.interestNames:
                    if classesName == interestName:
                        # imgMask += r.masks.masks[bi, :, :].cpu().numpy()
                        if edgeErode:
                            indvMask.append(cv2.erode(r.masks.masks[bi, :, :].cpu().numpy(), self.erodeKernel, cv2.BORDER_REFLECT))
                        else:
                            indvMask.append(r.masks.masks[bi, :, :].cpu().numpy())

        indvCentroid = None
        if len(indvMask) != 0:
            indvCentroid = []
            for objMask in indvMask:
                contours, _ = cv2.findContours(objMask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    indvCentroid.append((cx, cy))

                    if drawImg:
                        cv2.drawContours(img, contours[0], -1, (0, 255, 0), 3)
                        cv2.circle(img, (cx, cy), 3, (255, 0, 255), 3)
                        cv2.putText(img, text=f"centroid at {cx}, {cy}", org=(cx, cy), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, color=(255, 0, 255))

        return indvCentroid


if __name__ == "__main__":
    from camera.v4l2camera import CVCamera

    cam = CVCamera(4)

    torchModelPath = "./datasave/neural_weight/yolov8x-seg.pt"
    cvc = CVYOLOCentroid(torchModelPath, interestNames=["wine glass", "cup"])

    while cam.capture.isOpened():
        success, img = cam.capture.read()
        indvCentroid = cvc.detect_centroid(img, edgeErode=True, drawImg=True)

        cv2.imshow("Apple mask only", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.capture.release()
    cv2.destroyAllWindows()

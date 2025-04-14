import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import ultralytics
import cv2
import numpy as np


class CVYOLOMask:

    def __init__(self, torchModelPath, interestNames) -> None:
        self.model = ultralytics.YOLO(torchModelPath)
        self.interestNames = interestNames
        self.erodeKernel = np.ones((6, 6), np.uint8)
        self.conf = 0.5
        print(f"Available Class Name are : {self.model.names}")

    def detect_mask(self, img, edgeErode=False, drawImg=False):
        indvMask = []
        result = self.model(img, stream=True)
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

        imgshow = None
        if drawImg:
            mergeOverallMask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool8)
            if len(indvMask) != 0:
                for mi in indvMask:
                    mergeOverallMask = np.logical_or(mergeOverallMask, mi)
            imgshow = np.where(mergeOverallMask[..., None], img, 0)

        return indvMask, imgshow

    def detect_mask_all(self, img, edgeErode=False, drawImg=False):
        indvMask = []
        result = self.model(img, stream=True, conf=0.5)
        for r in result:
            for bi, box in enumerate(r.boxes):
                classesNumber = int(box.cls[0].item())
                classesName = self.model.names[classesNumber]
                print(f"class name ={classesName}, index = {bi}")
                if edgeErode:
                    indvMask.append(cv2.erode(r.masks.masks[bi, :, :].cpu().numpy(), self.erodeKernel, cv2.BORDER_REFLECT))
                else:
                    indvMask.append(r.masks.masks[bi, :, :].cpu().numpy())

        imgshow = None
        if drawImg:
            mergeOverallMask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool8)
            if len(indvMask) != 0:
                for mi in indvMask:
                    mergeOverallMask = np.logical_or(mergeOverallMask, mi)
            imgshow = np.where(mergeOverallMask[..., None], img, 0)

        return indvMask, imgshow


if __name__ == "__main__":
    from camera.v4l2camera import CVCamera

    cam = CVCamera(4)

    torchModelPath = "/home/yuth/ws_yuthdev/neural_network/datasave/neural_weight/yolov8x-seg.pt"
    cvc = CVYOLOMask(torchModelPath, interestNames=["apple"])

    while cam.capture.isOpened():
        success, img = cam.capture.read()
        indvMask, imgshow = cvc.detect_mask(img, edgeErode=True, drawImg=True)

        cv2.imshow("Apple mask only", imgshow)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.capture.release()
    cv2.destroyAllWindows()

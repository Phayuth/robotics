import cv2
import matplotlib.pyplot as plt


class CVCamera:

    def __init__(self, camID=0, width=640, height=480) -> None:
        self.capture = cv2.VideoCapture(camID)
        self.capture.set(4, width)
        self.capture.set(3, height)
        self.camProperties = {}

    def get_cam_property(self):
        print(f"CAP_PROP_FRAME_WIDTH : {self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"CAP_PROP_FRAME_HEIGHT: {self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"CAP_PROP_FPS         : {self.capture.get(cv2.CAP_PROP_FPS)}")
        print(f"CAP_PROP_POS_MSEC    : {self.capture.get(cv2.CAP_PROP_POS_MSEC)}")
        print(f"CAP_PROP_FRAME_COUNT : {self.capture.get(cv2.CAP_PROP_FRAME_COUNT)}")
        print(f"CAP_PROP_BRIGHTNESS  : {self.capture.get(cv2.CAP_PROP_BRIGHTNESS)}")
        print(f"CAP_PROP_CONTRAST    : {self.capture.get(cv2.CAP_PROP_CONTRAST)}")
        print(f"CAP_PROP_SATURATION  : {self.capture.get(cv2.CAP_PROP_SATURATION)}")
        print(f"CAP_PROP_HUE         : {self.capture.get(cv2.CAP_PROP_HUE)}")
        print(f"CAP_PROP_GAIN        : {self.capture.get(cv2.CAP_PROP_GAIN)}")
        print(f"CAP_PROP_CONVERT_RGB : {self.capture.get(cv2.CAP_PROP_CONVERT_RGB)}")


if __name__ == "__main__":
    cam = CVCamera()
    success, imgBGR = cam.capture.read()
    cam.get_cam_property()
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    plt.imshow(imgRGB)
    plt.show()

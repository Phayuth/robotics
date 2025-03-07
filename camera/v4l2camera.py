import cv2


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


def single_shot():
    cam = CVCamera(4)
    success, imgRGB = cam.capture.read()
    cam.get_cam_property()
    # imgRGB = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB)
    cv2.imshow("Everything mask only", imgRGB)
    cv2.waitKey(0)
    cam.capture.release()
    cv2.destroyAllWindows()


def stream():
    cam = CVCamera(4)
    while cam.capture.isOpened():
        success, img = cam.capture.read()
        if success:
            cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream()

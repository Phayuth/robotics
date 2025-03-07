import cv2
import numpy as np

print(cv2.__version__)
print(dir(cv2.aruco))

mtx = np.array([[523.971998, 0.000000, 632.455814],
                [0.000000, 525.425192, 364.281755],
                [0.000000, 0.000000, 1.000000]])
dist = np.array([0.019854, -0.036200, -0.001037, 0.001803, 0.000000])

dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dict_aruco, parameters)

cap = cv2.VideoCapture(4)
while True:
    ret, image_raw = cap.read()
    if ret:
        h, w = image_raw.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        image_undst = cv2.undistort(image_raw, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        image_undst = image_undst[y : y + h, x : x + w]
        # dstroi = dst[y:y + h, x:x + w]

        cv2.imshow("raw", image_raw)
        # cv2.imshow('undistort', dst)
        # cv2.imshow('undistort and roi', dstroi)

        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image_undst)
        if markerIds is not None:
            cv2.aruco.drawDetectedMarkers(image_undst, markerCorners, markerIds)

        cv2.imshow("detect", image_undst)
        if cv2.waitKey(2) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

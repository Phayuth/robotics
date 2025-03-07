import cv2
import numpy as np

print(cv2.__version__)
print(f"aruco dirc = {dir(cv2.aruco)}")

cameraMatrix = np.array([[523.971998, 0.000000, 632.455814],
                         [0.000000, 525.425192, 364.281755],
                         [0.000000, 0.000000, 1.000000]])
distCoeffs = np.array([0.019854, -0.036200, -0.001037, 0.001803, 0.000000])

boardgrid = (5, 7)
squareLength = 0.0353  # m
markerLength = 0.0256  # m

detectorParams = cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

cap = cv2.VideoCapture(0)
while True:
    ret, image_raw = cap.read()
    if ret:
        h, w = image_raw.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
        image_undst = cv2.undistort(image_raw, cameraMatrix, distCoeffs, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        image_undst = image_undst[y : y + h, x : x + w]

        corners, ids, rej = detector.detectMarkers(image_undst)
        if not ids is None:
            cv2.aruco.drawDetectedMarkers(image_undst, corners, ids)  # aruco corner

            for i in range(len(ids)):
                marker_points = np.array([[-markerLength / 2.0, markerLength / 2.0, 0],
                                          [markerLength / 2.0, markerLength / 2.0, 0],
                                          [markerLength / 2.0, -markerLength / 2.0, 0],
                                          [-markerLength / 2.0, -markerLength / 2.0, 0]], dtype=np.float32)
                retval, rvc, tvc = cv2.solvePnP(marker_points, corners[i], cameraMatrix, distCoeffs, None, None, False, cv2.SOLVEPNP_IPPE_SQUARE)
                print(f"> retval: {retval}")
                print(f"> rvc: {rvc}")
                print(f"> tvc: {tvc}")
                rot_mat, _ = cv2.Rodrigues(rvc)
                print(f"> rot_mat: {rot_mat}")

                if retval:
                    cv2.drawFrameAxes(image_undst, cameraMatrix, distCoeffs, rvc, tvc, markerLength/2.0, 1)

        cv2.imshow("detect", image_undst)
        cv2.imshow("raw", image_raw)
        if cv2.waitKey(2) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

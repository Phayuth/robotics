import cv2
import numpy as np

print(cv2.__version__)
print(f"aruco dirc = {dir(cv2.aruco)}")

cameraMatrix = np.array([[598.460339, 0.000000, 317.880979], [0.000000, 597.424060, 233.262422], [0.000000, 0.000000, 1.000000]])
distCoeffs = np.array([0.142729, -0.282139, -0.005699, -0.012027, 0.000000])

boardgrid = (5, 7)
squareLength = 0.0353  # m
markerLength = 0.0256  # m

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard(boardgrid, squareLength, markerLength, dictionary)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.CharucoDetector(board)

generate = True
if generate == True:
    image = board.generateImage(outSize=(400, 500), marginSize=10, borderBits=1)

    import matplotlib.pyplot as plt

    plt.imshow(image, cmap="gray")
    plt.show()

cap = cv2.VideoCapture(4)
while True:
    ret, image_raw = cap.read()
    if ret:
        h, w = image_raw.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
        image_undst = cv2.undistort(image_raw, cameraMatrix, distCoeffs, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        image_undst = image_undst[y : y + h, x : x + w]
        # dstroi = dst[y:y + h, x:x + w]
        # cv2.imshow('undistort', dst)
        # cv2.imshow('undistort and roi', dstroi)

        # markerCorners, markerIds, rejectedCandidates = detector.detectBoard(image_undst)
        charucoCorners, charucoIds, markerCorners, markerIds = detector.detectBoard(image_undst)
        # charucoCorners	interpolated chessboard corners.
        # charucoIds	interpolated chessboard corners identifiers.
        # markerCorners	vector of already detected markers corners. For each marker, its four corners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of this array should be Nx4. The order of the corners should be clockwise. If markerCorners and markerCorners are empty, the function detect aruco markers and ids.
        # markerIds	list of identifiers for each marker in corners. If markerCorners and markerCorners are empty, the function detect aruco markers and ids.

        if not charucoIds is None:
            cv2.aruco.drawDetectedCornersCharuco(image_undst, charucoCorners, charucoIds)  # chess corner
            cv2.aruco.drawDetectedMarkers(image_undst, markerCorners, markerIds)  # aruco corner

        # estimate all board z should point outward from paper
        valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, None, None, False)
        if valid:
            cv2.drawFrameAxes(image_undst, cameraMatrix, distCoeffs, rvec, tvec, 0.1)

        cv2.imshow("detect", image_undst)
        cv2.imshow("raw", image_raw)
        if cv2.waitKey(2) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

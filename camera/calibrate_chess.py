import cv2
import numpy as np

cap = cv2.VideoCapture(4)

pattern_size = (10, 7)

samples = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    res, corners = cv2.findChessboardCorners(frame, pattern_size)
    if res:
        print(f"> corners.shape: {corners.shape}")
        print(corners[0])
    img_show = np.copy(frame)
    cv2.drawChessboardCorners(img_show, pattern_size, corners, res)
    cv2.putText(img_show, "Samples captured: %d" % len(samples), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("chessboard", img_show)

    wait_time = 0 if res else 30
    k = cv2.waitKey(wait_time)

    if k == ord("s") and res:
        samples.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners))
    elif k == 27:
        break

cap.release()
cv2.destroyAllWindows()


# raise SystemExit(0)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

for i in range(len(samples)):
    img, corners = samples[i]
    corners = cv2.cornerSubPix(img, corners, (10, 10), (-1, -1), criteria)


pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
print(f"> pattern_points.shape: {pattern_points.shape}")
print(f"> pattern_points: {pattern_points}")
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
print(f"> pattern_points: {pattern_points}")

# raise SystemExit(0)

images, corners = zip(*samples)

pattern_points = [pattern_points] * len(corners)

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(pattern_points, corners, images[0].shape, None, None)

print(camera_matrix)
print(dist_coefs)

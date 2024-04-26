import cv2 as cv2
from cv2 import aruco
import numpy as np


def aruco_setup():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector


cap = cv2.VideoCapture(1)
detector = aruco_setup()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    marker_corners, marker_IDs, reject = detector.detectMarkers(frame)

    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            cv2.putText(frame, str(*ids), bottom_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print(ids)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

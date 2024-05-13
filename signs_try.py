import cv2
from cv2 import aruco
import numpy as np
import imutils
import socket
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import time


last_command = '0'


class Img:
    cap = None
    raw_img = None
    img4masking = None
    blue_mask = None
    circles = None
    circ_mask = None
    lines = None
    circ_centers = None
    final_direction = [0, '']
    # total_mask = None
    contours = []
    centers = []
    ans_lst = []
    ans = ''

    def __init__(self):
        self.cap = cv2.VideoCapture(1)

    def get_raw_img(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.raw_img = frame

    def img_processing(self):
        def img_blur():
            blurred = cv2.medianBlur(self.raw_img, 7)
            blurred = cv2.blur(blurred, (21, 21))
            ret, thresh4 = cv2.threshold(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_TOZERO)
            return thresh4

        # def img_thresh():
        #     grayed = cv2.cvtColor(tmp_blured, cv2.COLOR_RGB2GRAY)
        #     threshed = cv2.adaptiveThreshold(grayed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
        #     return threshed

        self.img4masking = img_blur()

    def img_masking(self):

        self.centers = []
        self.contours = []

        def create_blue_mask():
            lower_limit = np.array([100, 60, 80])
            upper_limit = np.array([128, 255, 255])

            self.blue_mask = cv2.inRange(cv2.cvtColor(self.raw_img, cv2.COLOR_RGB2HSV), lower_limit, upper_limit)

        create_blue_mask()

    def img_circles(self):

        self.circ_mask = np.zeros_like(self.img4masking)

        img = apply_mask(self.blue_mask, self.img4masking)
        img = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 35, 125)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        edges = cv2.Canny(img, 170, 190)

        self.circles = img
        # self.img4masking = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_circles = cv2.HoughCircles\
            (edges , cv2.HOUGH_GRADIENT, 1, 500, param1=50, param2=35, minRadius=10, maxRadius=0)

        # Draw circles that are detected.
        if detected_circles is not None:

            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
            # print(list(detected_circles[0][0]))

            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Draw the circumference of the circle.
                cv2.circle(self.circ_mask, (a, b), r, (255, 255, 255), -1)
                # cv2.circle(self.raw_img, (a, b), r, (255, 255, 255), -1)
                self.circ_centers = [[a, b], r]

                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(self.raw_img, (a, b), r, (0, 255, 0), 3)

    def img_lines(self):
        signs_img = apply_mask(self.circ_mask, self.raw_img)
        self.lines = signs_img.copy()

        gray = cv2.cvtColor(signs_img, cv2.COLOR_BGR2GRAY)

        # Apply edge detection method on the image
        edges = cv2.Canny(gray, 190, 210, apertureSize=3)

        kernel = np.ones((3, 1), np.uint8)
        # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # self.lines = edges.copy()

        # This returns an array of r and theta values
        lines_list = []

        line = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=35, maxLineGap=15) # Max allowed gap between line for joining them)
        # line = None
        # Iterate over points
        if line is not None and len(line) >= 2:
            for i in range(2):
                # Extracted points nested in the list
                x1, y1, x2, y2 = line[i][0]
                # Draw the lines joing the points
                # On the original image
                if  x1-x2*0.05 <= x2 <= x1+x2*0.05 and x2-x1*0.05 <= x1 <= x2+x1*0.05:
                    cv2.line(self.lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Maintain a simples lookup list for points
                    lines_list.append([(x1, y1), (x2, y2)])
            if len(lines_list)>=2:
                pts = np.array([[lines_list[0][0]], [lines_list[0][1]], [lines_list[1][1]], [lines_list[1][0]]])
                cv2.fillPoly(self.lines, pts=[pts], color=(255, 0, 0))
                pts = np.array([[lines_list[0][0]], [lines_list[0][1]], [lines_list[1][0]], [lines_list[1][1]]])
                cv2.fillPoly(self.lines, pts=[pts], color=(255, 0, 0))

                if self.circ_centers[0][0] - self.circ_centers[1] * 0.15 <= (lines_list[0][0][0] + lines_list[1][0][0]) / 2 <= self.circ_centers[0][0] + self.circ_centers[1] * 0.15:
                    self.ans_lst.append(0)
                elif min(lines_list[0][0][0], lines_list[1][0][0]) <= self.circ_centers[0][0]-self.circ_centers[1]*0.15:
                    self.ans_lst.append(1)
                elif max(lines_list[1][0][0], lines_list[0][0][0]) >= self.circ_centers[0][0]+self.circ_centers[1]*0.15:
                    self.ans_lst.append(-1)

                if len(self.ans_lst) == 6:
                    if sum(self.ans_lst) < -5:
                        print("It's left!")
                        self.ans = "Left"
                    elif sum(self.ans_lst) > 5:
                        print("It's right!")
                        self.ans = "Right"
                    elif abs(sum(self.ans_lst)) < 1:
                        print("It's forward!")
                        self.ans = "Forward"
                    self.ans_lst = []


def show_img(img2show):
    # img2show = cv2.cvtColor(img2show, cv2.COLOR_BGR2RGB)
    # border_lines = [img2show.shape[1]*0.45, img2show.shape[1]*0.55]
    cv2.imshow("frame", img2show)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()


def apply_mask(mask, raw_img):
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    result = cv2.bitwise_and(raw_img, raw_img, mask=mask)
    return result


def setup():
    camera = Img()
    main(camera)


def main(camera):
    while True:

        camera.get_raw_img()
        camera.img_processing()

        camera.img_masking()
        camera.img_circles()
        camera.img_lines()

        # show_img(camera.img4masking)
        # show_img(apply_mask(camera.blue_mask, camera.raw_img))
        # show_img((apply_mask(camera.circ_mask, camera.raw_img)))
        cv2.putText(camera.raw_img, str("Next move: "+camera.ans), (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 2)
        show_img(cv2.cvtColor(camera.raw_img, cv2.COLOR_BGR2RGB))
        # show_img(cv2.cvtColor(camera.raw_img, cv2.COLOR_BGR2RGB))
        # show_img(camera.img4masking)
        # show_img(camera.img4masking)
        # camera.img_contours(camera.blue_mask)


setup()

import cv2
from cv2 import aruco
import numpy as np
import imutils
from tkinter import *
from PIL import Image, ImageTk
import time


class Img:

    cap = None
    raw_img = None
    img4masking = None
    blue_mask = None
    red_mask = None
    final_direction = [0, '']
    # total_mask = None
    contours = []
    centers = []

    def __init__(self):
        self.cap = cv2.VideoCapture(1)

    def get_raw_img(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.raw_img = frame

    def img_processing(self):
        def img_blur():
            blurred = cv2.medianBlur(self.raw_img, 11)
            blurred = cv2.blur(blurred, (11, 11))
            return blurred

        # def img_thresh():
        #     grayed = cv2.cvtColor(tmp_blured, cv2.COLOR_RGB2GRAY)
        #     threshed = cv2.adaptiveThreshold(grayed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
        #     return threshed

        self.img4masking = img_blur()

    def img_masking(self):

        self.centers = []
        self.contours = []

        def create_blue_mask():
            lower_limit = np.array([75, 75, 85])
            upper_limit = np.array([128, 255, 255])

            self.blue_mask = cv2.inRange(cv2.cvtColor(self.img4masking, cv2.COLOR_RGB2HSV), lower_limit, upper_limit)

        def create_red_mask():
            lower_limit1 = np.array([0, 50, 70])
            upper_limit1 = np.array([9, 255, 255])

            lower_limit2 = np.array([159, 50, 70])
            upper_limit2 = np.array([180, 255, 255])

            self.red_mask = cv2.inRange(cv2.cvtColor(self.raw_img, cv2.COLOR_RGB2HSV), lower_limit1, upper_limit1) + cv2.inRange(cv2.cvtColor(self.raw_img, cv2.COLOR_RGB2HSV), lower_limit2, upper_limit2)

        def center_contours(mask, color):
            grayed = cv2.cvtColor(apply_mask(mask, self.img4masking), cv2.COLOR_BGR2GRAY)
            contours = cv2.findContours(cv2.Canny(grayed, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            for cntr in contours:

                M = cv2.moments(cntr)
                area = cv2.contourArea(cntr)
                if M["m00"] != 0 and area > 300:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    self.centers.append([cX, cY, color, area, '', '', 'none', 0])
                    self.contours.append(cntr)

        create_red_mask()
        create_blue_mask()
        # self.total_mask = self.blue_mask + self.red_mask
        center_contours(self.red_mask, 'red')
        center_contours(self.blue_mask, 'blue')

    def center_analysis(self, mode):

        self.final_direction = [0, '']
        for i in self.centers:
            for j in self.centers:
                if i != j:
                    ij_distance = ((j[0] - i[0]) ** 2 + (j[1] - i[1]) ** 2) ** 0.5
                    if i[5] == '':
                        i[4] = ij_distance
                        i[5] = j
                    else:
                        if i[4] > ij_distance:
                            i[4] = ij_distance
                            i[5] = j

        for i in self.centers:
            for j in self.centers:
                if i != j:
                    if i[5] == j and j[5] == i and (i[4] != 'I have a pair!'):
                        # print('pair')
                        i[4] = 'I have a pair!'
                        j[4] = 'I have a pair!'

                        if j[3] > i[3] and j[2] == mode:
                            if i[0] < j[0]:
                                i[6] = 'Right'
                                j[6] = 'Right'
                                if self.final_direction[0] < j[3]:
                                    self.final_direction[1] = 'Right'
                                    self.final_direction[0] = j[3]
                            else:
                                i[6] = 'Left'
                                j[6] = 'Left'
                                if self.final_direction[0] < j[3]:
                                    self.final_direction[1] = 'Left'
                                    self.final_direction[0] = j[3]
                        elif j[3] < i[3] and i[2] == mode:
                            if i[0] > j[0]:
                                i[6] = 'Right'
                                j[6] = 'Right'
                                if self.final_direction[0] < i[3]:
                                    self.final_direction[1] = 'Right'
                                    self.final_direction[0] = i[3]
                            else:
                                i[6] = 'Left'
                                j[6] = 'Left'
                                if self.final_direction[0] < i[3]:
                                    self.final_direction[1] = 'Left'
                                    self.final_direction[0] = i[3]


def show_img(img2show, centers=[]):
    img2show = cv2.cvtColor(img2show, cv2.COLOR_BGR2RGB)
    border_lines = [img2show.shape[1]*0.475, img2show.shape[1]*0.525]
    if centers:
        for i in centers:
            if i[6] != 'none':

                robot_command = get_robot_command([i[0]+(i[5][0]-i[0])//2, i[1]+(i[5][1]-i[1])//2], border_lines)

                # cv2.circle(img2show, (i[0], i[1]), 3, (255, 0, 0), -1)
                cv2.circle(img2show, (i[0]+(i[5][0]-i[0])//2, i[1]+(i[5][1]-i[1])//2), 3, robot_command[1], -1)
                # cv2.circle(img2show, (i[0], i[1]), 200, (255, 255, 255), 0)
                cv2.putText(img2show, str(i[6])+' '+robot_command[0], (i[0]+(i[5][0]-i[0])//2 - 20, i[1]+(i[5][1]-i[1])//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, robot_command[1], 2)

    # print(img_size[1])
    cv2.line(img2show, (int(border_lines[0]), 0), (int(border_lines[0]), img2show.shape[1]), (0, 255, 0), thickness=1)
    cv2.line(img2show, (int(border_lines[1]), 0), (int(border_lines[1]), img2show.shape[1]), (0, 255, 0), thickness=1)
    cv2.imshow("frame", img2show)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()


def get_robot_command(center, borders):
    if borders[1] > center[0] > borders[0]:
        return '', (255, 255, 255)
    else:
        if center[0] <= borders[0]:
            return '<-', (0, 0, 255)
        elif center[0] >= borders[1]:
            return '->', (0, 0, 255)


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
        camera.center_analysis("blue")

        # show_img(camera.img4masking)
        # show_img(apply_mask(camera.thresh_mask, camera.img4masking))
        # show_img(apply_mask(camera.total_mask, camera.raw_img))
        show_img(camera.raw_img, camera.centers)
        if camera.final_direction[1] != '':
            print(camera.final_direction[1])
        # show_img(camera.img4masking)
        # camera.img_contours(camera.blue_mask)


setup()

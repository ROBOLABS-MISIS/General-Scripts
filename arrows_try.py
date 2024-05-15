import cv2
from cv2 import aruco
import numpy as np
import imutils
import socket
import threading
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import time

last_command = ''
focal = 2800
dist_lst = []
cur_command = None
next_turn = ''
stop_flag = False


class Img:

    cap = None
    raw_img = None
    img4masking = None
    blue_mask = None
    red_mask = None
    total_mask = None
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

    def img_blur(self, med_val, val):
        blurred = cv2.medianBlur(self.raw_img, med_val)
        blurred = cv2.blur(blurred, (val, val))
        self.img4masking = blurred

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
        center_contours(self.blue_mask, 'blue')
        center_contours(self.red_mask, 'red')


    def center_analysis(self, mode):

        self.final_direction = [0, '']
        for i in self.centers:
            for j in self.centers:
                if i != j and i[3] != i[j]:
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
                        print('pair')
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

    def find_marker(self):
        image = apply_mask((self.red_mask + self.blue_mask), self.raw_img)
        # convert the image to grayscale, blur it, and detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)
        # find the contours in the edged image and keep the largest one;
        # we'll assume that this is our piece of paper in the image
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) != 0:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            w = cv2.arcLength(c, True)
            # # compute the bounding box of thepaper region and return it
            W = 20  # cm
            # d = 60
            # f = (w*d)/W
            #
            # print(f"Focal length: {f}")

            return (W * focal) / w


def show_img(img2show, dist, centers=[]):

    global last_command, dist_lst, next_turn
    img2show = cv2.cvtColor(img2show, cv2.COLOR_BGR2RGB)
    border_lines = [img2show.shape[1]*0.45, img2show.shape[1]*0.55]
    if centers:
        for i in centers:
            if i[6] != 'none':

                t1 = threading.Thread(target=get_robot_command, args=([i[0] + (i[5][0] - i[0]) // 2, i[1] + (i[5][1] - i[1]) // 2], border_lines, dist))
                t1.start()

                if cur_command != None:
                    robot_command = cur_command
                else:
                    robot_command = ['none', (255, 255, 255)]

                    # cv2.circle(img2show, (i[0], i[1]), 3, (255, 0, 0), -1)
                cv2.circle(img2show, (i[0]+(i[5][0]-i[0])//2, i[1]+(i[5][1]-i[1])//2), 3, robot_command[1], -1)
                # cv2.circle(img2show, (i[0], i[1]), 200, (255, 255, 255), 0)
                cv2.putText(img2show, str(i[6])+' '+robot_command[0], (i[0]+(i[5][0]-i[0])//2 - 20, i[1]+(i[5][1]-i[1])//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, robot_command[1], 2)
                if str(i[6]) == 'Right' or str(i[6]) == 'Left':
                    next_turn = str(i[6])
    else:
        if last_command != "T-100":
            send_txt("T-100")
            last_command = "T-100"

    # print(img_size[1])
    cv2.line(img2show, (int(border_lines[0]), 0), (int(border_lines[0]), img2show.shape[1]), (0, 255, 0), thickness=1)
    cv2.line(img2show, (int(border_lines[1]), 0), (int(border_lines[1]), img2show.shape[1]), (0, 255, 0), thickness=1)
    cv2.imshow("frame", img2show)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()


def get_robot_command(center, borders, dist):
    global last_command, dist_lst, cur_command, stop_flag

    if borders[1] > center[0] > borders[0]:
        dist_lst.append(dist)
        if len(dist_lst) == 12:
            act_dist = sum(dist_lst) / 12
            print(act_dist)
            dist_lst = []
            if act_dist <= 36:
                print('WE are here!!')
                if next_turn == 'Left':
                    send_txt("ST7")
                    time.sleep(1)
                    send_txt("T-100")
                    last_command = "T-100"
                    stop_flag = True
                    time.sleep(4)
                    stop_flag = False
                    send_txt("ST3")
                elif next_turn == 'Right':
                    send_txt("ST7")
                    time.sleep(1)
                    send_txt("T100")
                    last_command = "T100"
                    stop_flag = True
                    time.sleep(4)
                    stop_flag = False
                    send_txt("ST3")


        else:
            if last_command != "0" and last_command != '1':
                send_txt("1")
                last_command = "1"
            cur_command = ('', (255, 255, 255))
    else:
        if center[0] <= borders[0]:
            if last_command != "T-100":
                send_txt("T-100")
                last_command = "T-100"
            cur_command = ('<-', (0, 0, 255))
        elif center[0] >= borders[1]:
            if last_command != "T100":
                send_txt("T100")
                last_command = "T100"
            cur_command = ('->', (0, 0, 255))


def send_txt(txt, destination=''):
    if not stop_flag:
        clientsocket.send(bytes(destination+txt, "utf-8"))
        print(destination+txt)


def apply_mask(mask, raw_img):
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    result = cv2.bitwise_and(raw_img, raw_img, mask=mask)
    return result


def setup():
    camera = Img()
    main(camera)


def main(camera):
    global clientsocket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 5001))
    s.listen(5)
    print('server is now running')

    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established")

    while True:

        camera.get_raw_img()
        camera.img_blur(1, 1)

        camera.img_masking()
        camera.center_analysis("red")



        # show_img(camera.img4masking)
        # show_img(apply_mask(camera.thresh_mask, camera.img4masking))
        # show_img(apply_mask(camera.total_mask, camera.raw_img))
        show_img(camera.raw_img, camera.find_marker(), camera.centers)
        # if camera.final_direction[1] != '':
              # print(camera.final_direction[1])
        # show_img(camera.img4masking)
        # camera.img_contours(camera.blue_mask)


setup()

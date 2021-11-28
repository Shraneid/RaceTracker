import time
import math
import cv2
import mss
import numpy as np

initialised = False

# replace with camera connection
video_source = {"top": 425, "left": 2900, "width": 350, "height": 250}

control_window_name = 'controls5'


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(self, other.__class__) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(int(self.x*1000 + self.y))


class Line:
    middle_point_spread = 70

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.firstPoint = Point(x1, y1)
        self.secondPoint = Point(x2, y2)

        center_point_x = abs(int(x1 + x2 / 2))
        center_point_y = abs(int(y1 + y2 / 2))

        self.centerPoint = Point(center_point_x, center_point_y)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and get_distance(self.centerPoint, other.centerPoint) < self.middle_point_spread

    def __hash__(self):
        return hash(self.centerPoint.__hash__())

    def is_left_of(self, other):
        return self.centerPoint.x < other.centerPoint.x


class Tracker:
    def __init__(self):
        self.left_line = None
        self.right_line = None
        self.tracked_lines = []
        self.difference_threshold = 50

    def update_lines(self, new_lines, init=False):
        if init:
            self.tracked_lines = new_lines

        for tracked_line in self.tracked_lines:
            for new_line in new_lines:
                if difference(new_line, tracked_line) < self.difference_threshold:
                    tracked_line = new_line
                    break

        if self.right_line.is_left_of(self.left_line):
            self.right_line, self.left_line = self.left_line, self.right_line


def slope(line):
    if line.firstPoint - line.secondPoint != 0:
        return -(line.secondPoint - line.firstPoint) / (line.secondPoint - line.firstPoint)
    return 0


def difference(line, pt1, pt2):
    old_slope = slope(line[0], line[1])
    new_slope = slope(pt1, pt2)
    x_diff = abs(abs(line[0][0]) - abs(pt1[0]))
    slope_diff = abs(abs(old_slope) - abs(new_slope))
    return ((x_diff * 5) + (slope_diff * 20))


def get_distance(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y - p2.y)**2)


def get_dedup_lines(lines):
    dedup_lines = []

    # need to remove like this so __eq__ is called since set() will call the __hash__ method
    for line in lines:
        if line not in dedup_lines:
            dedup_lines.append(line)

    return dedup_lines


def nop(value):
    pass


def init_cv2_window():
    cv2.namedWindow(control_window_name)
    cv2.createTrackbar('low1', control_window_name, 0, 255, nop)
    cv2.createTrackbar('low2', control_window_name, 0, 255, nop)
    cv2.createTrackbar('low3', control_window_name, 125, 255, nop)
    cv2.createTrackbar('high1', control_window_name, 12, 255, nop)
    cv2.createTrackbar('high2', control_window_name, 80, 255, nop)
    cv2.createTrackbar('high3', control_window_name, 229, 255, nop)


def process_image(image, mask):
    # to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # mask out unwanted pixel colors
    threshold_img = cv2.inRange(hsv, mask[0], mask[1])

    # percent of original size
    scale_percent = 30
    width = int(threshold_img.shape[1] * scale_percent / 100)
    height = int(threshold_img.shape[0] * scale_percent / 100)
    resized = cv2.resize(threshold_img, (width, height), interpolation=cv2.INTER_NEAREST)

    # convert to gray
    # gray = cv2.cvtColor(threshold_img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(threshold_img, 50, 150, apertureSize=3)

    return edges


def get_white_mask():
    # get mask values
    low1 = cv2.getTrackbarPos('low1', control_window_name)
    low2 = cv2.getTrackbarPos('low2', control_window_name)
    low3 = cv2.getTrackbarPos('low3', control_window_name)

    high1 = cv2.getTrackbarPos('high1', control_window_name)
    high2 = cv2.getTrackbarPos('high2', control_window_name)
    high3 = cv2.getTrackbarPos('high3', control_window_name)

    return np.array([low1, low2, low3]), np.array([high1, high2, high3])


def get_line_from_parameters(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)

    return Line(x1, y1, x2, y2)


def get_lines(img):
    linesOutput = cv2.HoughLines(img, 1, np.pi / 720, 80)

    lines = []
    if linesOutput is not None:
        for line in linesOutput:
            rho, theta = line[0]
            line_object = get_line_from_parameters(rho, theta)

            lines.append(line_object)

    else:
        lines = []

    # return [lines[0]]
    return lines


with mss.mss() as sct:
    init_cv2_window()

    # l1 = Line(0, 0, 10, 10)
    # l2 = Line(0, 1, 10, 10)
    # l3 = Line(0, 1, 1000, 1000)
    #
    # print(l1.centerPoint.__hash__())
    # print(l1.__hash__())
    # print(l1.__eq__(l2))
    #
    # lst = [l1, l2, l3]
    # s = set(lst)
    # l = []
    # for i in lst:
    #     if i not in l:
    #         l.append(i)
    #
    # brk = 0

    running_lines_tracker = Tracker()

    while "Screen capturing":
        last_time = time.time()

        white_mask = get_white_mask()

        # get image
        img = np.array(sct.grab(video_source))

        # process
        processed_image = process_image(img, white_mask)

        all_lines = get_lines(processed_image)
        lines = get_dedup_lines(all_lines)

        print(len(lines))
        # print(lines)

        if lines is not None:
            for line in lines:
                cv2.line(
                    img,
                    (line.firstPoint.x, line.firstPoint.y),
                    (line.secondPoint.x, line.secondPoint.y),
                    (0, 255, 0), 2
                )



        cv2.imshow("RACE_TRACKER", img)
        cv2.imshow("DEBUG", processed_image)

        if cv2.waitKey(1) & 0xFF == ord("s"):
            initialised = True
            print("Initialisation Done")

import time
import math
from random import random
from typing import List

import cv2
import mss
import numpy
import numpy as np


# Class that defines the points, the hash is multiplying the x value by 1000 because you can never get higher values
# on my screen but feel free to change it
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(self, other.__class__) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(int(self.x * 1000 + self.y))

    def get_distance_to(self, other_point) -> int:
        return int(math.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2))


# This class defines all the lines that we work with, the __eq__ method is called when we deduplicate similar lines in
# the processing. The middle point spread is the max distance between two different lines middle points that is used
# to consider two lines as the same. (would be better to use in window line middle but can't figure it out now)
class Line:
    middle_point_spread = 70  # arbitrary value

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
        return isinstance(other, self.__class__) and \
               self.centerPoint.get_distance_to(other.centerPoint) < self.middle_point_spread

    def __hash__(self):
        return hash(self.centerPoint.__hash__())

    def is_left_of(self, other) -> bool:
        return self.centerPoint.x < other.centerPoint.x

    def slope(self) -> int:
        if self.firstPoint.y - self.secondPoint.y != 0:  # div by 0
            return abs(int((self.firstPoint.x - self.secondPoint.x) / (self.firstPoint.y - self.secondPoint.y)))
        return 1000  # vertical line

    def get_difference_with(self, line2) -> int:
        slope_diff = abs(self.slope() - line2.slope())
        x_diff = abs(self.centerPoint.x - line2.centerPoint.x)
        return int(x_diff + slope_diff * 4)

    @staticmethod
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


# Main class of our tracker, this is keeping track of the lines we have interest in and tracks them over time
# There is an initialisation period during which the lines are not checked with the previous ones.
class Tracker:
    initialized: bool
    left_line: Line
    right_line: Line
    previous_all_lines: List[Line]
    tracked_lines: list[Line]
    difference_threshold: int

    def __init__(self):
        self.initialized = False

        self.image = None
        self.hsv_image = None
        self.threshold_image = None
        self.edges_image = None
        self.pre_processed_image = None

        self.left_line = None
        self.right_line = None

        self.all_lines = []
        self.previous_all_lines = []
        self.tracked_lines = []

        self.deduplicated_lines = []

        self.difference_threshold = 50

    def update(self, new_lines: List[Line]) -> None:
        if self.previous_all_lines is not None \
                and len(self.previous_all_lines) == 2 \
                and self.previous_all_lines[0].get_difference_with(self.previous_all_lines[1]) > 200 \
                and not self.initialized and new_lines != self.previous_all_lines:
            # We revert them at the end anyways so we can just assign randomly
            self.left_line = self.previous_all_lines[0]
            self.right_line = self.previous_all_lines[1]

            self.initialized = True
            print("Initialization Done, starting")
        else:
            self.previous_all_lines = new_lines

        if self.initialized:
            # here we shuffle so that we get a random line from a cluster of lines each time, since Hough lines work
            # with a circular algorithm, it would always give the lines on the inside or outside, that way we can
            # sort of keep an average
            new_lines = sorted(new_lines, key=lambda x: random())
            self.tracked_lines = new_lines

            # if no new line is found, that means we might have not found a matching line on the new image so we don't
            # update
            # TODO: update with how much the other one is moved (failsafe mechanism when none are found)
            for tracked_line in self.tracked_lines:
                for new_line in new_lines:
                    if new_line.get_difference_with(tracked_line) < self.difference_threshold:
                        # update old tracked line, this is why we need the shuffle
                        self.tracked_lines[self.tracked_lines.index(tracked_line)] = new_line
                        break

            # keep track of which is where for the tracker to know if we are too far right or left
            if self.right_line.is_left_of(self.left_line):
                self.right_line, self.left_line = self.left_line, self.right_line

        self.previous_all_lines = new_lines

    def set_image(self, image):
        self.image = image

    def add_lines_to_image(self):
        if self.deduplicated_lines is not None:
            for line in self.deduplicated_lines:
                cv2.line(
                    self.image,
                    (line.firstPoint.x, line.firstPoint.y),
                    (line.secondPoint.x, line.secondPoint.y),
                    (0, 255, 0), 2
                )

    def pre_process_image(self, mask):
        # to HSV
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # mask out unwanted pixel colors
        self.threshold_image = cv2.inRange(self.hsv_image, mask[0], mask[1])

        # percent of original size
        # scale_percent = 30
        # width = int(threshold_img.shape[1] * scale_percent / 100)
        # height = int(threshold_img.shape[0] * scale_percent / 100)
        # resized = cv2.resize(threshold_img, (width, height), interpolation=cv2.INTER_NEAREST)

        # convert to gray
        # gray = cv2.cvtColor(threshold_img, cv2.COLOR_BGR2GRAY)

        # edge detection
        self.edges_image = cv2.Canny(self.threshold_image, 50, 150, apertureSize=3)

        self.pre_processed_image = self.edges_image

    def update_all_lines(self):
        lines_output = cv2.HoughLines(self.edges_image, 1, np.pi / 720, 80)

        self.all_lines = []
        if lines_output is not None:
            for line in lines_output:
                rho, theta = line[0]
                line_object = Line.get_line_from_parameters(rho, theta)

                self.all_lines.append(line_object)
        # self.all_lines = [lines[0]]

    def update_deduplicated_lines(self) -> None:
        self.deduplicated_lines = []

        # need to remove like this so __eq__ is called since set() will call the __hash__ method
        for line in self.all_lines:
            if line not in self.deduplicated_lines:
                self.deduplicated_lines.append(line)

    def get_image(self):
        return self.image


# noinspection PyUnusedLocal
def nop(value: object) -> None:
    pass


def init_cv2_window(control_window_name) -> None:
    cv2.namedWindow(control_window_name)
    cv2.createTrackbar('low1', control_window_name, 0, 255, nop)
    cv2.createTrackbar('low2', control_window_name, 0, 255, nop)
    cv2.createTrackbar('low3', control_window_name, 125, 255, nop)
    cv2.createTrackbar('high1', control_window_name, 12, 255, nop)
    cv2.createTrackbar('high2', control_window_name, 80, 255, nop)
    cv2.createTrackbar('high3', control_window_name, 229, 255, nop)


def get_white_mask(control_window_name) -> numpy.array:
    # get mask values
    low1 = cv2.getTrackbarPos('low1', control_window_name)
    low2 = cv2.getTrackbarPos('low2', control_window_name)
    low3 = cv2.getTrackbarPos('low3', control_window_name)

    high1 = cv2.getTrackbarPos('high1', control_window_name)
    high2 = cv2.getTrackbarPos('high2', control_window_name)
    high3 = cv2.getTrackbarPos('high3', control_window_name)

    return np.array([low1, low2, low3]), np.array([high1, high2, high3])


def main():
    # replace with camera connection
    # Currently you need to have a video running on your desktop and this video source will take screen shots of it
    # so you can work with recorded data
    video_source = {"top": 425, "left": 2900, "width": 350, "height": 250}

    # Name of the window that adjusts the white mask for running lines, for some reason cv2 seems to cache the window
    # mode and then gets stuck once in full screen or hidden, so if the controls window doesn't show or bugs out,
    # you can increment the counter :D (tried to debug it for 1 hr, no idea why it won't even listen when you force
    # the window mode)
    control_window_name = 'controls5'

    with mss.mss() as sct:
        init_cv2_window(control_window_name)

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
            # last_time = time.time()

            white_mask = get_white_mask(control_window_name)

            # get image, replace by any source
            running_lines_tracker.set_image(np.array(sct.grab(video_source)))

            # pre process image for hough line
            running_lines_tracker.pre_process_image(white_mask)

            running_lines_tracker.update_all_lines()
            running_lines_tracker.update_deduplicated_lines()

            running_lines_tracker.update()

            # print(len(lines))

            cv2.imshow("RACE_TRACKER", running_lines_tracker.get_image())
            cv2.imshow("DEBUG", running_lines_tracker.edges_image)


if __name__ == "__main__":
    main()

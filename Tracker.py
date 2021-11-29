import cv2
import numpy as np

from random import random
from typing import List

from HelperObjects import Line


# Main class of our tracker, this is keeping track of the lines we have interest in and tracks them over time
# There is an initialisation period during which the lines are not checked with the previous ones.
class Tracker:
    initialized: bool
    left_line: Line
    right_line: Line
    previous_all_lines: List[Line]
    tracked_lines: List[Line]
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

    def update(self) -> None:
        if self.previous_all_lines is not None \
                and len(self.previous_all_lines) == 2 \
                and self.previous_all_lines[0].get_difference_with(self.previous_all_lines[1]) > 200 \
                and not self.initialized and self.deduplicated_lines != self.previous_all_lines:
            # We revert them at the end anyways so we can just assign randomly
            self.left_line = self.previous_all_lines[0]
            self.right_line = self.previous_all_lines[1]

            self.initialized = True
            print("Initialization Done, starting")

        if self.initialized:
            # here we shuffle so that we get a random line from a cluster of lines each time, since Hough lines work
            # with a circular algorithm, it would always give the lines on the inside or outside, that way we can
            # sort of keep an average
            self.deduplicated_lines = sorted(self.deduplicated_lines, key=lambda x: random())
            self.tracked_lines = self.deduplicated_lines

            # if no new line is found, that means we might have not found a matching line on the new image so we don't
            # update
            # TODO: update with how much the other one is moved (failsafe mechanism when none are found)
            for tracked_line in self.tracked_lines:
                for new_line in self.deduplicated_lines:
                    if new_line.get_difference_with(tracked_line) < self.difference_threshold:
                        # update old tracked line, this is why we need the shuffle
                        self.tracked_lines[self.tracked_lines.index(tracked_line)] = new_line
                        break

            # keep track of which is where for the tracker to know if we are too far right or left
            if self.right_line.is_left_of(self.left_line):
                self.right_line, self.left_line = self.left_line, self.right_line

        self.previous_all_lines = self.deduplicated_lines

    # TODO: create an image handler object instead of doing it in the tracker
    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image

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

    def add_lines_to_image(self):
        if self.deduplicated_lines is not None:
            for line in self.deduplicated_lines:
                cv2.line(
                    self.image,
                    (line.firstPoint.x, line.firstPoint.y),
                    (line.secondPoint.x, line.secondPoint.y),
                    (0, 255, 0), 2
                )

        if self.tracked_lines is not None:
            for line in self.tracked_lines:
                cv2.line(
                    self.image,
                    (line.firstPoint.x, line.firstPoint.y),
                    (line.secondPoint.x, line.secondPoint.y),
                    (255, 0, 0), 2
                )

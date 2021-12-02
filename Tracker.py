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
    x_difference_threshold: int

    def __init__(self, x_difference_threshold, threshold):
        self.x_difference_threshold = x_difference_threshold
        self.right_threshold = 350/2 + threshold
        self.left_threshold = 350/2 - threshold

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
        self.possible_lines = []

    def update_all_lines(self):
        lines_output = cv2.HoughLines(self.pre_processed_image, 1, np.pi / (720 * 2), 180)

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
                and self.previous_all_lines[0].get_difference_with(self.previous_all_lines[1]) > \
                self.x_difference_threshold \
                and not self.initialized and self.deduplicated_lines != self.previous_all_lines:
            # We revert them at the end anyways so we can just assign randomly
            self.left_line = self.previous_all_lines[0]
            self.right_line = self.previous_all_lines[1]

            self.tracked_lines = self.previous_all_lines

            self.initialized = True
            print("Initialization Done, starting")

        if self.initialized:
            self.possible_lines = self.deduplicated_lines.copy()

            # if no new line is found, that means we might have not found a matching line on the new image so we don't
            # update
            for tracked_line in self.tracked_lines:
                new_tracked_line: Line = None
                for new_line in self.possible_lines:
                    current_diff = self.x_difference_threshold

                    # print(new_line.get_difference_with(tracked_line))
                    if new_line.get_difference_with(tracked_line) < self.x_difference_threshold \
                            and new_line.get_difference_with(tracked_line) < current_diff:
                        # update old tracked line, this is why we need the shuffle
                        new_tracked_line = new_line
                        self.possible_lines.remove(new_line)
                if new_tracked_line is not None and \
                        new_tracked_line.get_difference_with(self.tracked_lines[self.tracked_lines.index(tracked_line)]) < 10000:
                    self.tracked_lines[self.tracked_lines.index(tracked_line)] = new_tracked_line

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
        scale_percent = 30
        width = int(self.threshold_image.shape[1] * scale_percent / 100)
        height = int(self.threshold_image.shape[0] * scale_percent / 100)
        self.resized = cv2.resize(self.threshold_image, (width, height), interpolation=cv2.INTER_NEAREST)

        # convert to gray
        # gray = cv2.cvtColor(threshold_img, cv2.COLOR_BGR2GRAY)

        # edge detection
        # self.edges_image = cv2.Canny(self.threshold_image, 50, 150, apertureSize=3)

        # self.pre_processed_image = self.edges_image
        self.pre_processed_image = self.threshold_image

    def add_lines_to_image(self):
        if self.deduplicated_lines is not None:
            for line in self.deduplicated_lines:
                cv2.line(
                    self.image,
                    (line.pointOne.x, line.pointOne.y),
                    (line.pointTwo.x, line.pointTwo.y),
                    (0, 255, 0), 2
                )

        if self.tracked_lines is not None:
            for line in self.tracked_lines:
                cv2.line(
                    self.image,
                    (line.pointOne.x, line.pointOne.y),
                    (line.pointTwo.x, line.pointTwo.y),
                    (255, 0, 0), 2
                )

    def process_is_running_straight(self) -> None:
        if len(self.tracked_lines) != 2:
            return

        low1, _ = self.tracked_lines[0].get_low_and_high_points()
        low2, _ = self.tracked_lines[1].get_low_and_high_points()

        low_avg_x = (low1.x + low2.x) / 2

        if self.left_threshold < low_avg_x < self.right_threshold:
            print("straight")
        elif low_avg_x < self.left_threshold:
            print("too much left, go right")
        elif self.right_threshold < low_avg_x:
            print("too much right, go left")

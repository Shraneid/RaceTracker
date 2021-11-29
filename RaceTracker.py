import time
import math
from random import random
from typing import List

import cv2
import mss
import numpy
import numpy as np


# noinspection PyUnusedLocal
from Tracker import Tracker


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
    if control_window_name is None:
        return np.array([0, 0, 125]), np.array([12, 80, 229])

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
    show_sliders = False
    control_window_name = 'controls6'
    if not show_sliders:
        control_window_name = None

    with mss.mss() as sct:
        if show_sliders:
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

            running_lines_tracker.add_lines_to_image()

            # print(len(lines))
            cv2.imshow("RACE_TRACKER", running_lines_tracker.get_image())
            cv2.imshow("DEBUG", running_lines_tracker.edges_image)

            if cv2.waitKey(1) & 0xFF == ord("h"):
                pass


if __name__ == "__main__":
    main()

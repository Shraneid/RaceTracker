import math
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
        self.pointOne = Point(x1, y1)
        self.pointTwo = Point(x2, y2)

        center_point_x = abs(int(x1 + x2 / 2))
        center_point_y = abs(int(y1 + y2 / 2))

        self.middlePoint = Point(center_point_x, center_point_y)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               self.middlePoint.get_distance_to(other.middlePoint) < self.middle_point_spread

    def __hash__(self):
        return hash(self.middlePoint.__hash__())

    def is_left_of(self, other) -> bool:
        return self.middlePoint.x < other.middlePoint.x

    def abs_slope(self) -> int:
        slope = 1000  # vertical line
        if self.pointOne.y - self.pointTwo.y != 0:  # div by 0
            slope = abs((self.pointOne.x - self.pointTwo.x) / (self.pointOne.y - self.pointTwo.y))

        return slope

    def get_difference_with(self, line2) -> int:
        slope_diff = abs(self.abs_slope() - line2.abs_slope())
        low_point_line_1 = self.pointOne if self.y1 > self.y2 else self.pointTwo
        low_point_line_2 = line2.pointOne if line2.y1 > line2.y2 else line2.pointTwo
        x_diff = (abs(low_point_line_1.x - low_point_line_2.x)/10 +
                  abs(self.middlePoint.get_distance_to(line2.middlePoint))/10)**2

        # print(int(x_diff * 10 + slope_diff * 4))
        return int(x_diff * 10 + slope_diff * 4)

    def get_low_and_high_points(self):
        if self.pointOne.y > self.pointTwo.y:
            return self.pointTwo, self.pointOne
        else:
            return self.pointOne, self.pointTwo

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

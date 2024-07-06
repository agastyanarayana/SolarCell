import cv2
import numpy as np
from math import atan2, degrees, atan


def AngleBtw2Points(l):
    '''
    calculates the angle of a line
    '''

    pointA = l[0][:2]
    pointB = l[0][2:]
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    return degrees(atan2(changeInY, changeInX))  # remove degrees if you want your answer in radians


def getLines(c_img, threshold=10, min_line_length=20, max_line_gap=20):
    '''
    Hough lines can detect lines from a contoured or edged image.
    We can further classify these lines in horizontal and vertical lines
    based on the angle. 
    '''

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = threshold  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = min_line_length  # minimum number of pixels making up a line
    max_line_gap = max_line_gap  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(c_img[:, :, 0], rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    horizontal_lines = []
    vertical_lines = []

    for l in lines:
        angle = abs(AngleBtw2Points(l))

        if angle < 5:
            horizontal_lines.append(l)

        elif angle > 85:
            vertical_lines.append(l)

    return vertical_lines, horizontal_lines


def getMargins(img, lines, grid=(12, 6), mode='v', offset=10):
    h, w, c = img.shape
    img_name = img_name = getattr(img, 'name', '')  # Assuming 'img' has a 'name' attribute for image name
    
    if mode == 'v':
        gap = w // grid[1]
        valid_lines = []

        if img_name.startswith("R"):
            # Adding the first line at the left border of the image with offset
            valid_lines.append(np.array([[offset, 0, offset, h]]))
        else:
            # Adding the first line at the left border of the image without offset
            valid_lines.append(np.array([[0, 0, 0, h]]))

        # Adding subsequent lines uniformly
        for i in range(1, grid[1] + 1):
            x = i * gap + (offset if img_name.startswith("L") else 0)
            valid_lines.append(np.array([[x, 0, x, h]]))

        return valid_lines

    elif mode == 'h':
        gap = h // grid[0]
        valid_lines = []

        # Adding the first line at the top border of the image with offset
        valid_lines.append(np.array([[0, offset, w, offset]]))

        # Adding subsequent lines uniformly
        for i in range(1, grid[0] + 1):
            y = i * gap + offset
            valid_lines.append(np.array([[0, y, w, y]]))

        return valid_lines

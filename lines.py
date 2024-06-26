import cv2
import numpy as np
from math import atan2,degrees, atan


def AngleBtw2Points(l):
    '''
    calculates the angle of a line
    '''

    pointA = l[0][:2]
    pointB = l[0][2:]
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    return degrees(atan2(changeInY,changeInX)) #remove degrees if you want your answer in radians




def getLines(c_img, threshold = 10, min_line_length = 20, max_line_gap = 20):

    '''
    Hough lines can detect lines from a contoured or edged image.
    We can further classify these lines in horizontal and vertical lines
    based on the angle. 
    '''

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = threshold  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = min_line_length# minimum number of pixels making up a line
    max_line_gap = max_line_gap  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(c_img[:,:,0], rho, theta, threshold, np.array([]),
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






def getMargins(img, lines, grid = (12,6), mode = 'v'):

    '''
    Hough lines generates a lot of lines. We need a way to eliminate noisy lines. This can be done with the
    help of grid priors. We know the grid size. Hence, we can draw vertical and horizonal lines approximately
    at their respective positions (with some error margines) . We can call all the lines near by valid and let go of all the lines 
    that are far from any given vertical or horizontal line.  

    '''

        
    h, w, c = img.shape
    if mode == 'v':
        
        gap = w//grid[1]
        bottom_stems = [0+gap*i for i in range(grid[1]+1)]
        base_lines_0 = [np.array([[0+gap*i, h//2 , 0+gap*i, h//2+250]]) for i in range(grid[1]+1)]

        valid_lines = []
        valid_lines.extend(base_lines_0)

        for l in np.squeeze(lines).tolist():
            x = l[0]
            for k in bottom_stems:
                if abs(k-x) <= 0.1*gap:
                    valid_lines.append(np.array([l]))

        return valid_lines
    
    elif mode == 'h':

        gap = h//grid[0]
        bottom_stems = [0+gap*i for i in range(grid[0]+1)]
        base_lines_0 = [np.array([[w//2+20, gap*i , h//2+250 , gap*i ]]) for i in range(grid[0]+1)]
        
        valid_lines = []
        valid_lines.extend(base_lines_0)
        
        
        for l in np.squeeze(lines).tolist():
            x = l[1]
            for k in bottom_stems:
                if abs(k-x) <= 0.25*gap:
                    valid_lines.append(np.array([l]))

        return valid_lines
        
        
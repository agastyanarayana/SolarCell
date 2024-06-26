import os
import cv2
import numpy as np 
from utils import findContours, getMinFitRect, four_point_transform




def filterContours(img, contours, hierarchy, stage=1):

    c_img = np.zeros_like(img)
    h,w,c = c_img.shape
    hie = np.squeeze(hierarchy).tolist()
    cv2.drawContours(c_img, contours, -1, (255,255,255), -1)
    return c_img



def perspectiveTransform(gray, img, area_thresh = 40, k_size = (5,5), iterations = 3):

    '''
    Funtion for creating prespective transformation.
    We would want to detect the panel and crop it out from the image. 
    '''

    # noise reduction
    kernel = np.ones(k_size,np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # contouring to find the biggest contour
    # it should be the panel it self

    # add area based filtering to make it more robust

    contours, hierarchy = findContours(gray)
    c_img = filterContours(img, contours, hierarchy)
    c_img, box, area = getMinFitRect(c_img, contours)

    h, w, c = img.shape


    # if we have a situation where the final crop is less than 40% in the size of the original image
    # it would indicate some kind of failure in the transormation.
    area_ratio = (100*area/(h*w))

    if area_ratio > area_thresh:

        transformed_img =  four_point_transform(img.copy(), box)
        transformed_img = cv2.resize(transformed_img, (img.shape[1],img.shape[0]))

        return True, transformed_img

    else:
        print(f'Failure in perspective transformation')
        return False, img

import os
import numpy as np
import cv2
import sys

from transformation import prespectiveTransform
from utils import *
import random

from sklearn.cluster import DBSCAN

from crop import cropper
from lines import getLines, getMargins


import json

with open('config.json','r') as f:
    config = json.load(f) 




def resize(img, f=0.5):
    '''
    f=0.5 stands for 50% reduction in image size
    '''

    return cv2.resize(img, (0,0), fx=f, fy=f)



        
def clusterNRegression(lines,img, grid=(12,6), mode='v'):

    '''
    So now we have lines. We have vertical and horizontal lines.

    A lines may have x1, y1 and x2, y2 as the end point. Several such small lines may make 
    the first horizontal line. Hence, in this specific case, we will cluster the y cordinates of all lines.
    At the end, we should get the number of cluster centers = grid [row].

    Similarly, number of cluster centers for vertical lines = grid [ column] 


    '''
    
    if mode == 'v':
        
        '''
        we are processing vertical lines. Typicaly, verical lines can be expressed as x = constant. 

        step1: create points list
        step2: create list of all y cordinates

        notice that len(points) will be equal to len(y cordinates)

        step3: we are now going to cluster all the ys. (DBSCAN)

        Once clustering is done, we will identify all the points belonging to a specific cluster and 
        create a line using regression. Since all the images are fairly square, we can skip the regression 
        and directly use the cluster center and extream end of the image to get the two points of the line.

        '''

        ys = []
        pts = []
        pred_lines = []
        for l in lines:
            pts.append(l[0][:2].tolist())
            pts.append(l[0][2:].tolist())

            ys.append((l[0][0],0))
            ys.append((l[0][2],0))


        d = 0.2*img.shape[0]//grid[1]
        
        print(f'd: {d}')

        clustering = DBSCAN(eps=d, min_samples=2).fit(ys)
        lbs = clustering.labels_.tolist()

        print(f'lbl max: {max(lbs)} lbl min: {min(lbs)}')

        collections = {}

        for l in lbs:
            collections[l] = []

        for p in range(len(pts)):

            cluster_label = lbs[p]
            collections[cluster_label].append(pts[p])

        for k, v in collections.items():
            
            r,g,b = random.randint(0,255),random.randint(0,255),random.randint(0,255)

            x = np.array(v)[:,:1].reshape(-1)
            y = np.array(v)[:,1:].reshape(-1)

            # we can do a np.polyfit is required

            x1,y1 =    int(x.mean()), 0
            x2,y2 =    int(x.mean()), img.shape[0]
            
            pred_lines.append(np.array([[x1,y1,x2,y2]]))

            cv2.line(img,(x1,y1),(x2,y2), (r,g,b), 15)
    
    elif mode == 'h':

        
        xs = []
        pts = []
        pred_lines = []
        for l in lines:
            pts.append(l[0][:2].tolist())
            pts.append(l[0][2:].tolist())

            xs.append((0,l[0][1]))
            xs.append((0,l[0][3]))


        d = 0.5*img.shape[1]//grid[0]

        clustering = DBSCAN(eps=d, min_samples=2).fit(xs)
        lbs = clustering.labels_.tolist()

        print(f'lbl max: {max(lbs)} lbl min: {min(lbs)}')

        collections = {}

        for l in lbs:
            collections[l] = []
        
#         print(collections.keys())

        for p in range(len(pts)):

            cluster_label = lbs[p]
            collections[cluster_label].append(pts[p])

        for k, v in collections.items():
            
            r,g,b = random.randint(0,255),random.randint(0,255),random.randint(0,255)

            x = np.array(v)[:,:1].reshape(-1)
            y = np.array(v)[:,1:].reshape(-1)
            
            m, b = np.polyfit(x, y, 1)

            x1,y1 =    0,int(m*x.mean() + int(b))
            x2,y2 =    img.shape[1], int((img.shape[1]*m) + b)
            
            pred_lines.append(np.array([[x1,y1,x2,y2]]))

            cv2.line(img,(x1,y1),(x2,y2), (r,g,b),15)
    
    return img, pred_lines
    



def drawContours(img, cnts,cont_size_percentage =0.001, area_thresh = None):

    '''
    Draws contours that are greated than a specific area. Basically a noise reduction step.

    provide one of the following:

        cont_size_percentage : least percentage area of cnt w.r.t image to be considered useful
        area_thresh: least area of cnt in pixel 

    '''
    
    c_img = np.zeros_like(img)
    if area_thresh == None:
        area_thresh = img.shape[0]*img.shape[1]*cont_size_percentage#how do we set the resolution
    
    for i,c in enumerate(cnts):

        a = cv2.contourArea(c)
        if a > area_thresh:
            cv2.drawContours(c_img, [c], 0, (255,255,255), 1)
    return c_img


def removeShadow(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    return cv2.merge(result_norm_planes)

def preprocess_ld(im_s, grid, 
                  image_name = 'abc.jpg', 
                  path = './results', 
                  write = False, 
                  debug = False, 
                  bypass_prespective_transform = False,
                  remove_shadow = False):

    '''
    parameters: 

        im_s: original image
        image_name: name of the image
        path: where you want to store results (required for debugging and result writing)
        write: (bool) if you want to write the image results to the path
        debug: (bool) will write images indicative of intermediate processing
        bypass_prespective_transform: (bool) should be false for images where prespective transformation fail.
        remove_shadow: (bool) helpful in some images but largely for experimental purpose
    
    '''
    # res_path = os.path.join(path, image_name)
    debug_path = os.path.join(path,'debug')
    crp_path = os.path.join(path, 'crops', image_name)

    # os.makedirs(res_path, exist_ok=True)
    os.makedirs(crp_path , exist_ok=True)
    os.makedirs(debug_path , exist_ok=True)

    if remove_shadow: 
        print('Equalizing image to remove shadow')
        im_o = removeShadow(im_s)
    else:
        im_o = im_s.copy()

    print(f'Preprocessing and prespective transforming the image')
    im = cv2.bitwise_not(im_o.copy())
    gray = preprocess(im)

    if not bypass_prespective_transform:
        sts, transformed_img = prespectiveTransform(gray, im, area_thresh=config['presp_fail_safe_thresh'])


        if not sts:
            print(f'Prespective transformed failed!')
            sys.exit()

        img = resize(transformed_img)

    else:
        print('Skipping prespective transformation. Assuming that the image is an accurate crop of the panel.')
        img = im.copy()


    print(f'Processing the image to find elements')
    blured = makeBlur(img.copy())
    preprocessed_img = cleanImage(blured)
    # preprocessed_img = cv2.bitwise_not(preprocessed_img.copy())
    contours = findContours(preprocessed_img,l=config['cont_img_low'], h=config['cont_img_high'])[0]
    # print(f'Detected: {len(contours)} contours')


    print(f'Extracting primary contours')
    c_img = drawContours(img, contours,area_thresh=config['cnt_min_area_thresh'])
    v_lines, h_lines = getLines(c_img)

    print(f'Refining lines using priors of grid size {grid}')
    v_lines_filtered = getMargins(img, v_lines, grid = grid, mode='v')
    h_lines_filtered = getMargins(img, h_lines, grid = grid, mode='h' )
    

    print(f'Clustering the refined lines to predict complete lines ')
    vertical_predicted_lines, v_lines_final = clusterNRegression(v_lines_filtered,img.copy(), mode='v')
    horizontal_predicted_lines, h_lines_final = clusterNRegression(h_lines_filtered,img.copy(), mode='h')


    if debug:

        # debug_im1 = np.hstack((img,preprocessed_img ,c_img ))
        debug_im2 = np.hstack((vertical_predicted_lines,c_img, horizontal_predicted_lines))
        # debug_im = np.vstack((debug_im1, debug_im2))

        cv2.imwrite(os.path.join(debug_path, image_name),cv2.resize(debug_im2,(0,0),fx=0.5,fy=0.5))


    print(f'Cropping the cells')


    crops = None

    # if write:
    #     cv2.imwrite(os.path.join(crp_path, 'lines.jpg'),np.hstack((horizontal_predicted_lines,vertical_predicted_lines)))


    crops = cropper(h_lines_final, v_lines_final, im_s.copy(), write=write, path=crp_path)
    print(f'Completed processing. Detected: {len(crops)} cells.')
    return crops, vertical_predicted_lines, horizontal_predicted_lines


if __name__ == '__main__':



    import glob
    # modify the following to run on your batch with same grid size
    files = glob.glob('../ResizedImages/**/*.jpg')



    for i in files:

            # if i == '22714686.jpg':
            # continue

        print(100*'-')
        print(100*'-')
        im = cv2.imread(i)

        basename = os.path.basename(i)
        print(basename)

        preprocess_ld(im, grid=(6,24), image_name = basename, 
                        bypass_prespective_transform=False, 
                        debug=True,
                        remove_shadow= False,
                        write=True)

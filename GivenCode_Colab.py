# main script
import os
import sys
import numpy as np
import cv2
import random
import json



try:
    
    
    from transformation import prespectiveTransform
    from utils import *
    from crop import cropper
    from lines import getLines, getMargins
except ImportError as e:
    print(f"Error importing module: {e}")
    sys.exit(1)

from sklearn.cluster import DBSCAN

config_path = os.path.join("/content", 'config.json')
if not os.path.exists(config_path):
    print(f"Config file not found at {config_path}")
    sys.exit(1)

with open(config_path, 'r') as f:
    config = json.load(f)

def resize(img, f=0.5):
    return cv2.resize(img, (0, 0), fx=f, fy=f)

def clusterNRegression(lines, img, grid=(12, 6), mode='v'):
    if mode == 'v':
        ys = []
        pts = []
        pred_lines = []
        for l in lines:
            pts.append(l[0][:2].tolist())
            pts.append(l[0][2:].tolist())
            ys.append((l[0][0], 0))
            ys.append((l[0][2], 0))

        d = 0.2 * img.shape[0] // grid[1]
        clustering = DBSCAN(eps=d, min_samples=2).fit(ys)
        lbs = clustering.labels_.tolist()

        collections = {l: [] for l in lbs}

        for p in range(len(pts)):
            cluster_label = lbs[p]
            collections[cluster_label].append(pts[p])

        for k, v in collections.items():
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            x = np.array(v)[:, :1].reshape(-1)
            y = np.array(v)[:, 1:].reshape(-1)
            x1, y1 = int(x.mean()), 0
            x2, y2 = int(x.mean()), img.shape[0]
            pred_lines.append(np.array([[x1, y1, x2, y2]]))
            cv2.line(img, (x1, y1), (x2, y2), (r, g, b), 15)

    elif mode == 'h':
        xs = []
        pts = []
        pred_lines = []
        for l in lines:
            pts.append(l[0][:2].tolist())
            pts.append(l[0][2:].tolist())
            xs.append((0, l[0][1]))
            xs.append((0, l[0][3]))

        d = 0.5 * img.shape[1] // grid[0]
        clustering = DBSCAN(eps=d, min_samples=2).fit(xs)
        lbs = clustering.labels_.tolist()

        collections = {l: [] for l in lbs}

        for p in range(len(pts)):
            cluster_label = lbs[p]
            collections[cluster_label].append(pts[p])

        for k, v in collections.items():
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            x = np.array(v)[:, :1].reshape(-1)
            y = np.array(v)[:, 1:].reshape(-1)
            m, b = np.polyfit(x, y, 1)
            x1, y1 = 0, int(m * x.mean() + int(b))
            x2, y2 = img.shape[1], int((img.shape[1] * m) + b)
            pred_lines.append(np.array([[x1, y1, x2, y2]]))
            cv2.line(img, (x1, y1), (x2, y2), (r, g, b), 15)

    return img, pred_lines

def drawContours(img, cnts, cont_size_percentage=0.001, area_thresh=None):
    c_img = np.zeros_like(img)
    if area_thresh is None:
        area_thresh = img.shape[0] * img.shape[1] * cont_size_percentage
    for i, c in enumerate(cnts):
        a = cv2.contourArea(c)
        if a > area_thresh:
            cv2.drawContours(c_img, [c], 0, (255, 255, 255), 1)
    return c_img

def removeShadow(img):
   rgb_planes = cv2.split(img)
   result_planes = []
   result_norm_planes = []
   for plane in rgb_planes:
       dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
       bg_img = cv2.medianBlur(dilated_img, 21)
       diff_img = 255 - cv2.absdiff(plane, bg_img)
       norm_img = cv2.normalize(diff_img, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
       result_planes.append(diff_img)
       result_norm_planes.append(norm_img)
   return cv2.merge(result_planes), cv2.merge(result_norm_planes)

def preprocess_ld(im_s, grid, image_name='abc.jpg', path='./results', write=False, debug=False, bypass_prespective_transform=False, remove_shadow=False):
    debug_path = os.path.join(path, 'debug')
    crp_path = os.path.join(path, 'crops', image_name)

    os.makedirs(crp_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)

    if remove_shadow:
        print('Equalizing image to remove shadow')
        im_o = removeShadow(im_s)[1]
    else:
        im_o = im_s.copy()

    print(f'Preprocessing and prespective transforming the image')
    im = cv2.bitwise_not(im_o.copy())
    gray = preprocess(im)

    if not bypass_prespective_transform:
        sts, transformed_img = prespectiveTransform(gray, im, area_thresh=config['presp_fail_safe_thresh'])
        if not sts:
            print(f'Perspective transformation failed for {image_name}')
            return None, None, None
        img = resize(transformed_img)
    else:
        print('Skipping perspective transformation. Assuming that the image is an accurate crop of the panel.')
        img = im.copy()

    print(f'Processing the image to find elements')
    blured = makeBlur(img.copy())
    preprocessed_img = cleanImage(blured)
    contours = findContours(preprocessed_img, l=config['cont_img_low'], h=config['cont_img_high'])[0]

    print(f'Extracting primary contours')
    c_img = drawContours(img, contours, area_thresh=config['cnt_min_area_thresh'])
    v_lines, h_lines = getLines(c_img)

    print(f'Refining lines using priors of grid size {grid}')
    v_lines_filtered = getMargins(img, v_lines, grid=grid, mode='v')
    h_lines_filtered = getMargins(img, h_lines, grid=grid, mode='h')

    print(f'Clustering the refined lines to predict complete lines')
    vertical_predicted_lines, v_lines_final = clusterNRegression(v_lines_filtered, img.copy(), mode='v')
    horizontal_predicted_lines, h_lines_final = clusterNRegression(h_lines_filtered, img.copy(), mode='h')

    if debug:
        debug_im2 = np.hstack((vertical_predicted_lines, c_img, horizontal_predicted_lines))
        cv2.imwrite(os.path.join(debug_path, image_name), cv2.resize(debug_im2, (0, 0), fx=0.5, fy=0.5))

    print(f'Cropping the cells')
    crops = cropper(h_lines_final, v_lines_final, im_s.copy(), write=write, path=crp_path)
    print(f'Completed processing. Detected: {len(crops)} cells.')
    return crops, vertical_predicted_lines, horizontal_predicted_lines

if __name__ == '__main__':

    inp_dir = "/content/drive/MyDrive/Trial/V crack/v crack"
    output_dir = os.path.join(inp_dir, "Results")
    output_dir_edge = os.path.join(inp_dir,'Cropped')

    files = [os.path.join(inp_dir, f) for f in os.listdir(inp_dir) if f.lower().endswith(('.jpg', '.png'))]

    for i in files:
        print(100 * '-')
        print(100 * '-')
        im = cv2.imread(i)

        if im is None:
            print(f"Error reading image {i}. Skipping...")
            continue

        basename = os.path.basename(i)
        print(basename)

        crops, v_lines, h_lines = preprocess_ld(im, grid=(6, 24), image_name=basename, 
                                                path=output_dir_edge, 
                                                bypass_prespective_transform=False, 
                                                debug=True,
                                                remove_shadow=False,
                                                write=True)
        if crops is None:
            print(f"Processing failed for {basename}")

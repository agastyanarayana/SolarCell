import numpy as np
import cv2
import os

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = int(Dx / D)
        y = int(Dy / D)
        return x, y
    else:
        return False

def cropper(h_lines_final, v_lines_final, img, write=False, path=None, prefix=""):

    '''
    create pairs of adjacent lines and get the intersection points. crop it from the original image. 
    '''    
    
    h, w, _ = img.shape

    crops = []
    
    for i in range(len(h_lines_final) - 1):
    
        h0 = np.squeeze(h_lines_final[i]) * 2
        h1 = np.squeeze(h_lines_final[i + 1]) * 2

        hl0 = line(h0[:2], h0[2:])
        hl1 = line(h1[:2], h1[2:])

        for j in range(len(v_lines_final) - 1):

            v0 = np.squeeze(v_lines_final[j]) * 2
            v1 = np.squeeze(v_lines_final[j + 1]) * 2

            vl0 = line(v0[:2], v0[2:])
            vl1 = line(v1[:2], v1[2:])

            tl = intersection(hl0, vl0)
            br = intersection(hl1, vl1)

            # contingency when lines are intersecting out of image plane
            tlx, tly = tl
            brx, bry = br

            tlx = max(0, tlx)
            tly = max(0, tly)

            brx = min(w, brx)
            bry = min(h, bry)

            c = img[tly:bry, tlx:brx]
            crops.append(c)

            if write:
                try:
                    name = os.path.join(path, f"{prefix}{i}{i + 1}_{j}{j + 1}.jpg")
                    cv2.imwrite(name, c)
                except Exception as e:
                    print(f"Error while writing {name}")
                    print(e)

    return crops

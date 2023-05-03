
# import the necessary packages
import numpy as np
import pandas as pd
import cv2
import os
import glob
import time
start_time = time.time()

far_locations = pd.read_csv('assets/demo/far_location_clean.csv')
close_locations = pd.read_csv('assets/demo/close_location_clean.csv')
court_coords = pd.read_csv('assets/demo/court_coords_clean.csv')

# load the image
images = sorted(glob.glob("assets/demo/all-frames/*.jpg"))
coords_df = np.zeros(shape=(len(images), 6))
i = -1

while i < len(images) - 1:

    i+= 1 

    print(i/len(images))

    if i == 0: continue

    img_ = images[i]
    img_1 = images[i-1]
    splits = os.path.splitext(os.path.basename(img_))[0].split('-')
    splits_prev = os.path.splitext(os.path.basename(img_1))[0].split('-')
    print(splits)
    if splits[3] != splits_prev[3]: continue
    if int(splits[4]) != int(splits_prev[4]) + 1: continue
    
    img = cv2.imread(img_)
    coords = court_coords.loc[(court_coords['chunk'] == int(splits[3])) & (court_coords['frame'] == int(splits[4]))].to_numpy().reshape(1, 10)

    srcpts = np.float32([[coords[0][2], coords[0][3]], [coords[0][4], coords[0][5]], [coords[0][6], coords[0][7]], [coords[0][8], coords[0][9]]])
    destpts = np.float32([[150, 930], [510, 930], [150, 150], [510, 150]])

    resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
    resultimage = cv2.warpPerspective(img, resmatrix, (660, 1080))

    p = close_locations.loc[(close_locations['chunk'] == int(splits[3])) & (close_locations['frame'] == int(splits[4]))].to_numpy().reshape(1, 4)[0]
    px = (resmatrix[0][0]*p[2] + resmatrix[0][1]*p[3] + resmatrix[0][2]) / ((resmatrix[2][0]*p[2] + resmatrix[2][1]*p[3] + resmatrix[2][2]))
    py = (resmatrix[1][0]*p[2] + resmatrix[1][1]*p[3] + resmatrix[1][2]) / ((resmatrix[2][0]*p[2] + resmatrix[2][1]*p[3] + resmatrix[2][2]))
    p_after = (int(px), int(py))

    q = far_locations.loc[(far_locations['chunk'] == int(splits[3])) & (far_locations['frame'] == int(splits[4]))].to_numpy().reshape(1, 4)[0]
    qx = (resmatrix[0][0]*q[2] + resmatrix[0][1]*q[3] + resmatrix[0][2]) / ((resmatrix[2][0]*q[2] + resmatrix[2][1]*q[3] + resmatrix[2][2]))
    qy = (resmatrix[1][0]*q[3] + resmatrix[1][1]*q[3] + resmatrix[1][2]) / ((resmatrix[2][0]*q[3] + resmatrix[2][1]*q[3] + resmatrix[2][2]))
    q_after = (int(qx), int(qy))

    r = [coords[0][2], coords[0][3]]
    rx = (resmatrix[0][0]*r[0] + resmatrix[0][1]*r[1] + resmatrix[0][2]) / ((resmatrix[2][0]*r[0] + resmatrix[2][1]*r[1] + resmatrix[2][2]))
    ry = (resmatrix[1][0]*r[0] + resmatrix[1][1]*r[1] + resmatrix[1][2]) / ((resmatrix[2][0]*r[0] + resmatrix[2][1]*r[1] + resmatrix[2][2]))
    r_after = (int(rx), int(ry))

    s = [coords[0][8], coords[0][9]]
    sx = (resmatrix[0][0]*s[0] + resmatrix[0][1]*s[1] + resmatrix[0][2]) / ((resmatrix[2][0]*s[0] + resmatrix[2][1]*s[1] + resmatrix[2][2]))
    sy = (resmatrix[1][0]*s[0] + resmatrix[1][1]*s[1] + resmatrix[1][2]) / ((resmatrix[2][0]*s[0] + resmatrix[2][1]*s[1] + resmatrix[2][2]))
    s_after = (int(sx), int(sy))
    
    coords_df[i] = splits[3:] + [int(px), int(py), int(qx), int(qy)]
    cv2.rectangle(resultimage, r_after, s_after,(0,255,0),3)
    cv2.circle(resultimage, p_after, 3, (255, 255, 0), -1)
    cv2.circle(resultimage, q_after, 3, (0, 255, 255), -1)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", resultimage)
    cv2.waitKey(50)

pd.DataFrame(coords_df, columns = ['chunk', 'frame', 'x1', 'y1', 'x2', 'y2']).to_csv('assets/demo/final_locations.csv', index = False)
print("--- %s seconds ---" % (time.time() - start_time))


# import the necessary packages
import numpy as np
import pandas as pd
import cv2
import os
import glob
import time
start_time = time.time()

court_coords = pd.read_csv('assets/demo/court_coords.csv')

# load the image
images = sorted(glob.glob("assets/demo/test/test-.jpg"))
coords_df = np.zeros(shape=(len(images), 6))
j = -1
for i in court_coords['frame']:
    j+=1
    if i == 0: continue

    img_ = "assets/demo/test/test-" + str(int(i)) + ".jpg"
    print(img_)
    #img_1 = images[i-1]
    splits = os.path.splitext(os.path.basename(img_))[0].split('-')
    #splits_prev = os.path.splitext(os.path.basename(img_1))[0].split('-')
    print(splits)
    #if splits[3] != splits_prev[3]: continue
    #if int(splits[4]) != int(splits_prev[4]) + 1: continue
    
    img = cv2.imread(img_)
    coords = court_coords.loc[j].to_numpy().reshape(1, 9)
    srcpts = np.float32([[coords[0][2], coords[0][3]], [coords[0][0], coords[0][1]], [coords[0][6], coords[0][7]], [coords[0][4], coords[0][5]]])
    destpts = np.float32([[150, 750], [150, 930], [510, 750], [510, 930]])

    resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
    resultimage = cv2.warpPerspective(img, resmatrix, (660, 1080))

    r = [coords[0][2], coords[0][3]]
    rx = (resmatrix[0][0]*r[0] + resmatrix[0][1]*r[1] + resmatrix[0][2]) / ((resmatrix[2][0]*r[0] + resmatrix[2][1]*r[1] + resmatrix[2][2]))
    ry = (resmatrix[1][0]*r[0] + resmatrix[1][1]*r[1] + resmatrix[1][2]) / ((resmatrix[2][0]*r[0] + resmatrix[2][1]*r[1] + resmatrix[2][2]))
    r_after = (int(rx), int(ry))

    s = [coords[0][4], coords[0][5]]
    sx = (resmatrix[0][0]*s[0] + resmatrix[0][1]*s[1] + resmatrix[0][2]) / ((resmatrix[2][0]*s[0] + resmatrix[2][1]*s[1] + resmatrix[2][2]))
    sy = (resmatrix[1][0]*s[0] + resmatrix[1][1]*s[1] + resmatrix[1][2]) / ((resmatrix[2][0]*s[0] + resmatrix[2][1]*s[1] + resmatrix[2][2]))
    s_after = (int(sx), int(sy))
    

    cv2.rectangle(resultimage, r_after, s_after,(0,255,0), 3)
    cv2.rectangle(resultimage, (150, 930), (510, 150),(0,255,0), 3)
    cv2.imshow("Court Transform", resultimage)
    cv2.waitKey(50)

pd.DataFrame(coords_df, columns = ['chunk', 'frame', 'x1', 'y1', 'x2', 'y2']).to_csv('assets/demo/final_locations.csv', index = False)
print("--- %s seconds ---" % (time.time() - start_time))
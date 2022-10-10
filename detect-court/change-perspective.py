
# import the necessary packages
import numpy as np
import pandas as pd
import cv2
import time
start_time = time.time()

game = "assets/game-frames/hard-w-2022-70-" 
far_locations = pd.read_csv('far_location_clean.csv')
close_locations = pd.read_csv('close_location_clean.csv')
court_coords = pd.read_csv('court_coords_clean.csv')

# load the image
i = 210

while i < 234:

    img = cv2.imread(game + str(i) + ".jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    coords = court_coords[(court_coords['frame'] == i)].to_numpy().reshape(1, 9)

    srcpts = np.float32([[coords[0][1], coords[0][2]], [coords[0][3], coords[0][4]], [coords[0][5], coords[0][6]], [coords[0][7], coords[0][8]]])
    destpts = np.float32([[100, 880], [460, 880], [100, 100], [460, 100]])

    resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
    resultimage = cv2.warpPerspective(img, resmatrix, (560, 980))

    p = close_locations[(close_locations['frame'] == i)].to_numpy().reshape(1, 3)[0]
    px = (resmatrix[0][0]*p[1] + resmatrix[0][1]*p[2] + resmatrix[0][2]) / ((resmatrix[2][0]*p[1] + resmatrix[2][1]*p[2] + resmatrix[2][2]))
    py = (resmatrix[1][0]*p[1] + resmatrix[1][1]*p[2] + resmatrix[1][2]) / ((resmatrix[2][0]*p[1] + resmatrix[2][1]*p[2] + resmatrix[2][2]))
    p_after = (int(px), int(py))

    q = far_locations[(far_locations['frame'] == i)].to_numpy().reshape(1, 3)[0]
    qx = (resmatrix[0][0]*q[1] + resmatrix[0][1]*q[2] + resmatrix[0][2]) / ((resmatrix[2][0]*q[1] + resmatrix[2][1]*q[2] + resmatrix[2][2]))
    qy = (resmatrix[1][0]*q[1] + resmatrix[1][1]*q[2] + resmatrix[1][2]) / ((resmatrix[2][0]*q[1] + resmatrix[2][1]*q[2] + resmatrix[2][2]))
    q_after = (int(qx), int(qy))

    # cv2.circle(resultimage, p_after, 3, (255, 255, 0), -1)
    # cv2.circle(resultimage, q_after, 3, (0, 255, 255), -1)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", resultimage)
    # cv2.waitKey(0)
    i+=1

print("--- %s seconds ---" % (time.time() - start_time))

# import the necessary packages
import numpy as np
import pandas as pd
import cv2


# load the image
img = cv2.imread("assets/game-frames/hard-w-2020-82-450.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

srcpts = np.float32([[72, 450], [780, 450], [265, 120], [587, 120]])
destpts = np.float32([[0, 0], [360, 0], [0, 780], [360, 780]])
resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
resultimage = cv2.warpPerspective(img, resmatrix, (360, 780))

p = (500,400)
px = (resmatrix[0][0]*p[0] + resmatrix[0][1]*p[1] + resmatrix[0][2]) / ((resmatrix[2][0]*p[0] + resmatrix[2][1]*p[1] + resmatrix[2][2]))
py = (resmatrix[1][0]*p[0] + resmatrix[1][1]*p[1] + resmatrix[1][2]) / ((resmatrix[2][0]*p[0] + resmatrix[2][1]*p[1] + resmatrix[2][2]))
p_after = (int(px), int(py))
# cv2.perspectiveTransform(points, matrix)
cv2.circle(resultimage, p_after, 3, (0, 255, 0), -1)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", resultimage)
cv2.waitKey(0)
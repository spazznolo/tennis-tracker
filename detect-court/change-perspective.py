
# import the necessary packages
import numpy as np
import pandas as pd
import cv2


# load the image
img = cv2.imread("assets/game-frames/clay-m-2009-65-250.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

srcpts = np.float32([[100, 400], [750, 400], [250, 100], [610, 100]])
destpts = np.float32([[0, 0], [360, 0], [0, 780], [360, 780]])
#applying PerspectiveTransform() function to transform the perspective of the given source image to the corresponding points in the destination image
resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
# #applying warpPerspective() function to display the transformed image
resultimage = cv2.warpPerspective(img, resmatrix, (360, 780))

cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", resultimage)
cv2.waitKey(0)

# import the necessary packages
import numpy as np
import cv2
from skimage.feature import (corner_harris, corner_subpix, corner_peaks, plot_matches)

# load the image
img = cv2.imread("assets/game-frames/hard-m-2019-128-800.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,0,100,apertureSize = 3)
dst = cv2.cornerHarris(edges, 2, 11, 0.2)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.1*dst.max()]=[0,0,255]
cv2.imshow('dst', img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
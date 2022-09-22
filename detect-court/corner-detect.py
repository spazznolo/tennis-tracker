
# import the necessary packages
import numpy as np
import cv2
from skimage.feature import (corner_harris, corner_subpix, corner_peaks, plot_matches)

# load the image
img = cv2.imread("assets/game-frames/grass-w-2019.mp4/frame_28_1400.jpg")
#img = cv2.imread("assets/chunk_0/frame_1173.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,100,apertureSize = 3)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# define range of white color in HSV
# change it according to your need !
lower_white = np.array([0,0,250], dtype=np.uint8)
upper_white = np.array([255,255,255], dtype=np.uint8)

# Threshold the HSV image to get only white colors
mask = cv2.inRange(hsv, lower_white, upper_white)

dst = cv2.cornerHarris(edges, 8, 29, 0.2)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.1*dst.max()]=[0,0,255]
cv2.imshow('dst', img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
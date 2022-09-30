
# import the necessary packages
import math
import numpy as np
import pandas as pd
import cv2

# load the image
img = cv2.imread("assets/game-frames/hard-w-2020-70-1319.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel_size = 1
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
edges = cv2.Canny(blur_gray, 50, 100, apertureSize = 3)

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower_white = np.array([0,0,180], dtype=np.uint8)
# upper_white = np.array([255,255,255], dtype=np.uint8)
# mask = cv2.inRange(hsv, lower_white, upper_white)

lines = cv2.HoughLinesP(edges, 2, np.pi/180, 100, minLineLength=20, maxLineGap=10)

# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  

#lines_df.to_csv('assets/temp/hough_lines.csv')
#cv2.imwrite('/Users/ada/Documents/projects/spazznolo.github.io/figs/hough-line-ex-2.jpg',img)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)
cv2.waitKey(0)

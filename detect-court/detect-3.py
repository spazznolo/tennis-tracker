
import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread("assets/game-frames/clay-m-2009-65-250.jpg")
#crop_img = img[50:145, 375:470]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 0], dtype=np.uint8)
upper_white = np.array([250, 250, 200], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_white, upper_white)

lines = cv2.HoughLinesP(mask, 1, np.pi/180, 100, minLineLength=20, maxLineGap=20)

# Draw lines on the image
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)  


#lines_df.to_csv('assets/temp/hough_lines.csv')
#cv2.imwrite('/Users/ada/Documents/projects/spazznolo.github.io/figs/hough-line-ex-3.jpg',img)

cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", mask)
cv2.waitKey(0)
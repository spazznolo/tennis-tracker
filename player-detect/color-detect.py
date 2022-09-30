
import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread("assets/game-frames/hard-w-2020-58-560.jpg")
#crop_img = img[50:145, 375:470]

def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
lower_white = np.array([bincount_app(hsv)[0] - 10, bincount_app(hsv)[1] - 10, bincount_app(hsv)[2] - 10], dtype=np.uint8)
upper_white = np.array([bincount_app(hsv)[0] + 10, bincount_app(hsv)[1] + 10, bincount_app(hsv)[2] + 10], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_white, upper_white)

lines = cv2.HoughLinesP(mask, 1, np.pi/180, 100, minLineLength=20, maxLineGap=20)

# Draw lines on the image
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)  


#lines_df.to_csv('assets/temp/hough_lines.csv')
#cv2.imwrite('/Users/ada/Documents/projects/spazznolo.github.io/figs/hough-line-ex-3.jpg',img)

cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", h)
cv2.waitKey(0)
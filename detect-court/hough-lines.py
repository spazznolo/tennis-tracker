
# import the necessary packages
import math
import numpy as np
import pandas as pd
import cv2

width_error = 10
height_error = 10
kernel_size = 1

# load the image
img = cv2.imread("assets/game-frames/hard-w-2022-89-1100.jpg")
print(img.shape)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
edges = cv2.Canny(blur_gray, 50, 100, apertureSize = 3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)

# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        slope = ((y0 + 1000*(a)) - (y0 - 1000*(a)))/((x0 + 1000*(-b)) - (x0 - 1000*(-b)))
        if abs(slope) > 1 and abs(slope) < 3:
            cv2.line(img, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)

cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)
cv2.waitKey(0)
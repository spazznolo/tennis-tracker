
# import the necessary packages
import math
import numpy as np
import pandas as pd
import cv2
from skimage.morphology import skeletonize

# load the image
img = cv2.imread("assets/game-frames/hard-w-2004.mp4/frame_17_7.jpg")
#img = cv2.imread("assets/game-frames/hard-m-2021.mp4/frame_13_202.jpg")
#img = cv2.imread("assets/game-frames/grass-w-2019.mp4/frame_3_350.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100, apertureSize = 3)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_white = np.array([0,0,250], dtype=np.uint8)
upper_white = np.array([255,255,255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_white, upper_white)

lines = cv2.HoughLinesP(edges, 2, np.pi/180, 100, minLineLength=20, maxLineGap=10)
blank_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
df1 = pd.DataFrame(np.reshape([0, 0, 0, 0], (1, 4)), columns = ['x1', 'y1', 'x2', 'y2'])

# Draw lines on the image
for line in lines:

    x1, y1, x2, y2 = line[0]

    length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    df1 = pd.concat([df1, pd.DataFrame(np.reshape(line[0], (1, 4)), columns = ['x1', 'y1', 'x2', 'y2'])])

    if abs((y2 - y1)/(x2 - x1)) > 1.5 and abs((y2 - y1)/(x2 - x1)) < 3 and length > 100:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

    else:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)

#skeleton = skeletonize(blank_image)
df1.to_csv('hough_lines.csv')
#cv2.imwrite('houghlines3.jpg',img)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)
cv2.waitKey(0)
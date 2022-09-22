
# import the necessary packages
import math
import numpy as np
import pandas as pd
import cv2
from skimage.morphology import skeletonize

# load the image
img = cv2.imread("assets/game-chunks/game-2/chunk_20/frame_294.jpg")
#img = cv2.imread("assets/test frames/clay.jpeg")
#img = cv2.imread("assets/chunk_0/frame_1173.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100, apertureSize = 3)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_white = np.array([0,0,250], dtype=np.uint8)
upper_white = np.array([255,255,255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_white, upper_white)

lines = cv2.HoughLinesP(mask, 2, np.pi/180, 100, minLineLength=20, maxLineGap=10)
blank_image = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
df1 = pd.DataFrame(np.reshape([0, 0, 0, 0], (1, 4)), columns = ['x1', 'y1', 'x2', 'y2'])

# Draw lines on the image
for line in lines:

    x1, y1, x2, y2 = line[0]

    length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    df1 = pd.concat([df1, pd.DataFrame(np.reshape(line[0], (1, 4)), columns = ['x1', 'y1', 'x2', 'y2'])])
    
    if x1 == x2 or x1 < 10 or y1 < 100: 
        continue

    # elif abs((y2 - y1)/(x2 - x1)) > 1 and abs((y2 - y1)/(x2 - x1)) < 5 and length > 100:
    #     cv2.line(blank_image, (x1, y1), (x2, y2), (0, 0, 255), 5)

    else:
        cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

#skeleton = skeletonize(blank_image)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", blank_image)
cv2.waitKey(0)
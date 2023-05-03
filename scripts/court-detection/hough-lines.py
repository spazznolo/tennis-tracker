
# import the necessary packages
import os
import math
import numpy as np
import cv2
import time

width_error = 10
height_error = 10
kernel_size = 1

# Start the timer
start_time = time.time()

# load the image
path = 'assets/highlights/object-detection-train-frames/'

for filename in os.listdir(path)[0:499]:

    if filename.endswith('.jpg'):

        img = cv2.imread(os.path.join(path, filename))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        edges = cv2.Canny(blur_gray, 50, 100, apertureSize = 3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        # Draw the lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 1 and abs(slope) < 5:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                if abs(slope) < 0.1:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Detected Lines (in red) - HoughLinesP", img)
        cv2.waitKey(0)

# End the timer
end_time = time.time()

# Calculate the total execution time
total_time = end_time - start_time

# Print the total execution time in seconds
print(f'Total execution time: {total_time} seconds')

# import the necessary packages
import random
import numpy as np
import pandas as pd
import cv2

width_error = 10
height_error = 10
kernel_size = 1

import os

appended_data = []

for x in range(5):

    # load the image
    random_file = random.choice(os.listdir("assets/train/on"))
    print(random_file)
    img = cv2.imread('assets/train/on/' + str(random_file))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    edges = cv2.Canny(blur_gray, 50, 100, apertureSize = 3)
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, 100, minLineLength=20, maxLineGap=10)

    lines_df = pd.DataFrame(np.reshape(lines, (np.int32(lines.size/4), 4)), columns = ['x1', 'y1', 'x2', 'y2'])
    lines_df['y_min'] = lines_df[["y1", "y2"]].min(axis=1)
    lines_df['y_max'] = lines_df[["y1", "y2"]].max(axis=1)
    lines_df['x_min'] = lines_df[["x1", "x2"]].min(axis=1)
    lines_df['x_max'] = lines_df[["x1", "x2"]].max(axis=1)
    lines_df['slope'] = (lines_df['y2'] - lines_df['y1'])/(lines_df['x2'] - lines_df['x1'])
    lines_df['length'] = np.sqrt(((lines_df['y_max'] - lines_df['y_min'])**2) + ((lines_df['x_max'] - lines_df['x_min'])**2))
    lines_df['image'] = random_file

    lengthwise_lines = lines_df[(abs(lines_df['slope']) > 1.0) & (abs(lines_df['slope']) < 3.0) & (lines_df['length'] > 100)]

    y_max = lengthwise_lines['y_max'].max()
    y_min = lengthwise_lines['y_min'].min()
    x_max = lengthwise_lines['x_max'].max()
    x_min = lengthwise_lines['x_min'].min()

    widthwise_lines = lines_df[(abs(lines_df['slope']) < 0.10)]
    bottom_lines = widthwise_lines[(widthwise_lines['y_max'] < y_max + height_error) & (widthwise_lines['y_max'] > y_max - height_error)]
    bottom_lines = bottom_lines[(bottom_lines['x_max'] < x_max + width_error) & (bottom_lines['x_min'] > x_min - width_error)]
    top_lines = widthwise_lines[(widthwise_lines['y_min'] < y_min + height_error) & (widthwise_lines['y_min'] > y_min - height_error)]
    top_lines = top_lines[(top_lines['x_max'] < x_max + width_error) & (top_lines['x_min'] > x_min - width_error)]

    appended_data.append(lengthwise_lines)
    appended_data.append(bottom_lines)
    appended_data.append(top_lines)

    for row in lengthwise_lines.itertuples():
        cv2.line(img, (row.x1, row.y1), (row.x2, row.y2), (255, 0, 255), 2) 

    for row in bottom_lines.itertuples():
        cv2.line(img, (row.x1, row.y1), (row.x2, row.y2), (0, 255, 255), 2) 

    for row in top_lines.itertuples():
        cv2.line(img, (row.x1, row.y1), (row.x2, row.y2), (255, 255, 0), 2) 

    #lines_df.to_csv('assets/temp/hough_lines.csv')
    cv2.imwrite('assets/hough-batch/' + str(random_file) + '.jpg',img)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)
    cv2.waitKey(0)

# see pd.concat documentation for more info
appended_data = pd.concat(appended_data)
# write DataFrame to an excel sheet 
appended_data.to_csv('assets/temp/hough_lines.csv')
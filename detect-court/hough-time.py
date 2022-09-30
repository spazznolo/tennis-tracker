
# import the necessary packages
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import AgglomerativeClustering
from time import process_time

width_error = 10
height_error = 10
kernel_size = 11

t1_start = process_time() 

for x in range(25):
        
    # load the image
    img = cv2.imread("assets/game-frames/hard-m-2019-128-824.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    edges = cv2.Canny(blur_gray, 0, 100, apertureSize = 3)
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, 100, minLineLength=20, maxLineGap=10)

    lines_df = pd.DataFrame(np.reshape(lines, (np.int32(lines.size/4), 4)), columns = ['x1', 'y1', 'x2', 'y2'])
    lines_df['y_min'] = lines_df[["y1", "y2"]].min(axis=1)
    lines_df['y_max'] = lines_df[["y1", "y2"]].max(axis=1)
    lines_df['x_min'] = lines_df[["x1", "x2"]].min(axis=1)
    lines_df['x_max'] = lines_df[["x1", "x2"]].max(axis=1)
    lines_df['slope'] = (lines_df['y2'] - lines_df['y1'])/(lines_df['x2'] - lines_df['x1'])
    lines_df['abs_slope'] = abs(lines_df['slope'])
    lines_df['length'] = np.sqrt(((lines_df['y_max'] - lines_df['y_min'])**2) + ((lines_df['x_max'] - lines_df['x_min'])**2))

    lengthwise_lines = lines_df[(lines_df['abs_slope'] > 1.00) & (lines_df['abs_slope'] < 3.5) & (lines_df['length'] > 100)]

    if len(lengthwise_lines.index) == 0:
        print('No lengthwise lines!')
        exit()

    ward = AgglomerativeClustering(n_clusters = None, distance_threshold = 0.50, linkage = "ward").fit(lengthwise_lines[['abs_slope']])
    lengthwise_lines['clust'] = ward.labels_

    want_clusts = lengthwise_lines['clust'].value_counts()[:2].index.tolist()
    lengthwise_lines = lengthwise_lines[lengthwise_lines['clust'].isin(want_clusts)]

    ward = AgglomerativeClustering(n_clusters = None, distance_threshold = 10, linkage = "ward").fit(lengthwise_lines[['y_min']])
    lengthwise_lines['clust_top'] = ward.labels_
    want_clusts = lengthwise_lines['clust_top'].value_counts()[:1].index.tolist()
    y_min = lengthwise_lines[lengthwise_lines['clust_top'].isin(want_clusts)].groupby('clust_top')['y_min'].transform('median').median()

    ward = AgglomerativeClustering(n_clusters = None, distance_threshold = 10, linkage = "ward").fit(lengthwise_lines[['y_max']])
    lengthwise_lines['clust_bot'] = ward.labels_
    want_clusts = lengthwise_lines['clust_bot'].value_counts()[:1].index.tolist()
    y_max = lengthwise_lines[lengthwise_lines['clust_bot'].isin(want_clusts)].groupby('clust_bot')['y_max'].transform('median').median()

    lengthwise_lines['avg_slope'] = lengthwise_lines.groupby('clust')['abs_slope'].transform('mean')
    lengthwise_lines = lengthwise_lines[lengthwise_lines['avg_slope'] == lengthwise_lines['avg_slope'].min()]
    lengthwise_lines['b'] = lengthwise_lines['y1'] - (lengthwise_lines['slope']*lengthwise_lines['x1'])


    l_left = lengthwise_lines[lengthwise_lines['slope'] < 0]
    l_right = lengthwise_lines[lengthwise_lines['slope'] > 0]


    #y_max = lengthwise_lines['y_max'].max()
    #y_min = lengthwise_lines['y_min'].min()
    x_max = lengthwise_lines['x_max'].max()
    x_min = lengthwise_lines['x_min'].min()

    widthwise_lines = lines_df[(abs(lines_df['slope']) < 0.10)]

    bottom_lines = widthwise_lines[(widthwise_lines['y_max'] < y_max + height_error) & (widthwise_lines['y_max'] > y_max - height_error)]
    bottom_lines = bottom_lines[(bottom_lines['x_max'] < x_max + width_error) & (bottom_lines['x_min'] > x_min - width_error)]

    top_lines = widthwise_lines[(widthwise_lines['y_min'] < y_min + height_error) & (widthwise_lines['y_min'] > y_min - height_error)]
    top_lines = top_lines[(top_lines['x_max'] < x_max + width_error) & (top_lines['x_min'] > x_min - width_error)]

    # for row in lengthwise_lines.itertuples():
    #     cv2.line(img, (row.x1, row.y1), (row.x2, row.y2), (255, 0, 255), 2) 

    # for row in widthwise_lines.itertuples():
    #     cv2.line(img, (row.x1, row.y1), (row.x2, row.y2), (0, 255, 255), 2) 

    # for row in bottom_lines.itertuples():
    #     cv2.line(img, (row.x1, row.y1), (row.x2, row.y2), (0, 255, 255), 2) 

    # for row in top_lines.itertuples():
    #     cv2.line(img, (row.x1, row.y1), (row.x2, row.y2), (255, 255, 0), 2) 

    # cv2.line(img, (int(-l_left['b'].median()/l_left['slope'].median()), int(0)), (int((854-l_left['b'].median())/l_left['slope'].median()), int(854)), (0, 255, 0), 2)
    # cv2.line(img, (int(-l_right['b'].median()/l_right['slope'].median()), int(0)), (int((854-l_right['b'].median())/l_right['slope'].median()), int(854)), (0, 255, 0), 2)

    # #lines_df.to_csv('assets/temp/hough_lines.csv')
    # cv2.imwrite('/Users/ada/Documents/projects/spazznolo.github.io/figs/hough-line-ex-2.jpg',img)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)
    # cv2.waitKey(0)
    # 
t1_stop = process_time() 

print(t1_stop-t1_start)
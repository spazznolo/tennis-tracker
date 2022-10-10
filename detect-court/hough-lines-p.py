
# import the necessary packages
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import AgglomerativeClustering

width_error = 10
height_error = 10
kernel_size = 7
kernel_dilate = np.ones((1, 1), 'uint8')
kernel = np.ones((5, 5), 'uint8')

# load the image
img = cv2.imread("assets/game-frames/hard-w-2022-67-1320.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
edges = cv2.Canny(blur_gray, 0, 100, apertureSize = 3)
dilate_img = cv2.dilate(edges, kernel_dilate, iterations=3)
closing = cv2.morphologyEx(dilate_img, cv2.MORPH_CLOSE, kernel)
lines = cv2.HoughLinesP(closing, 2, np.pi/180, 100, minLineLength=20, maxLineGap=10)

lines_df = pd.DataFrame(np.reshape(lines, (np.int32(lines.size/4), 4)), columns = ['x1', 'y1', 'x2', 'y2'])
lines_df['y_min'] = lines_df[["y1", "y2"]].min(axis=1)
lines_df['y_max'] = lines_df[["y1", "y2"]].max(axis=1)
lines_df['x_min'] = lines_df[["x1", "x2"]].min(axis=1)
lines_df['x_max'] = lines_df[["x1", "x2"]].max(axis=1)
lines_df['slope'] = (lines_df['y2'] - lines_df['y1'])/(lines_df['x2'] - lines_df['x1'])
lines_df['abs_slope'] = abs(lines_df['slope'])
lines_df['length'] = np.sqrt(((lines_df['y_max'] - lines_df['y_min'])**2) + ((lines_df['x_max'] - lines_df['x_min'])**2))

lengthwise_lines = lines_df[(lines_df['abs_slope'] > 1.20) & (lines_df['abs_slope'] < 3.5) & (lines_df['length'] > 100)]

if len(lengthwise_lines.index) == 0:
    print('No lengthwise lines!')
    exit()

ward = AgglomerativeClustering(n_clusters = None, distance_threshold = 0.50, linkage = "ward").fit(lengthwise_lines[['abs_slope']])
lengthwise_lines['clust'] = ward.labels_

want_clusts = lengthwise_lines['clust'].value_counts()[:2].index.tolist()
print(lengthwise_lines)
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

x_max = lengthwise_lines['x_max'].max()
x_min = lengthwise_lines['x_min'].min()

widthwise_lines = lines_df[(abs(lines_df['slope']) < 0.10)]
widthwise_lines['b'] = widthwise_lines['y1'] - (widthwise_lines['slope']*widthwise_lines['x1'])

bottom_lines = widthwise_lines[(widthwise_lines['y_max'] < y_max + height_error) & (widthwise_lines['y_max'] > y_max - height_error)]
bottom_lines = bottom_lines[(bottom_lines['x_max'] < x_max + width_error) & (bottom_lines['x_min'] > x_min - width_error)]

top_lines = widthwise_lines[(widthwise_lines['y_min'] < y_min + height_error) & (widthwise_lines['y_min'] > y_min - height_error)]
top_lines = top_lines[(top_lines['x_max'] < x_max + width_error) & (top_lines['x_min'] > x_min - width_error)]

l_left_b = l_left['b'].median()
l_left_m = l_left['slope'].median()

l_right_b = l_right['b'].median()
l_right_m = l_right['slope'].median()

bottom_lines_b = bottom_lines['b'].median()
bottom_lines_m = bottom_lines['slope'].median()

top_lines_b = top_lines['b'].median()
top_lines_m = top_lines['slope'].median()

def line_intersect(m1, b1, m2, b2):

    if m1 == m2:
        print ("These lines are parallel!!!")
        return None

    x = int((b2 - b1) / (m1 - m2))

    y = int(m1 * x + b1)
    return x,y

int_pt1 = line_intersect(l_left_m, l_left_b, bottom_lines_m, bottom_lines_b)
int_pt2 = line_intersect(l_right_m, l_right_b, bottom_lines_m, bottom_lines_b)
int_pt3 = line_intersect(l_left_m, l_left_b, top_lines_m, top_lines_b)
int_pt4 = line_intersect(l_right_m, l_right_b, top_lines_m, top_lines_b)
print(int_pt1)
print(int_pt2)
print(int_pt3)
print(int_pt4)
cv2.circle(img, int_pt1, 1, (255, 0, 0), -1)
cv2.circle(img, int_pt2, 1, (255, 0, 0), -1)
cv2.circle(img, int_pt3, 1, (255, 0, 0), -1)
cv2.circle(img, int_pt4, 1, (255, 0, 0), -1)

cv2.imwrite('/Users/ada/Documents/projects/spazznolo.github.io/figs/hough-line-ex-4.jpg',img)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)
cv2.waitKey(0)




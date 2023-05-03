
import numpy as np
import pandas as pd
import glob
import os
import time
import cv2
from sklearn.cluster import AgglomerativeClustering

start_time = time.time()

width_error = 10
height_error = 10
kernel_size = 7
kernel_dilate = np.ones((1, 1), 'uint8')
kernel = np.ones((5, 5), 'uint8')

i = -1

def line_intersect(m1, b1, m2, b2):

    if m1 == m2:
        print ("These lines are parallel!!!")
        return None

    x = int((b2 - b1) / (m1 - m2))

    y = int(m1 * x + b1)

    return [x,y]

images = sorted(glob.glob("assets/train/on/hard-m-2019-20-3*.jpg"))
#images = sorted(glob.glob("assets/demo/test/*.jpg"))
court_coords = np.zeros(shape=(len(images), 10))

for img_ in images:
    i+=1
    print(i/len(images))
    print(img_)

    splits = os.path.splitext(os.path.basename(img_))[0].split('-')
    splits = splits[3:]
    img = cv2.imread(img_)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    edges = cv2.Canny(blur_gray, 0, 100, apertureSize = 3)
    dilate_img = cv2.dilate(edges, kernel_dilate, iterations=3)
    closing = cv2.morphologyEx(dilate_img, cv2.MORPH_CLOSE, kernel)

    # bottle-neck: too loose a filter on hough lines? takes 1x real-time
    lines = cv2.HoughLinesP(closing, 2, np.pi/180, 100, minLineLength=20, maxLineGap=10)
    if lines is None: continue
    lines_np = np.reshape(lines, (np.int32(lines.size/4), 4))
    lines_np = lines_np[(lines_np[:, 2] - lines_np[:, 0]) != 0]
    lines_np = np.c_[lines_np, (lines_np[:, 3] - lines_np[:, 1])/(lines_np[:, 2] - lines_np[:, 0])]
    lines_np = np.c_[lines_np, np.sqrt((lines_np[:, 3] - lines_np[:, 1])**2 + (lines_np[:, 2] - lines_np[:, 0])**2)]
    lengthwise_lines = lines_np[(abs(lines_np[:, 4]) < 3.5) & (abs(lines_np[:, 4]) > 1.20) & (lines_np[:, 5] > 100)]

    court_coords[i] = splits + [0, 0, 0, 0, 0, 0, 0, 0]

    if lengthwise_lines[:, 4].size < 2:
        print('No lengthwise lines!')
        continue

    # bottle-neck: takes 0.175 real-time
    ward_length = AgglomerativeClustering(n_clusters = None, distance_threshold = 0.50, linkage = "ward").fit(abs(lengthwise_lines[:, 4].reshape(-1,1)))
    lengthwise_lines = np.c_[lengthwise_lines, ward_length.labels_]
    length_clusts = np.bincount(ward_length.labels_).argsort()[-2:]
    lengthwise_lines = lengthwise_lines[np.isin(ward_length.labels_, length_clusts)]
    x_max = np.max(lengthwise_lines[:,[0,2]].reshape(-1,1))
    x_min = np.amin(lengthwise_lines[:,[0,2]])

    # bottle-neck: takes 0.100 real-time
    y_coords = lengthwise_lines[:, [1, 3]].reshape(-1,1)
    ward = AgglomerativeClustering(n_clusters = None, distance_threshold = 10, linkage = "ward").fit(y_coords)
    want_clusts = np.bincount(ward.labels_).argsort()[-2:]
    y_ext1 = np.median(y_coords[np.isin(ward.labels_, want_clusts[0])])
    y_ext2 = np.median(y_coords[np.isin(ward.labels_, want_clusts[1])])

    # bottle-neck: takes 0.100 real-time
    length_want = ward_length.labels_[np.isin(ward_length.labels_, length_clusts)][np.argmin(abs(lengthwise_lines[:,4]))]
    lengthwise_lines = lengthwise_lines[lengthwise_lines[:, 6] == length_want]
    lengthwise_lines = np.c_[lengthwise_lines, (lengthwise_lines[:,1] - (lengthwise_lines[:,4]*lengthwise_lines[:,0]))]

    lengthwise_lines = np.c_[lengthwise_lines, ((np.max(lengthwise_lines[:,1]) - lengthwise_lines[:,5])/lengthwise_lines[:,4])]

    l_left = lengthwise_lines[lengthwise_lines[:,4] < 0]
    l_right = lengthwise_lines[lengthwise_lines[:,4] > 0]

    y_max = max(y_ext1, y_ext2)
    y_min = min(y_ext1, y_ext2)

    widthwise_lines = lines_np[(abs(lines_np[:, 4]) < 0.10)]
    widthwise_lines = np.c_[widthwise_lines, (widthwise_lines[:,1] - (widthwise_lines[:,4]*widthwise_lines[:,0]))]

    bottom_lines = widthwise_lines[np.where((np.amin(widthwise_lines[:,[1,3]], axis = 1) >= y_min - height_error) & (np.amin(widthwise_lines[:,[1,3]], axis = 1) <= y_min + height_error) & (np.amin(widthwise_lines[:,[0,2]], axis = 1) >= x_min - width_error) & (np.amax(widthwise_lines[:,[0,2]], axis = 1) <= x_max + width_error))[0]]
    top_lines = widthwise_lines[np.where((np.amax(widthwise_lines[:,[1,3]], axis = 1) >= y_max - height_error) & (np.amax(widthwise_lines[:,[1,3]], axis = 1) <= y_max + height_error) & (np.amin(widthwise_lines[:,[0,2]], axis = 1) >= x_min - width_error) & (np.amax(widthwise_lines[:,[0,2]], axis = 1) <= x_max + width_error))[0]]

    if top_lines.size == 0 or bottom_lines.size == 0 or l_left.size == 0 or l_right.size == 0:
        print('Nope!')
        continue

    l_left_b = np.median(l_left[:,7])
    l_left_m = np.median(l_left[:,4])

    l_right_b = np.median(l_right[:,7])
    l_right_m = np.median(l_right[:,4])

    top_lines_b = np.median(top_lines[:,6])
    top_lines_m = np.median(top_lines[:,4])
    if top_lines_m == 0: top_lines_b = np.median(top_lines[:,3])

    bottom_lines_b = np.median(bottom_lines[:,6])
    bottom_lines_m = np.median(bottom_lines[:,4])

    if top_lines_m == 0: 
        bottom_lines_m = 0

    if bottom_lines_m == 0: 
        bottom_lines_b = np.median(bottom_lines[:,3])

    if l_left.size == 0 or l_right.size == 0:
        print('No lengthwise lines!')
        continue

    max_x = int(l_left[np.argmax(l_left[:,1]),0])
    max_y = int(l_left[np.argmax(l_left[:,1]),1])
    min_x = int(l_left[np.argmin(l_left[:,3]),2])
    min_y = int(l_left[np.argmin(l_left[:,3]),3])

    l_left_m = (max_y - min_y)/(max_x - min_x)
    l_left_b = max_y - (l_left_m*max_x)

    int_pt1 = line_intersect(l_left_m, l_left_b, bottom_lines_m, bottom_lines_b)
    int_pt2 = line_intersect(l_right_m, l_right_b, bottom_lines_m, bottom_lines_b)
    int_pt3 = line_intersect(l_left_m, l_left_b, top_lines_m, top_lines_b)
    int_pt4 = line_intersect(l_right_m, l_right_b, top_lines_m, top_lines_b)

    court_coords[i] = splits + int_pt1 + int_pt2 + int_pt3 + int_pt4

    cv2.circle(img, int_pt1, 2, (255, 255, 255), -1)
    cv2.circle(img, int_pt2, 2, (255, 255, 0), -1)
    cv2.circle(img, int_pt3, 2, (255, 0, 255), -1)
    cv2.circle(img, int_pt4, 2, (0, 255, 255), -1)

    # show image
    cv2.imshow('frame diff ', img)
    cv2.waitKey(0)

pd.DataFrame(
    court_coords, 
    columns = ['chunk', 'frame', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 'x_4', 'y_4']
    ).to_csv(
        'assets/demo/court_coords.csv', 
        index = False
        )

print("--- %s seconds ---" % (time.time() - start_time))


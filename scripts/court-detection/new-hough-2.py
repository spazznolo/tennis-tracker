
# import the necessary packages
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import AgglomerativeClustering
import time
start_time = time.time()

width_error = 10
height_error = 10
kernel_size = 7
kernel_dilate = np.ones((1, 1), 'uint8')
kernel = np.ones((5, 5), 'uint8')

court_coords = np.zeros(shape=(316, 9))
game = "assets/game-frames/hard-m-2019-20-" 

i = 1684
while i < 1693:
    i+=1
    img = cv2.imread(game + str(i) + ".jpg")

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    edges = cv2.Canny(blur_gray, 0, 100, apertureSize = 3)
    dilate_img = cv2.dilate(edges, kernel_dilate, iterations=3)
    closing = cv2.morphologyEx(dilate_img, cv2.MORPH_CLOSE, kernel)

    # bottle-neck: too loose a filter on hough lines? takes 1x real-time
    lines = cv2.HoughLinesP(closing, 1, np.pi/180, 80, minLineLength=100, maxLineGap=20)
    lines_np = np.reshape(lines, (np.int32(lines.size/4), 4))
    lines_np = lines_np[(lines_np[:, 2] - lines_np[:, 0]) != 0]
    lines_np = np.c_[lines_np, (lines_np[:, 3] - lines_np[:, 1])/(lines_np[:, 2] - lines_np[:, 0])]
    lines_np = np.c_[lines_np, np.sqrt((lines_np[:, 3] - lines_np[:, 1])**2 + (lines_np[:, 2] - lines_np[:, 0])**2)]
    #lengthwise_lines = lines_np[(abs(lines_np[:, 4]) < 3.5) & (abs(lines_np[:, 4]) > 1.20) & (lines_np[:, 5] > 50)]
    lengthwise_lines = lines_np[(abs(lines_np[:, 4]) < 3.5) & (abs(lines_np[:, 4]) > 1.20)]

    if lengthwise_lines.size == 0:
        print('No lengthwise lines!')
        exit()

    # # bottle-neck: takes 0.175 real-time
    ward_length = AgglomerativeClustering(n_clusters = None, distance_threshold = 0.50, linkage = "ward").fit(abs(lengthwise_lines[:, 4].reshape(-1,1)))
    lengthwise_lines = np.c_[lengthwise_lines, ward_length.labels_]
    length_clusts = np.bincount(ward_length.labels_).argsort()[-2:]
    lengthwise_lines = lengthwise_lines[np.isin(ward_length.labels_, length_clusts)]
    x_max = np.max(lengthwise_lines[:,[0,2]].reshape(-1,1))
    x_min = np.amin(lengthwise_lines[:,[0,2]])

    # # bottle-neck: takes 0.100 real-time
    y_coords = lengthwise_lines[:, [1, 3]].reshape(-1,1)
    ward = AgglomerativeClustering(n_clusters = None, distance_threshold = 10, linkage = "ward").fit(y_coords)
    want_clusts = np.bincount(ward.labels_).argsort()[-2:]
    y_ext1 = np.median(y_coords[np.isin(ward.labels_, want_clusts[0])])
    y_ext2 = np.median(y_coords[np.isin(ward.labels_, want_clusts[1])])

    widthwise_lines = lines_np[(abs(lines_np[:, 4]) < 0.30)]
    widthwise_lines = np.c_[widthwise_lines, (widthwise_lines[:,1] - (widthwise_lines[:,4]*widthwise_lines[:,0]))]

    y_max = max(y_ext1, y_ext2)
    y_min = min(y_ext1, y_ext2)

    bottom_lines = widthwise_lines[np.where((np.amin(widthwise_lines[:,[1,3]], axis = 1) >= y_min - height_error) & (np.amin(widthwise_lines[:,[1,3]], axis = 1) <= y_min + height_error) & (np.amin(widthwise_lines[:,[0,2]], axis = 1) >= x_min - width_error) & (np.amax(widthwise_lines[:,[0,2]], axis = 1) <= x_max + width_error))[0]]
    top_lines = widthwise_lines[np.where((np.amax(widthwise_lines[:,[1,3]], axis = 1) >= y_max - height_error) & (np.amax(widthwise_lines[:,[1,3]], axis = 1) <= y_max + height_error) & (np.amin(widthwise_lines[:,[0,2]], axis = 1) >= x_min - width_error) & (np.amax(widthwise_lines[:,[0,2]], axis = 1) <= x_max + width_error))[0]]

    for x in range(0, len(top_lines)):
        cv2.line(img, 
            (int(top_lines[x,0]), int(top_lines[x,1])), 
                (int(top_lines[x,2]), int(top_lines[x,3])), (0,0,255), 1, cv2.LINE_AA)


    length_want = ward_length.labels_[np.isin(ward_length.labels_, length_clusts)][np.argmin(abs(lengthwise_lines[:,4]))]
    lengthwise_lines = lengthwise_lines[lengthwise_lines[:, 6] == length_want]
    lengthwise_lines = np.c_[lengthwise_lines, (lengthwise_lines[:,1] - (lengthwise_lines[:,4]*lengthwise_lines[:,0]))]


    for x in range(0, len(lengthwise_lines)):
        cv2.line(img, 
            (int(lengthwise_lines[x,0]), int(lengthwise_lines[x,1])), 
                (int(lengthwise_lines[x,2]), int(lengthwise_lines[x,3])), (0,0,255), 1, cv2.LINE_AA)

    cv2.imshow('frame diff ', img)
    cv2.waitKey(0)
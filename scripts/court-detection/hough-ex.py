
import os
import math
import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering

# find intersection of lines
def line_intersect(m1, b1, m2, b2):

    if m1 == m2:
        print ("These lines are parallel!!!")
        return None

    x = int((b2 - b1) / (m1 - m2))

    y = int(m1 * x + b1)

    return [x,y]

# method specs
width_error = 10
height_error = 10
kernel_size = 7
kernel_dilate = np.ones((1, 1), 'uint8')
kernel = np.ones((5, 5), 'uint8')

top_circles = np.zeros(shape=(2, 2))
bot_circles = np.zeros(shape=(2, 2))
left_circles = np.zeros(shape=(2, 2))
right_circles = np.zeros(shape=(2, 2))

# read and process image
path = 'assets/highlights/object-detection-train-frames/'

for filename in os.listdir(path):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(path, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        edges = cv2.Canny(blur_gray, 0, 100, apertureSize = 3)
        dilate_img = cv2.dilate(edges, kernel_dilate, iterations=5)
        closing = cv2.morphologyEx(dilate_img, cv2.MORPH_CLOSE, kernel)

        # find hough lines
        lines = cv2.HoughLinesP(closing, 2, np.pi/180, 100, minLineLength = 20, maxLineGap = 10)
        if lines is None: continue
        lines_np = np.reshape(lines, (np.int32(lines.size/4), 4))
        lines_np = lines_np[(lines_np[:, 2] - lines_np[:, 0]) != 0]
        lines_np = np.c_[lines_np, (lines_np[:, 3] - lines_np[:, 1])/(lines_np[:, 2] - lines_np[:, 0])]
        lines_np = np.c_[lines_np, np.sqrt((lines_np[:, 3] - lines_np[:, 1])**2 + (lines_np[:, 2] - lines_np[:, 0])**2)]

        # find lengthwise lines
        lengthwise_lines = lines_np[(abs(lines_np[:, 4]) < 3.5) & (abs(lines_np[:, 4]) > 1.20) & (lines_np[:, 5] > 100)]
        if lengthwise_lines[:, 4].size < 2: 
            print('No lengthwise lines!') 
            continue
        ward_length = AgglomerativeClustering(n_clusters = None, distance_threshold = 0.50, linkage = "ward").fit(abs(lengthwise_lines[:, 4].reshape(-1,1)))
        lengthwise_lines = np.c_[lengthwise_lines, ward_length.labels_]
        length_clusts = np.bincount(ward_length.labels_).argsort()[-2:]
        lengthwise_lines = lengthwise_lines[np.isin(ward_length.labels_, length_clusts)]

        # find extremities
        x_max = np.max(lengthwise_lines[:,[0,2]].reshape(-1,1))
        x_min = np.amin(lengthwise_lines[:,[0,2]])
        y_coords = lengthwise_lines[:, [1, 3]].reshape(-1,1)
        ward = AgglomerativeClustering(n_clusters = None, distance_threshold = 10, linkage = "ward").fit(y_coords)
        want_clusts = np.bincount(ward.labels_).argsort()[-2:]
        y_ext1 = np.median(y_coords[np.isin(ward.labels_, want_clusts[0])])
        y_ext2 = np.median(y_coords[np.isin(ward.labels_, want_clusts[1])])
        y_max = max(y_ext1, y_ext2)
        y_min = min(y_ext1, y_ext2)

        # find outer lengthwise lines
        length_want = ward_length.labels_[np.isin(ward_length.labels_, length_clusts)][np.argmin(abs(lengthwise_lines[:,4]))]
        lengthwise_lines = lengthwise_lines[lengthwise_lines[:, 6] == length_want]
        lengthwise_lines = np.c_[lengthwise_lines, (lengthwise_lines[:,1] - (lengthwise_lines[:,4]*lengthwise_lines[:,0]))]
        lengthwise_lines = np.c_[lengthwise_lines, ((np.max(lengthwise_lines[:,1]) - lengthwise_lines[:,5])/lengthwise_lines[:,4])]

        # subset left and right outer lengthwise lines
        l_left = lengthwise_lines[lengthwise_lines[:,4] < 0]
        l_right = lengthwise_lines[lengthwise_lines[:,4] > 0]

        # find top and bottom lines
        widthwise_lines = lines_np[(abs(lines_np[:, 4]) < 0.10)]
        widthwise_lines = np.c_[widthwise_lines, (widthwise_lines[:,1] - (widthwise_lines[:,4]*widthwise_lines[:,0]))]
        top_lines = widthwise_lines[np.where((np.amax(widthwise_lines[:,[1,3]], axis = 1) >= y_max - height_error) & (np.amax(widthwise_lines[:,[1,3]], axis = 1) <= y_max + height_error) & (np.amin(widthwise_lines[:,[0,2]], axis = 1) >= x_min - width_error) & (np.amax(widthwise_lines[:,[0,2]], axis = 1) <= x_max + width_error))[0]]

        if top_lines[:, 4].size < 2: 
            print('No lengthwise lines!') 
            continue

        ward = AgglomerativeClustering(n_clusters = None, distance_threshold = 5, linkage = "ward").fit(top_lines[:, [4, 6]])
        top_lines = top_lines[np.where(ward.labels_ == np.bincount(ward.labels_, weights = top_lines[:, 5]).argmax())]

        bottom_lines = widthwise_lines[
            np.where((np.amin(widthwise_lines[:,[1,3]], axis = 1) >= np.min(top_lines[:,[1,3]]) - 150) & 
                (np.amin(widthwise_lines[:,[1,3]], axis = 1) <= np.min(top_lines[:,[1,3]]) - height_error) & 
                    (np.amin(widthwise_lines[:,[0,2]], axis = 1) >= np.min(top_lines[:,[0,2]]) - width_error) & 
                        (np.amax(widthwise_lines[:,[0,2]], axis = 1) <= np.max(top_lines[:,[0,2]]) + width_error))[0]]

        # if no lines found for any border, stop
        if top_lines.size == 0 or bottom_lines.size == 0 or l_left.size == 0 or l_right.size == 0: 
            print('No Lengthwise lines!')
            continue


        # take median of top widthwise lines
        y_max = np.median(l_left[:,1])
        y_min = np.median(l_left[:,3])
        
        for x in range(1, 3):
            splint_ = int(y_min + (x*(y_max - y_min)/3))
            splint_sub = np.where((l_left[:,1] > splint_) & (l_left[:,3] < splint_))[0]
            x_level = int(np.median((splint_ - l_left[splint_sub, 7])/l_left[splint_sub, 4]))
            left_circles[x-1,] = [x_level, splint_]

        left_lines_m = (left_circles[1, 1] - left_circles[0, 1])/(left_circles[1, 0] - left_circles[0, 0])
        left_lines_b = left_circles[1, 1] - (left_lines_m*left_circles[1, 0])

        point_test_y_1 = int(y_min + 10)
        point_test_x_1 = int((point_test_y_1 - left_lines_b)/left_lines_m)

        point_test_y_2 = int(y_max - 60)
        point_test_x_2 = int((point_test_y_2 - left_lines_b)/left_lines_m)

        # take median of top widthwise lines
        y_max = np.median(l_right[:,3])
        y_min = np.median(l_right[:,1])

        for x in range(1, 3):
            splint_ = int(y_min + (x*(y_max - y_min)/3))
            splint_sub = np.where((l_right[:,1] < splint_) & (l_right[:,3] > splint_))[0]
            x_level = int(np.median((splint_ - l_right[splint_sub, 7])/l_right[splint_sub, 4]))
            right_circles[x-1,] = [x_level, splint_]

        right_lines_m = (right_circles[1, 1] - right_circles[0, 1])/(right_circles[1, 0] - right_circles[0, 0])
        right_lines_b = right_circles[1, 1] - (right_lines_m*right_circles[1, 0])

        # take median of top widthwise lines
        x_max = np.amax(bottom_lines[:,[0,2]])
        x_min = np.median(bottom_lines[:,0])

        for x in range(1, 3):
            splint_ = int(x_min + (x*(x_max - x_min)/3))
            splint_sub = np.where((bottom_lines[:,0] < splint_) & (bottom_lines[:,2] > splint_))[0]
            y_level = np.median(bottom_lines[splint_sub, 4]*splint_ + bottom_lines[splint_sub, 6])
            if np.isnan(y_level): continue
            y_level = int(y_level)
            bot_circles[x-1,] = [splint_, y_level]

        bot_lines_m = (bot_circles[1, 1] - bot_circles[0, 1])/(bot_circles[1, 0] - bot_circles[0, 0])
        bot_lines_b = bot_circles[1, 1] - (bot_lines_m*bot_circles[1, 0])

        point_test_x_1 = int(x_min + 5)
        point_test_y_1 = int(point_test_x_1*bot_lines_m + bot_lines_b)

        point_test_x_2 = int(x_max - 10)
        point_test_y_2 = int(point_test_x_2*bot_lines_m + bot_lines_b)

        # take median of top widthwise lines
        x_max = np.median(top_lines[:,2])
        x_min = np.median(top_lines[:,0])

        for x in range(1, 3):
            splint_ = int(x_min + (x*(x_max - x_min)/3))
            splint_sub = np.where((top_lines[:,0] < splint_) & (top_lines[:,2] > splint_))[0]
            y_level = np.median(top_lines[splint_sub, 4]*splint_ + top_lines[splint_sub, 6])
            if np.isnan(y_level): continue
            y_level = int(y_level)
            top_circles[x-1,] = [splint_, y_level]

        top_lines_m = (top_circles[1, 1] - top_circles[0, 1])/(top_circles[1, 0] - top_circles[0, 0])
        top_lines_b = top_circles[1, 1] - (top_lines_m*top_circles[1, 0])

        point_test_x_1 = int(x_min + 35)
        point_test_y_1 = int(point_test_x_1*top_lines_m + top_lines_b)

        point_test_x_2 = int(x_max)
        point_test_y_2 = int(point_test_x_2*top_lines_m + top_lines_b)

        # find interesction of lines
        int_pt1 = line_intersect(left_lines_m, left_lines_b, bot_lines_m, bot_lines_b)
        int_pt2 = line_intersect(right_lines_m, right_lines_b, bot_lines_m, bot_lines_b)
        int_pt3 = line_intersect(left_lines_m, left_lines_b, top_lines_m, top_lines_b)
        int_pt4 = line_intersect(right_lines_m, right_lines_b, top_lines_m, top_lines_b)

        ok = closing[(point_test_y_2-10):(point_test_y_2+10), (point_test_x_2-10):(point_test_x_2+10)]
        ok = np.transpose(ok)
        ok1 = np.argwhere(ok!=0)
        ok2 = np.split(ok1[:,1], np.unique(ok1[:,0], return_index=True)[1][1:])
        ok3 = np.array(ok2, dtype=object)
        ok4 = np.array([np.mean(i) for i in ok3])
        if len(ok4) % 2 == 0: ok4 = ok4[:(len(ok4) - 1)]
        ok5 = np.argwhere(ok4 == np.median(ok4))[0]
        ok6 = int(ok4[ok5])

        ok7 = closing[(point_test_y_1-10):(point_test_y_1+10), (point_test_x_1-10):(point_test_x_1+10)]
        ok7 = np.transpose(ok7)
        ok8 = np.argwhere(ok7!=0)
        ok9 = np.split(ok8[:,1], np.unique(ok8[:,0], return_index=True)[1][1:])
        ok10 = np.array(ok9, dtype=object)
        ok11 = np.array([np.mean(i) for i in ok10])
        if len(ok11) % 2 == 0: ok11 = ok11[:(len(ok11) - 1)]
        ok12 = np.argwhere(ok11 == np.median(ok11))[0]
        ok13 = int(ok11[ok12]+0.1)

        print(((point_test_y_2 + 1 - 10 + int(ok6)) - (point_test_y_1 + 1 - 10 + int(ok13)))/((point_test_x_2 + 1 - 10 + int(ok5)) - (point_test_x_1 + 1 - 10 + int(ok12))))
        # draw intersections
        cv2.circle(img, (point_test_x_2 + 1 - 10 + int(ok5), point_test_y_2 + 1 - 10 + int(ok6)), 2, (255, 255, 0), -1)
        cv2.circle(img, (point_test_x_1 + 1 - 10 + int(ok12), point_test_y_1 + 1 - 10 + int(ok13)), 2, (255, 255, 0), -1)

        cv2.line(img, (point_test_x_2 + 1 - 10 + int(ok5), point_test_y_2 + 1 - 10 + int(ok6)), 
        (point_test_x_1 + 1 - 10 + int(ok12), point_test_y_1 + 1 - 10 + int(ok13)), (0,255,255), 1, cv2.LINE_AA)
        # cv2.line(img, (0, int(top_lines_b)), (854, int((854*top_lines_m) + top_lines_b)), (0,255,255), 1, cv2.LINE_AA)
        # cv2.line(img, (int(-left_lines_b/left_lines_m), 0), (int((480 - left_lines_b)/left_lines_m), 480), (255,255,255), 1, cv2.LINE_AA)
        # cv2.line(img, (int(-right_lines_b/right_lines_m), 0), (int((480 - right_lines_b)/right_lines_m), 480), (255,0,255), 1, cv2.LINE_AA)

        # show image
        cv2.imshow('frame diff ', img)
        cv2.waitKey(0)
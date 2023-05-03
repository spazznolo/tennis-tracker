
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering
import time

start_time = time.time()

# find intersection of lines
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = int(det(d, xdiff) / div)
    y = int(det(d, ydiff) / div)
    
    return x, y

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

court_coords = np.zeros(shape=(4400, 9))
start_time1 = 0
# read and process image
i = 4000
j = -1
while i < 4500:

    print("--- %s new seconds ---" % (time.time() - start_time1))
    start_time1 = time.time()
    i += 1
    j += 1
    if i % 25 == 0: print(i)
    img = cv2.imread('assets/demo/test/test-' + str(i) + '.jpg')
    #img = cv2.imread('assets/train/on/hard-m-2019-20-' + str(i) + '.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    edges = cv2.Canny(blur_gray, 0, 100, apertureSize = 3)
    dilate_img = cv2.dilate(edges, kernel_dilate, iterations=2)
    closing = cv2.morphologyEx(dilate_img, cv2.MORPH_CLOSE, kernel)

    print("--- %s wow seconds ---" % (time.time() - start_time))
    start_time = time.time()

    # find hough lines
    lines = cv2.HoughLinesP(closing, 2, np.pi/180, 100, minLineLength = 20, maxLineGap = 10)
    if lines is None: continue

    print("--- %s seconds4 ---" % (time.time() - start_time))
    start_time = time.time()

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

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    # find top and bottom lines
    widthwise_lines = lines_np[(abs(lines_np[:, 4]) < 0.10)]
    widthwise_lines = np.c_[widthwise_lines, (widthwise_lines[:,1] - (widthwise_lines[:,4]*widthwise_lines[:,0]))]

    top_lines = widthwise_lines[np.where((np.amax(widthwise_lines[:,[1,3]], axis = 1) >= y_max - height_error) & (np.amax(widthwise_lines[:,[1,3]], axis = 1) <= y_max + height_error) & (np.amin(widthwise_lines[:,[0,2]], axis = 1) >= x_min - width_error) & (np.amax(widthwise_lines[:,[0,2]], axis = 1) <= x_max + width_error))[0]]

    if top_lines[:, 4].size < 2: 
        print('No widthwise lines!') 
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

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    # take median of top widthwise lines
    if len(l_left) % 2 == 0: l_left = l_left[0:(len(l_left)-1),]
    y_max = np.median(l_left[:,3])
    y_min = np.median(l_left[:,1])

    left_y_1 = int(y_min)
    left_x_1 = int(l_left[np.argwhere(l_left[:,1] == y_min)[0], 0])

    left_y_2 = int(y_max)
    left_x_2 = int(l_left[np.argwhere(l_left[:,3] == y_max)[0], 2])

    # take median of top widthwise lines
    if len(l_right) % 2 == 0: l_right = l_right[0:(len(l_right)-1),]
    y_max = np.median(l_right[:,3])
    y_min = np.median(l_right[:,1])

    right_y_1 = int(y_min)
    right_x_1 = int(l_right[np.argwhere(l_right[:,1] == y_min)[0], 0])

    right_y_2 = int(y_max)
    right_x_2 = int(l_right[np.argwhere(l_right[:,3] == y_max)[0], 2])

    # take median of top widthwise lines
    if len(bottom_lines) % 2 == 0: bottom_lines = bottom_lines[0:(len(bottom_lines)-1),]
    x_max = np.amax(bottom_lines[:,2])
    x_min = np.median(bottom_lines[:,0])

    bottom_x_1 = int(x_min + 5)
    bottom_y_1 = int(bottom_lines[np.argwhere(bottom_lines[:,0] == x_min)[0], 1])

    bottom_x_2 = int(x_max - 10)
    bottom_y_2 = int(bottom_lines[np.argwhere(bottom_lines[:,2] == x_max)[0], 3])

    # take median of top widthwise lines
    x_max = np.max(top_lines[:,2])
    x_min = np.min(top_lines[:,0])

    top_x_1 = int(x_min + 5)
    top_y_1 = int(top_lines[np.argwhere(top_lines[:,0] == x_min)[0], 1])

    top_x_2 = int(x_max - 10)
    top_y_2 = int(top_lines[np.argwhere(top_lines[:,2] == x_max)[0], 1])

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    ok = closing[(bottom_y_2-10):(bottom_y_2+10), (bottom_x_2-10):(bottom_x_2+10)]
    ok = np.transpose(ok)
    ok1 = np.argwhere(ok!=0)
    if len(ok1) < 1: continue
    ok2 = np.split(ok1[:,1], np.unique(ok1[:,0], return_index=True)[1][1:])
    ok3 = np.array(ok2, dtype=object)
    ok4 = np.array([np.mean(i) for i in ok3])
    if len(ok4) % 2 == 0: ok4 = ok4[:(len(ok4) - 1)]
    ok5 = np.argwhere(ok4 == np.median(ok4))[0]
    ok6 = int(ok4[ok5])

    ok7 = closing[(bottom_y_1-10):(bottom_y_1+10), (bottom_x_1-10):(bottom_x_1+10)]
    ok7 = np.transpose(ok7)
    ok8 = np.argwhere(ok7!=0)
    if len(ok8) < 1: continue
    ok9 = np.split(ok8[:,1], np.unique(ok8[:,0], return_index=True)[1][1:])
    ok10 = np.array(ok9, dtype=object)
    ok11 = np.array([np.mean(i) for i in ok10])
    if len(ok11) % 2 == 0: ok11 = ok11[:(len(ok11) - 1)]
    ok12 = np.argwhere(ok11 == np.median(ok11))[0]
    ok13 = int(ok11[ok12]+0.1)

    the_slope = ((bottom_y_2 + 1 - 10 + int(ok6)) - (bottom_y_1 + 1 - 10 + int(ok13)))/((bottom_x_2 + 1 - 10 + int(ok5)) - (bottom_x_1 + 1 - 10 + int(ok12)))
    the_b = the_slope*(bottom_x_1 + 1 - 10 + int(ok12)) + (bottom_y_1 + 1 - 10 + int(ok13)) 

    bottom_line = [
        [bottom_x_2 + 1 - 10 + ok5, bottom_y_2 + 1 - 10 + ok6],
        [bottom_x_1 + 1 - 10 + ok12, bottom_y_1 + 1 - 10 + ok13]]

    cv2.line(
        img, 
        (bottom_x_2 + 1 - 10 + int(ok5), bottom_y_2 + 1 - 10 + int(ok6)), 
        (bottom_x_1 + 1 - 10 + int(ok12), bottom_y_1 + 1 - 10 + int(ok13)), 
        (0,255,255), 1, cv2.LINE_AA)

    cv2.line(
        img, 
        (bottom_x_2 + 1 - 10 + int(ok5), bottom_y_2 + 1 - 10 + int(ok6)), 
        (bottom_x_1 + 1 - 10 + int(ok12), bottom_y_1 + 1 - 10 + int(ok13)), 
        (0,255,255), 1, cv2.LINE_AA)

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    ok = closing[(top_y_2-10):(top_y_2+10), (top_x_2-10):(top_x_2+10)]
    ok = np.transpose(ok)
    ok1 = np.argwhere(ok!=0)
    if len(ok1) < 1: continue
    ok2 = np.split(ok1[:,1], np.unique(ok1[:,0], return_index=True)[1][1:])
    ok3 = np.array(ok2, dtype=object)
    ok4 = np.array([np.mean(i) for i in ok3])
    if len(ok4) % 2 == 0: ok4 = ok4[:(len(ok4) - 1)]
    ok5 = np.argwhere(ok4 == np.median(ok4))[0]
    ok6 = ok4[ok5]

    ok7 = closing[(top_y_1-10):(top_y_1+10), (top_x_1-10):(top_x_1+10)]
    ok7 = np.transpose(ok7)
    ok8 = np.argwhere(ok7!=0)
    if len(ok8) < 1: continue
    ok9 = np.split(ok8[:,1], np.unique(ok8[:,0], return_index=True)[1][1:])
    ok10 = np.array(ok9, dtype=object)
    ok11 = np.array([np.mean(i) for i in ok10])
    if len(ok11) % 2 == 0: ok11 = ok11[:(len(ok11) - 1)]
    ok12 = np.argwhere(ok11 == np.median(ok11))[0]
    ok13 = ok11[ok12]+0.1

    top_line = [
        [top_x_2 + 1 - 10 + ok5, top_y_2 + 1 - 10 + ok6],
        [top_x_1 + 1 - 10 + ok12, top_y_1 + 1 - 10 + ok13]]

    cv2.line(
        img, 
        (top_x_2 + 1 - 10 + int(ok5), top_y_2 + 1 - 10 + int(ok6)), 
        (top_x_1 + 1 - 10 + int(ok12), top_y_1 + 1 - 10 + int(ok13)), 
        (0,255,255), 1, cv2.LINE_AA)

    the_slope = ((top_y_2 + 1 - 10 + int(ok6)) - (top_y_1 + 1 - 10 + int(ok13)))/((top_x_2 + 1 - 10 + int(ok5)) - (top_x_1 + 1 - 10 + int(ok12)))
    the_b = the_slope*(top_x_1 + 1 - 10 + int(ok12)) + (top_y_1 + 1 - 10 + int(ok13)) 

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    ok = closing[(right_y_2-10):(right_y_2+10), (right_x_2-10):(right_x_2+10)]
    ok1 = np.argwhere(ok!=0)
    if len(ok1) < 1: continue
    ok2 = np.split(ok1[:,1], np.unique(ok1[:,0], return_index=True)[1][1:])
    ok3 = np.array(ok2, dtype=object)
    ok4 = np.array([np.mean(i) for i in ok3])
    if len(ok4) % 2 == 0: ok4 = ok4[:(len(ok4) - 1)]
    ok5 = np.argwhere(ok4 == np.median(ok4))[0]
    ok6 = ok4[ok5]
    ok61 = np.unique(ok1[:,0])[ok5]

    ok7 = closing[(right_y_1-10):(right_y_1+10), (right_x_1-10):(right_x_1+10)]
    ok8 = np.argwhere(ok7!=0)
    if len(ok8) < 1: continue
    ok9 = np.split(ok8[:,1], np.unique(ok8[:,0], return_index=True)[1][1:])
    ok10 = np.array(ok9, dtype=object)
    ok11 = np.array([np.mean(i) for i in ok10])
    if len(ok11) % 2 == 0: ok11 = ok11[:(len(ok11) - 1)]
    ok12 = np.argwhere(ok11 == np.median(ok11))[0]
    ok13 = ok11[ok12]+0.1
    ok14 = np.unique(ok8[:,0])[ok12]

    right_line = [
        [right_x_2 + 1 - 10 + ok6, right_y_2 + 1 - 10 + ok61],
        [right_x_1 + 1 - 10 + ok13, right_y_1 + 1 - 10 + ok14]]

    cv2.line(
        img, 
        (right_x_2 + 1 - 10 + int(ok6), right_y_2 + 1 - 10 + int(ok61)), 
        (right_x_1 + 1 - 10 + int(ok13), right_y_1 + 1 - 10 + int(ok14)), 
        (0,255,255), 1, cv2.LINE_AA)

    the_slope = ((right_y_2 + 1 - 10 + int(ok6)) - (right_y_1 + 1 - 10 + int(ok13)))/((right_x_2 + 1 - 10 + int(ok5)) - (right_x_1 + 1 - 10 + int(ok12)))
    the_b = the_slope*(right_x_1 + 1 - 10 + int(ok13)) + (right_y_1 + 1 - 10 + int(ok14)) 

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    ok = closing[(left_y_2-10):(left_y_2+10), (left_x_2-15):(left_x_2+15)]
    ok1 = np.argwhere(ok!=0)
    if len(ok1) < 1: continue
    ok2 = np.split(ok1[:,1], np.unique(ok1[:,0], return_index=True)[1][1:])
    ok3 = np.array(ok2, dtype=object)
    ok4 = np.array([np.mean(i) for i in ok3])
    if len(ok4) % 2 == 0: ok4 = ok4[:(len(ok4) - 1)]
    ok5 = np.argwhere(ok4 == np.median(ok4))[0]
    ok6 = ok4[ok5]+0.1
    ok61 = np.unique(ok1[:,0])[ok5]

    ok7 = closing[(left_y_1-10):(left_y_1+10), (left_x_1-10):(left_x_1+10)]
    ok8 = np.argwhere(ok7!=0)
    if len(ok8) < 1: continue
    ok9 = np.split(ok8[:,1], np.unique(ok8[:,0], return_index=True)[1][1:])
    ok10 = np.array(ok9, dtype=object)
    ok11 = np.array([np.mean(i) for i in ok10])
    if len(ok11) % 2 == 0: ok11 = ok11[:(len(ok11) - 1)]
    ok12 = np.argwhere(ok11 == np.median(ok11))[0]
    ok13 = int(ok11[ok12])
    ok14 = np.unique(ok8[:,0])[ok12]

    left_line = [
        [left_x_2 + 1 - 15 + ok6, left_y_2 + 1 - 10 + ok61],
        [left_x_1 + 1 - 10 + ok13, left_y_1 + 1 - 10 + ok14]]

    cv2.circle(img, (int(left_x_2 + 1 - 15 + ok6), int(left_y_2 + 1 - 10 + ok61)), 4, (255, 0, 0), -1)    

    cv2.line(
        img, 
        (left_x_2 + 1 - 15 + int(ok6), left_y_2 + 1 - 10 + int(ok61)), 
        (left_x_1 + 1 - 10 + int(ok13), left_y_1 + 1 - 10 + int(ok14)), 
        (0,255,255), 1, cv2.LINE_AA)

    the_slope = ((left_y_2 + 1 - 10 + int(ok6)) - (left_y_1 + 1 - 10 + int(ok13)))/((left_x_2 + 1 - 10 + int(ok5)) - (left_x_1 + 1 - 10 + int(ok12)))
    the_b = the_slope*(left_x_1 + 1 - 10 + int(ok12)) + (left_y_1 + 1 - 10 + int(ok13)) 

    print("--- %s seconds1 ---" % (time.time() - start_time))
    start_time = time.time()

    int1 = line_intersection(left_line, bottom_line)
    int2 = line_intersection(left_line, top_line)
    int3 = line_intersection(right_line, bottom_line)
    int4 = line_intersection(right_line, top_line)

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    cv2.circle(img, int1, 2, (255, 255, 0), -1)    
    cv2.circle(img, int2, 2, (255, 255, 0), -1)
    cv2.circle(img, int3, 2, (255, 255, 0), -1)
    cv2.circle(img, int4, 2, (255, 255, 0), -1)

    court_coords[j] = int2 + int1 + int4 + int3 + (i,)

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    # show image
    cv2.imshow('frame diff ', img)
    cv2.waitKey(0)

pd.DataFrame(
    court_coords, 
    columns = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 'x_4', 'y_4', 'frame']
    ).to_csv(
        'assets/demo/court_coords.csv', 
        index = False
        )

print("--- %s seconds ---" % (time.time() - start_time))
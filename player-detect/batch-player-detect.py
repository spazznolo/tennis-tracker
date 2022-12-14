
import cv2
import numpy as np
import pandas as pd
import glob
import os
import time
start_time = time.time()

kernel_size = 1
kernel_dilate = np.ones((3, 3), 'uint8')
kernel = np.ones((3, 3), 'uint8')
kernel_open = np.ones((3, 3), 'uint8')

images = sorted(glob.glob("assets/demo/all-frames/*.jpg"))
loc_close_df = np.zeros(shape=(len(images), 4))
loc_far_df = np.zeros(shape=(len(images), 4))

i = -1

while i < len(images) - 1:

    i+= 1 

    print(i/len(images))

    if i == 0: continue

    img_ = images[i]
    img_1 = images[i-1]
    splits = os.path.splitext(os.path.basename(img_))[0].split('-')
    splits_prev = os.path.splitext(os.path.basename(img_1))[0].split('-')
    print(splits)
    if splits[3] != splits_prev[3]: continue
    if int(splits[4]) != int(splits_prev[4]) + 1: continue

    current_frame = cv2.imread(img_)
    previous_frame = cv2.imread(img_1)

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    

    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
    dilate_img = cv2.dilate(frame_diff, kernel_dilate, iterations=3)
    closing = cv2.morphologyEx(dilate_img, cv2.MORPH_OPEN, kernel_open)
    median_blur = cv2.medianBlur(closing, 1)
    thresh = cv2.threshold(median_blur, 10, 120, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    loc_close_df[i] = splits[3:] + [0, 0]
    loc_far_df[i] = splits[3:] + [0, 0]

    close_cnts = []
    far_cnts = []

    for c in cnts:

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        if extLeft[0] > 95 and extRight[0] < 750 and extBot[1] > 250 and cv2.contourArea(c) < 7000 and cv2.contourArea(c) > 1500 and (abs(extLeft[0] - extRight[0]) < 50 + abs(extTop[1] - extBot[1])):
            close_cnts.append(c)
            break

    for c in cnts:

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        if extLeft[0] > 190 and extRight[0] < 650 and extBot[1] > 100 and extBot[1] < 250 and cv2.contourArea(c) < 3000 and cv2.contourArea(c) > 800 and (abs(extLeft[0] - extRight[0]) < 50 + abs(extTop[1] - extBot[1])):
            far_cnts.append(c)
            break


    for c in close_cnts[0:1]:

        x_loc_close = int((tuple(c[c[:, :, 0].argmax()][0])[0] + tuple(c[c[:, :, 0].argmin()][0])[0])/2)
        y_loc_close = tuple(c[c[:, :, 1].argmax()][0])[1]
        loc_close_df[i] = splits[3:] + [x_loc_close, y_loc_close]

    for c in far_cnts[0:1]:

        x_loc_far = int((tuple(c[c[:, :, 0].argmax()][0])[0] + tuple(c[c[:, :, 0].argmin()][0])[0])/2)
        y_loc_far = tuple(c[c[:, :, 1].argmax()][0])[1]
        loc_far_df[i] = splits[3:] + [x_loc_far, y_loc_far]



pd.DataFrame(loc_close_df, columns = ['chunk', 'frame', 'x', 'y']).to_csv('assets/demo/close_location.csv', index = False)
pd.DataFrame(loc_far_df, columns = ['chunk', 'frame', 'x', 'y']).to_csv('assets/demo/far_location.csv', index = False)

print("--- %s seconds ---" % (time.time() - start_time))
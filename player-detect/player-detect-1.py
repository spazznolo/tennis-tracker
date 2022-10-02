
import numpy as np
from matplotlib import pyplot as plt
import cv2

kernel_size = 1
kernel_dilate = np.ones((5, 5), 'uint8')
kernel = np.ones((3, 3), 'uint8')
kernel_open = np.ones((3, 3), 'uint8')
fgbg = cv2.createBackgroundSubtractorMOG2(history=5, detectShadows=True)

old = "assets/game-frames/hard-m-2019-128-" # 800-850

i = 400
while i < 430:
    
    current_frame = cv2.imread("assets/game-frames/hard-w-2022-67-" + str(i) + ".jpg")
    previous_frame = cv2.imread("assets/game-frames/hard-w-2022-67-" + str(i-1) + ".jpg")
    previous_frame_1 = cv2.imread("assets/game-frames/hard-w-2022-67-" + str(i-2) + ".jpg")

    fgmask = fgbg.apply(previous_frame_1)
    fgmask = fgbg.apply(previous_frame)
    fgmask = fgbg.apply(current_frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    
    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

    # current_blur_gray = cv2.GaussianBlur(current_frame_gray,(kernel_size, kernel_size),0)
    # previous_blur_gray = cv2.GaussianBlur(previous_frame_gray,(kernel_size, kernel_size),0)
    # frame_diff = cv2.absdiff(current_blur_gray, previous_blur_gray)
    #test = cv2.adaptiveThreshold(frame_diff, 100, blockSize=9, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, C=5) 
    
    dilate_img = cv2.dilate(frame_diff, kernel_dilate, iterations=3)
    closing = cv2.morphologyEx(dilate_img, cv2.MORPH_OPEN, kernel_open)
    median_blur = cv2.medianBlur(closing, 3)
    
    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = 1
    params.blobColor = 255

    # params.minThreshold = 20
    # params.maxThreshold = 50

    params.filterByArea = True
    params.minArea = 10

    params.filterByCircularity = True
    params.minCircularity = 0.20

    params.filterByInertia = True
    params.minInertiaRatio = 0.08

    params.filterByConvexity = True
    params.minConvexity = 0.2

    #params.minDistBetweenBlobs = 100

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(closing)
    im_with_keypoints = cv2.drawKeypoints(closing, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    opening = cv2.morphologyEx(dilate_img, cv2.MORPH_OPEN, kernel)

    thresh = cv2.threshold(median_blur, 25, 110, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts = cnts[0:2]

    for c in cnts:
        # Highlight largest contour
        cv2.drawContours(thresh, [c], -1, (255,0,0), 3)

    cv2.imshow('frame diff ', thresh) 
    cv2.waitKey(0) 
    i = i+1


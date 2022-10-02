
import numpy as np
from matplotlib import pyplot as plt
import cv2

kernel_size = 1
kernel_dilate = np.ones((3, 3), 'uint8')
kernel = np.ones((3, 3), 'uint8')
kernel_open = np.ones((3, 3), 'uint8')

kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()

old = "assets/game-frames/hard-m-2019-128-" # 800-850

i = 700
while i < 800:
    
    current_frame = cv2.imread("assets/game-frames/hard-m-2019-103-" + str(i - 1) + ".jpg")
    previous_frame = cv2.imread("assets/game-frames/hard-m-2019-103-" + str(i) + ".jpg")

    fgmask = fgbg.apply(previous_frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    
    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

    # current_blur_gray = cv2.GaussianBlur(current_frame_gray,(kernel_size, kernel_size),0)
    # previous_blur_gray = cv2.GaussianBlur(previous_frame_gray,(kernel_size, kernel_size),0)
    # frame_diff = cv2.absdiff(current_blur_gray, previous_blur_gray)
    # test = cv2.adaptiveThreshold(frame_diff, 100, blockSize=9, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, C=5) 
    
    dilate_img = cv2.dilate(frame_diff, kernel_dilate, iterations=3)
    opening = cv2.morphologyEx(dilate_img, cv2.MORPH_OPEN, kernel_open)
    median_blur = cv2.medianBlur(opening, 5)

    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = 1
    params.blobColor = 255

    params.minThreshold = 75
    params.maxThreshold = 150

    params.filterByArea = True
    params.minArea = 80
    params.maxArea = 150

    params.filterByCircularity = True
    params.minCircularity = 0.01

    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    params.filterByConvexity = True
    params.minConvexity = 0.05

    #params.minDistBetweenBlobs = 100

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(median_blur)
    im_with_keypoints = cv2.drawKeypoints(median_blur, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # contours = cv2.findContours(dilate_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    # result = median_blur.copy()

    # for cntr in contours[0]:
    #     x,y,w,h = cv2.boundingRect(cntr)
    #     cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 255), 2)

    cv2.imshow('frame diff ', im_with_keypoints) 
    cv2.waitKey(0) 
    i = i+1


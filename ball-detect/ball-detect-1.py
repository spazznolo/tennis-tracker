
import numpy as np
from matplotlib import pyplot as plt
import cv2

kernel_size = 1
kernel_dilate = np.ones((3, 3), 'uint8')
kernel = np.ones((3, 3), 'uint8')
kernel_open = np.ones((3, 3), 'uint8')

kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()

game = "assets/game-frames/hard-m-2019-128-" # 800-850

i = 800
while i < 890:
    
    current_frame = cv2.imread(str(game) + str(i) + ".jpg")
    previous_frame = cv2.imread(str(game) + str(i-1) + ".jpg")
    previous_frame_1 = cv2.imread(str(game) + str(i-2) + ".jpg")

    fgmask = fgbg.apply(previous_frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    
    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

    dilate_img = cv2.dilate(frame_diff, kernel_dilate, iterations=3)
    opening = cv2.morphologyEx(dilate_img, cv2.MORPH_OPEN, kernel_open)
    median_blur = cv2.medianBlur(opening, 1)

    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = 1
    params.blobColor = 255

    params.minThreshold = 25
    params.maxThreshold = 150

    params.filterByArea = True
    params.minArea = 80
    params.maxArea = 200

    # params.filterByCircularity = True
    # params.minCircularity = 0.01

    # params.filterByInertia = True
    # params.minInertiaRatio = 0.3

    # params.filterByConvexity = True
    # params.minConvexity = 0.05

    # params.minDistBetweenBlobs = 100

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(median_blur)
    im_with_keypoints = cv2.drawKeypoints(median_blur, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('frame diff ', im_with_keypoints) 
    cv2.waitKey(0) 
    i = i+1


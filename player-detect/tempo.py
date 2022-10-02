
import numpy as np
from matplotlib import pyplot as plt
import cv2

kernel_size = 1
kernel = np.ones((5, 5), 'uint8')

fgbg = cv2.createBackgroundSubtractorMOG2()
back_frame = cv2.imread("assets/game-frames/hard-m-2019-85-" + str(700) + ".jpg")

i = 800
while i < 810:
    
    current_frame = cv2.imread("assets/game-frames/hard-m-2019-128-" + str(i - 1) + ".jpg")
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    test = cv2.adaptiveThreshold(current_frame_gray, 1, blockSize=3, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, C=1) 
    #dilate_img = cv2.dilate(test, kernel, iterations=3)

    cv2.imshow('frame diff ', test) 
    cv2.waitKey(0) 
    i = i+1


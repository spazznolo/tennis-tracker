
import numpy as np
import cv2

kernel_size = 1
kernel_dilate = np.ones((3, 3), 'uint8')
kernel = np.ones((3, 3), 'uint8')
kernel_open = np.ones((3, 3), 'uint8')
fgbg = cv2.createBackgroundSubtractorMOG2(history=5, detectShadows=True)

game = "assets/game-frames/hard-w-2022-70-" # 800-850

i = 300
while i < 500:
    
    current_frame = cv2.imread(str(game) + str(i) + ".jpg")
    previous_frame = cv2.imread(str(game) + str(i-1) + ".jpg")
    previous_frame_1 = cv2.imread(str(game) + str(i-2) + ".jpg")

    fgmask = fgbg.apply(previous_frame_1)
    fgmask = fgbg.apply(previous_frame)
    fgmask = fgbg.apply(current_frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    

    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
    dilate_img = cv2.dilate(frame_diff, kernel_dilate, iterations=3)
    closing = cv2.morphologyEx(dilate_img, cv2.MORPH_OPEN, kernel_open)
    median_blur = cv2.medianBlur(closing, 1)
    blur = cv2.GaussianBlur(frame_diff,(5,5),0)
    thresh = cv2.threshold(median_blur, 10, 120, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts_sizes = [cv2.contourArea(contour) for contour in cnts]

    close_cnts = []
    far_cnts = []

    for c in cnts:

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        if cv2.contourArea(c) < 7000 and cv2.contourArea(c) > 1500 and (abs(extLeft[0] - extRight[0]) < 50 + abs(extTop[1] - extBot[1])):
            close_cnts.append(c)
        
        if extLeft[0] > 150 and extRight[0] < 700 and extBot[1] < 250 and cv2.contourArea(c) < 3000 and cv2.contourArea(c) > 800 and (abs(extLeft[0] - extRight[0]) < 50 + abs(extTop[1] - extBot[1])):
            far_cnts.append(c)

    close_cnts = close_cnts[0:1]
    far_cnts = far_cnts[0:1]

    for c in close_cnts:
        t = ()
        t += (int((tuple(c[c[:, :, 0].argmax()][0])[0] + tuple(c[c[:, :, 0].argmin()][0])[0])/2), )
        t += (tuple(c[c[:, :, 1].argmax()][0])[1],)
        cv2.circle(current_frame, t, 10, (0, 255, 0), -1)
        cv2.drawContours(current_frame, [c], -1, (255,255,0), 3)

    for c in far_cnts:
        t = ()
        t += (int((tuple(c[c[:, :, 0].argmax()][0])[0] + tuple(c[c[:, :, 0].argmin()][0])[0])/2), )
        t += (tuple(c[c[:, :, 1].argmax()][0])[1],)
        print(t)
        cv2.circle(current_frame, t, 10, (0, 255, 0), -1)
        cv2.drawContours(current_frame, [c], -1, (255,0,255), 3)

    cv2.imshow('frame diff ', current_frame) 
    cv2.waitKey(0) 
    i = i+1


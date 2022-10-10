import cv2

backSub_mog = cv2.createBackgroundSubtractorMOG2(history = 50)
cap_v = cv2.VideoCapture("assets/game-frames/hard-w-2020-55-%01d.jpg")

while True:
    ret, frame = cap_v.read()
    if frame is None:
        break

    fgMask = backSub_mog.apply(frame, learningRate = 0.005)

    cv2.imshow('Input Frame', frame)
    cv2.imshow('Foreground Mask', fgMask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
cv2.destroyAllWindows()
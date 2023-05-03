import cv2

# Load video
cap = cv2.VideoCapture('assets/highlights/video-full/Basel-2019-Tuesday-Highlights.mp4')

# Set starting frame
start_frame = 580
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Read the first frame
ret, frame10 = cap.read()
ret, frame9 = cap.read()
ret, frame8 = cap.read()
ret, frame7 = cap.read()
ret, frame6 = cap.read()
ret, frame5 = cap.read()
ret, frame4 = cap.read()
ret, frame3 = cap.read()
ret, frame2 = cap.read()

# Loop through frames and compute differences
for i in range(500):

    # Read the second frame
    ret, frame1 = cap.read()

    average_7 = cv2.addWeighted(frame7, 0.50, frame10, 0.50, 0)
    average_4 = cv2.addWeighted(frame4, 0.50, average_7, 0.50, 0)
    average_1 = cv2.addWeighted(frame1, 0.70, average_4, 0.70, 0)

    frame10 = frame9
    frame9 = frame8
    frame8 = frame7
    frame7 = frame6
    frame6 = frame5
    frame5 = frame4
    frame4 = frame3
    frame3 = frame2
    frame2 = frame1

    # Display the difference image
    cv2.imshow('title', average_1)

    cv2.waitKey(200)

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()

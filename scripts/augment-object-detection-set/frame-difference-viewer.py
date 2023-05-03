import cv2

# Load video
cap = cv2.VideoCapture('assets/US-Open-2022-Final-First-Set-W.mp4')

# Set starting frame
start_frame = 1300
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Define kernel for dilation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Initialize variables for previous differences
prev_diff = None
prev_diff_2 = None

# Read the first frame
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Loop through frames and compute differences
for i in range(500):

    # Read the second frame
    ret, frame2 = cap.read()

    # Convert the frames to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the frames
    diff = cv2.absdiff(gray1, gray2)

    # Apply dilation to the difference image
    diff = cv2.dilate(diff, kernel)

    # Apply weighted addition to previous differences
    if prev_diff is None:
        prev_diff = diff
    if prev_diff_2 is None:
        prev_diff_2 = prev_diff

    new_diff = cv2.addWeighted(diff, 1, prev_diff, 0.33, 0)
    new_diff = cv2.addWeighted(new_diff, 1, prev_diff_2, 0.33, 0)

    # Update previous differences
    prev_diff_2 = prev_diff
    prev_diff = diff

    # Update current frame and grayscale image
    gray1 = gray2
    frame1 = frame2

    # Display the difference image
    cv2.imshow('title', new_diff)

    cv2.waitKey(50)

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()

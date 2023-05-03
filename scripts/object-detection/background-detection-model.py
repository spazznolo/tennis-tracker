
import cv2
import os
import time

start_time = time.time()

# Read the video
cap = cv2.VideoCapture('assets/highlights/video-chunk/Auckland-2023-Quarter-Final-Highlights-5.mp4')

# Create the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=25, varThreshold=100, detectShadows=True)

# Initialize the image counter
img_counter = 0

# Loop through each frame in the video
while True:

    # Read the frame from the video
    ret, frame = cap.read()
    
    if not ret:
        # End of video
        break

    if img_counter % 5 != 0 or img_counter > 125:
        img_counter += 1
        continue

    # Apply the background subtractor to obtain the foreground mask
    fg_mask = bg_subtractor.apply(frame)
    
    #cv2.imshow('title', fg_mask)

    #cv2.waitKey(50)
    # Save the foreground mask and the original frame as JPEG images
    cv2.imwrite(f'assets/highlights/background-frames/Auckland 2023 Quarter-Final Highlights-0_{img_counter:06d}.jpg', fg_mask)
    
    # Increment the image counter
    img_counter += 1
    
# Release the video
cap.release()

print("--- %s seconds ---" % (time.time() - start_time))
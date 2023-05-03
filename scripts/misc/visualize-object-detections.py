
import numpy as np
import pandas as pd
import cv2

# Read the csv file
df = pd.read_csv('assets/highlights/raw-player-ball-locations/Auckland-2023-Quarter-Final-Highlights.csv')

# Read the video
cap = cv2.VideoCapture('assets/Auckland-2023-Quarter-Final-Highlights-1080.mp4')
#cap = cv2.VideoCapture('assets/highlights/video-full/Auckland-2023-Quarter-Final-Highlights.mp4')

# Define the font and the text color
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 0, 255) # red color in BGR format
past_location = (0, 0)
past_location2 = (0, 0)

# Loop through each row in the dataframe
for index, row in df.iterrows():

    # Get the frame number, the frames to event, and the event
    frame_number = row['frame']
    x = row['x']
    y = row['y']
    w = row['w']
    h = row['h']

    # Read the frame from the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()
    y_stretch, x_stretch, z_stretch = frame.shape
    # Add the text to the frame
    cv2.putText(frame, f'Frame Number: {frame_number}', (10, 30), font, 1, color, 2, cv2.LINE_AA)
    #frame = np.zeros((576, 1024, 3), dtype=np.uint8)

    ball_color = (255, 0, 0)

    cv2.rectangle(frame, 
                  (int((x - 0.5*w)*x_stretch), int((y - 0.5*h)*y_stretch)),
                  (int((x + 0.5*w)*x_stretch), int((y + 0.5*h)*y_stretch)), 
                  ball_color, 2)

    # Display the frame
    cv2.imshow('Frame', frame)
    cv2.waitKey(250) # waits for 1ms before displaying the next frame

# Release the video and destroy all windows
cap.release()
cv2.destroyAllWindows()

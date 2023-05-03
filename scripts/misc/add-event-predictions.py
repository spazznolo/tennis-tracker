
import numpy as np
import pandas as pd
import cv2

# Read the csv file
df = pd.read_csv('assets/highlights/raw-player-ball-locations/Auckland-2023-Quarter-Final-Highlights.csv')

# Read the video
cap = cv2.VideoCapture('assets/highlights/video-full/Auckland-2023-Quarter-Final-Highlights.mp4')

# Define the font and the text color
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 0, 255) # red color in BGR format
past_location = (0, 0)
past_location2 = (0, 0)

# Loop through each row in the dataframe
for index, row in df.iterrows():

    # Get the frame number, the frames to event, and the event
    frame_number = row['frame']
    frames_to_hit = row['hit_avg']
    event_hit = row['hit_chg']
    frames_to_bounce = row['bounce_avg']
    event_bounce = row['bounce_chg']
    x_location = row['x_new']
    y_location = row['y_new']
    back_x_location = row['back_x']
    back_y_location = row['back_y']
    front_x_location = row['front_x']
    front_y_location = row['front_y']


    # Read the frame from the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    # Add the text to the frame
    cv2.putText(frame, f'Frame Number: {frame_number}', (10, 30), font, 1, color, 2, cv2.LINE_AA)
    #frame = np.zeros((576, 1024, 3), dtype=np.uint8)

    if event_hit > 0: ball_color = (0, 255, 0) 
    elif event_bounce > 0: ball_color = (0, 0, 255)
    else: ball_color = (255, 0, 0)

    frame2 = cv2.addWeighted(frame, 0.25, frame, 0, 0)
    cv2.circle(frame2, past_location2, 2, ball_color, 2)

    frame1 = cv2.addWeighted(frame, 0.50, frame, 0, 0)   
    cv2.circle(frame1, past_location, 2, ball_color, 2)

    cv2.circle(frame, (int(x_location*1024), int(y_location*576)), 2, ball_color, 2)
    cv2.circle(frame, (int(back_x_location*1024), int(back_y_location*576)), 5, ball_color, 5)
    cv2.circle(frame, (int(front_x_location*1024), int(front_y_location*576)), 5, ball_color, 5)

    frame = cv2.addWeighted(frame, 1, frame2, 0.5, 0)
    frame = cv2.addWeighted(frame, 1, frame1, 0.5, 0)

    past_location2 = past_location
    past_location = (int(x_location*1020), int(y_location*580))
    # Display the frame
    cv2.imshow('Frame', frame)
    cv2.waitKey(50) # waits for 1ms before displaying the next frame

# Release the video and destroy all windows
cap.release()
cv2.destroyAllWindows()

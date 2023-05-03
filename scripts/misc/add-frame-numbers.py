
import os
import math
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

input_dir = 'assets/highlights/video-full'
output_dir = 'assets/highlights/video-full-with-frame-number/'

for filename in os.listdir(input_dir):

    # Open the input video file
    cap = cv2.VideoCapture(str(input_dir) + filename)
    print(str(input_dir) + filename)
    # Define the FFMPEG writer object
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_dir) + str(filename), fourcc, fps, (width, height))

    # Read and process each frame
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add label to the frame
        label = 'Frame {}'.format(frame_number)
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the labeled frame to the output video file
        out.write(frame)

        frame_number += 1

    # Release the input and output video files
    cap.release()
    out.release()
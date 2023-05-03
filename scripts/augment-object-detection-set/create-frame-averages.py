
import cv2
import os

# Get the list of JPEGs in the repository
jpeg_dir = 'assets/highlights/object-detection-train-frames/'
video_dir = 'assets/highlights/video-full'

jpeg_files = sorted([f for f in os.listdir(jpeg_dir) if f.endswith('.jpg')])

# Create a structuring element for dilation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Loop through the rest of the frames and calculate the frame difference
for i in range(1, len(jpeg_files)):

    # Split the filename into parts using '-' as the delimiter
    parts = jpeg_files[i].split('-')

    # The last part will contain the number you want to manipulate
    last_part = parts[-1]
    video_file = '-'.join(parts[:-1]) + '.mp4'
    # Convert the last part to an integer
    number = int(last_part.split(".")[0])

    if number == 0: continue

    # Load video
    cap = cv2.VideoCapture('assets/highlights/video-chunk/' + video_file)
    print(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, number - 7)

    # Read the second frame
    ret, frame8 = cap.read()
    # Read the second frame
    ret, frame7 = cap.read()
    # Read the second frame
    ret, frame6 = cap.read()
    # Read the second frame
    ret, frame5 = cap.read()
    # Read the second frame
    ret, frame4 = cap.read()
    # Read the second frame
    ret, frame3 = cap.read()
    # Read the second frame
    ret, frame2 = cap.read()

    # Load the frames
    frame1 = cv2.imread(str('assets/highlights/object-detection-train-frames/' + jpeg_files[i]))
    
    average_7 = cv2.addWeighted(frame7, 0.55, frame8, 0.45, 0)
    average_6 = cv2.addWeighted(frame6, 0.55, average_7, 0.45, 0)
    average_5 = cv2.addWeighted(frame5, 0.55, average_6, 0.45, 0)
    average_4 = cv2.addWeighted(frame4, 0.55, average_5, 0.45, 0)
    average_3 = cv2.addWeighted(frame3, 0.55, average_4, 0.45, 0)
    average_2 = cv2.addWeighted(frame2, 0.55, average_3, 0.45, 0)
    average_1 = cv2.addWeighted(frame1, 0.70, average_2, 0.30, 0)

    cv2.imwrite(str('assets/highlights/object-detection-frame-averages/' + jpeg_files[i]), average_1)

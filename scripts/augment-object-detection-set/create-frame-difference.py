
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, number - 6)

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
    
    new_diff_2 = cv2.addWeighted(frame4, 0.60, frame7, 0.40, 0)
    new_diff = cv2.addWeighted(frame1, 0.75, new_diff_2, 0.5, 0)

    cv2.imwrite(str('assets/highlights/object-detection-frame-differences/' + jpeg_files[i]), new_diff)

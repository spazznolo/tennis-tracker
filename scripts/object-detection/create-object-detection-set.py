
import os
import math
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

dir = 'assets/highlights/video-full'

allfiles = os.listdir(dir)
full_games = [fname for fname in allfiles if fname.endswith('.mp4')]

for full_game in full_games:

    full_game_no_ext = os.path.splitext(full_game)[0]
    video = cv2.VideoCapture(str(dir) + str(full_game))
    fps = video.get(cv2.CAP_PROP_FPS) 

    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps

    # clip 60 seconds for each chunk (bleed into each other)
    chunks = math.floor(duration/60)

    for i in range(0, chunks):

        ffmpeg_extract_subclip(
            str(dir) + str(full_game), 
            (60*i), 
            (60*(i + 1)), 
            targetname = 'assets/highlights/video-chunk/' + str(full_game_no_ext) + '-' + str(i) + '.mp4')

        vidcap = cv2.VideoCapture('assets/highlights/video-chunk/' + str(full_game_no_ext) + '-' + str(i) + '.mp4')
        success, image = vidcap.read()

        for count in range(0, 5*25):
            if count % 5 == 0:
                cv2.imwrite('assets/highlights/object-detection-train-frames/' + str(full_game_no_ext) + '-' + str(i) + '-%d.jpg' % count, image)   
                success, image = vidcap.read()
                print('Wrote new frame: ', dir + '/' + str(full_game_no_ext) + '-' + str(i) + '-' + str(count) + '.jpg')
            else: success, image = vidcap.read()





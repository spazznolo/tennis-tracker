
import os
import math
import random
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

dir = 'assets/full-games'

allfiles = os.listdir(dir)
full_games = [fname for fname in allfiles if fname.endswith('.mp4')]

for full_game in full_games:

    full_game_no_ext = os.path.splitext(full_game)[0]
    video = cv2.VideoCapture('assets/full-games/' + str(full_game))
    fps = video.get(cv2.CAP_PROP_FPS) 
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps

    # clip 60 seconds for each chunk (bleed into each other)
    chunks = math.floor(duration/60)

    if os.path.isdir('assets/game-chunks/' + str(full_game)) == True:
        continue

    else:

        sampled_chunks = random.sample(range(chunks), 10)

        for i in sampled_chunks:

            ffmpeg_extract_subclip(
                'assets/full-games/' + str(full_game), 
                (60*i), 
                (60*(i + 1)), 
                targetname = 'assets/game-chunks/' + str(full_game_no_ext) + '-' + str(i) + '.mp4')

            vidcap = cv2.VideoCapture('assets/game-chunks/' + str(full_game_no_ext) + '-' + str(i) + '.mp4')
            success, image = vidcap.read()

            count = 0

            while success:
                
                cv2.imwrite('assets/game-frames/' + str(full_game_no_ext) + '-' + str(i) + '-%d.jpg' % count, image)     # save frame as JPEG file      
                success, image = vidcap.read()
                print('Read a new frame: ', success)
                count += 1
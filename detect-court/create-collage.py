
import os
import sys
import glob
from PIL import Image

images = [Image.open("assets/hough-batch/" + str(x)) for x in os.listdir("assets/hough-batch")]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

files = glob.glob('assets/hough-batch/*')
for f in files:
    os.remove(f)

new_im.save('/Users/ada/Documents/projects/spazznolo.github.io/figs/hough-collage-3.jpg')
import cv2
import os

# Read the images from the 'images' directory
images = sorted(os.listdir('assets/background-frames'))

# Loop through each image
for i in range(7, (len(images) - 7)):
    # Load the current image and the two previous images
    current_image = cv2.imread(f'assets/background-frames/{images[i]}')
    previous_image1 = cv2.imread(f'assets/background-frames/{images[i-3]}')
    previous_image2 = cv2.imread(f'assets/background-frames/{images[i-6]}')
    next_image1 = cv2.imread(f'assets/background-frames/{images[i+3]}')
    next_image2 = cv2.imread(f'assets/background-frames/{images[i+6]}')
    
    # Reduce the intensity of the two previous images by 50%
    previous_image1 = cv2.addWeighted(previous_image1, 0.8, previous_image1, 0, 0)
    previous_image2 = cv2.addWeighted(previous_image2, 0.9, previous_image2, 0, 0)
    next_image1 = cv2.addWeighted(next_image1, 0.8, next_image1, 0, 0)
    next_image2 = cv2.addWeighted(next_image2, 0.9, next_image2, 0, 0)

    # Add the two previous images to the current image
    current_image = cv2.addWeighted(current_image, 1, previous_image1, 0.5, 0)
    current_image = cv2.addWeighted(current_image, 1, previous_image2, 0.5, 0)
    current_image = cv2.addWeighted(current_image, 1, next_image1, 0.5, 0)
    current_image = cv2.addWeighted(current_image, 1, next_image2, 0.5, 0)
    
    # Display the resulting image
    #cv2.imshow('Result', current_image)
    #cv2.waitKey(10)
    cv2.imwrite(f'assets/trail-frames/{images[i]}', current_image)

# Destroy all windows
cv2.destroyAllWindows()

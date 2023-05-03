
import os
import numpy as np
import pandas as pd

# Define the path to the text file containing the YOLO labels

def get_txt_files(directory):

    txt_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

# Set the path to the directory
directory = "assets/highlights/object-detection-prediction-labels/"

# Call the function to get the names of all .txt files recursively
file_names = get_txt_files(directory)

full_array = np.zeros((500000, 7), dtype=object)
idx1 = 0

for file_name in file_names:

    if os.path.isfile(file_name):
        # Load the labels from the file into a list
        with open(file_name, "r") as f:
            labels = [line.strip() for line in f.readlines()]
    else:
        continue
    
    # Split the labels into individual coordinates and store them in a NumPy array
    labels_array = np.array([label.split() for label in labels], dtype=object)
    idx2 = idx1 + labels_array.shape[0]    
    full_array[idx1:idx2,] = np.insert(labels_array, 0, os.path.basename(file_name), axis=1)
    idx1 = idx2 + 1

# Convert the NumPy array to a Pandas DataFrame
my_dataframe = pd.DataFrame(full_array, columns = ['frame', 'class', 'x1', 'y1', 'width', 'height', 'prob'])

my_dataframe.to_csv('assets/locations.csv', index = False)
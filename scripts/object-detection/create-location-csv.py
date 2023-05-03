
import os
import numpy as np
import pandas as pd

# Define the path to the text file containing the YOLO labels

# find files in path
in_directory = 'assets/highlights/object-detection-prediction-labels/'
game_folders = os.listdir(in_directory)
game_folders = [fname for fname in game_folders if fname.endswith('Highlights')]

for game in game_folders:

    full_array = np.zeros((500000, 7))
    idx1 = 0
    labels_folder = os.listdir(in_directory + str(game))

    for game_labels in labels_folder:

        game_labels = in_directory + str(game) + str('/') + str(game_labels)
        split_string = game_labels.split("_")
        i = os.path.splitext(split_string[1])[0]

        if os.path.isfile(game_labels):
            # Load the labels from the file into a list
            with open(game_labels, "r") as f:
                labels = [line.strip() for line in f.readlines()]
        else:
            continue


        # Split the labels into individual coordinates and store them in a NumPy array
        labels_array = np.array([label.split() for label in labels], dtype=float)
        idx2 = idx1 + labels_array.shape[0]
        full_array[idx1:idx2,] = np.insert(labels_array, 0, i, axis=1)
        idx1 = idx2 + 1

    # Convert the NumPy array to a Pandas DataFrame
    my_dataframe = pd.DataFrame(full_array, columns = ['frame', 'class', 'x', 'y', 'w', 'h', 'prob'])
    my_dataframe = my_dataframe[my_dataframe['frame'] > 0]

    my_dataframe.to_csv('assets/highlights/raw-player-ball-locations/' + game + '.csv', index = False)




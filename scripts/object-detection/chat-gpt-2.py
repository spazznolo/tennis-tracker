
import numpy as np
from scipy.spatial.distance import cdist
import time

def generate_data(num_frames, num_objects):
    data = np.zeros((num_frames*num_objects, 3))
    for i in range(num_frames):
        for j in range(num_objects):
            data[i*num_objects+j] = [i, np.random.rand(), np.random.rand()]
    return data

def find_closest_objects_loop(frame, data):
    current_data = data[data[:, 0] == frame]
    output = np.zeros((current_data.shape[0], 4))
    for i in range(current_data.shape[0]):
        current_loc = current_data[i, 1:]
        prev_data = data[(data[:, 0] < frame) & (data[:, 0] >= frame-3)]
        prev_locs = prev_data[:, 1:]
        dists = np.linalg.norm(current_loc - prev_locs, axis=1)
        closest_index = np.argmin(dists)
        output[i] = [frame, *current_loc, dists[closest_index]]
    return output

def find_closest_objects_vec(frame, data):
    current_locs = data[data[:, 0] == frame]
    prev_locs = data[(data[:, 0] < frame) & (data[:, 0] > frame - 4)]
    dists = cdist(current_locs[:, 1:], prev_locs[:, 1:])
    min_idxs = np.argmin(dists, axis=1)

    # find unique values in the column to split on
    split_col = prev_locs[:, 0]
    unique_vals = np.unique(split_col)

    # split b based on the unique values
    b_groups = [prev_locs[split_col == val] for val in unique_vals]

    # calculate distances between each row in a and each group in b
    dists = [cdist(current_locs[:, 1:], b_group[:, 1:]) for b_group in b_groups]

    # find the index of the row in each group with the smallest distance for each row in a
    min_idxs = [np.argmin(d, axis=1) for d in dists]
    print(min_idxs)
    return min_idxs

# Generate data
data = generate_data(100000, 3)

# Time loop-based implementation
start_time = time.time()
for i in range(3, 100):
    find_closest_objects_loop(i, data)
loop_time = time.time() - start_time

# Time vectorized implementation
start_time = time.time()
for i in range(3, 100):
    find_closest_objects_vec(i, data)
vec_time = time.time() - start_time

print(f"Loop time: {loop_time:.4f} seconds")
print(f"Vectorized time: {vec_time:.4f} seconds")


import pandas as pd
import numpy as np
from pykalman import KalmanFilter

# create example dataframe with x and y coordinates
df = pd.read_csv('assets/locations2.csv')
df = df.loc[(df['class'] == 2) & (df['prob'] > 0.3)]

# define function to get rows with highest value for each group
def get_top_rows(group):
    return group.loc[group['prob'].idxmax()]

# group by 'group' column and apply function to get top rows
df = df.groupby('frame').apply(get_top_rows).reset_index(drop=True)

df_raw = df[['x1', 'y1']]

# create Kalman filter model
kf = KalmanFilter(n_dim_state=4, n_dim_obs=2)
kf.transition_matrices = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])  # use a 5-step window in both dimensions
kf.observation_matrices = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # observe position only

# apply Kalman filter to dataframe
filtered_state_means, _ = kf.filter(df_raw.values)

# create new dataframe with filtered x and y coordinates
df_filtered = pd.DataFrame(filtered_state_means, columns=['x', 'y', 'a', 'b'])
df_filtered = pd.concat([df_filtered, df], axis = 1)

# print original and filtered dataframes
print('Original dataframe:')
print(df)
print('Filtered dataframe:')
print(df_filtered)

df_filtered.to_csv('assets/locations3.csv', index = False)

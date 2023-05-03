def load_file(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file and returns it as a Pandas DataFrame.

    Parameters:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: The loaded CSV file as a Pandas DataFrame.
    """
    # Load file as a data frame
    df = pd.read_csv(file_path)
    # Extract match ID from file name
    match_id = os.path.splitext(os.path.basename(file_path))[0]
    # Add match ID as a column in the data frame
    df['match_id'] = match_id
    return df
    
def get_event_distances(df, event_types):
    """
    Calculates the distance between events of a specified type in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the events.
    event_type : str
        The type of event to calculate distances for. Must be one of {'bounce', 'hit', 'na'}.

    Returns:
    --------
    df : pandas.DataFrame
        The modified DataFrame with additional columns for the event distances.
    """
    for event_type in event_types:
      if event_type == 'bounce':
          df[f'{event_type}'] = np.where(df['event'] == 2, 1, 0)
      elif event_type == 'hit':
          df[f'{event_type}'] = np.where(df['event'] == 1, 1, 0)
      elif event_type == 'na':
          df[f'{event_type}'] = np.where(df['x1'].notna(), 1, 0)
      else:
          raise ValueError('No valid event type')

      df[f'{event_type}_group'] = df[f'{event_type}'].cumsum()
      df[f'{event_type}_dist_down'] = df.groupby(f'{event_type}_group').cumcount()
      df[f'{event_type}_dist_up'] = df.groupby(f'{event_type}_group').cumcount(ascending=False) + 1
      df[f'{event_type}_dist'] = df[[f'{event_type}_dist_up', f'{event_type}_dist_down']].min(axis=1)

    return df


def impute_locations(df, limit, columns_to_impute, na_dist):
    """
    Imputes missing values in specified columns using linear interpolation method
    with a given limit on the number of consecutive missing values that can be filled.

    Args:
    - df (pandas.DataFrame): the dataframe containing the data to be imputed
    - limit (int): the maximum number of consecutive missing values to be filled
    - columns_to_impute (list): a list of column names to be imputed
    - na_x_dist (pandas.Series): a series containing the distance between the current
        location and the previous non-missing location for each row in the dataframe

    Returns:
    - pandas.DataFrame: the dataframe with imputed values in the specified columns,
        and a new column 'size' containing the product of the imputed 'width_new' and 'height_new' columns
    """
    for col in columns_to_impute:
        df[f'{col}_new'] = df[col].interpolate(method="linear", limit=limit, limit_direction='both')
        df[f'{col}_new'] = np.where(df['na_dist'] >= na_dist, np.nan, df[f'{col}_new']) 

    return df


def calculate_speeds_and_distances(df, shift_values):
  
    for shift in shift_values:
        df[f'x_speed_past_{shift}'] = (df['x_new'] - df['x_new'].shift(shift))/shift
        df[f'y_speed_past_{shift}'] = (df['y_new'] - df['y_new'].shift(shift))/shift
        df[f'ball_speed_past_{shift}'] = np.sqrt(df[f'x_speed_past_{shift}']**2 + df[f'y_speed_past_{shift}']**2)
        
        df[f'x_speed_future_{shift}'] = (df['x_new'] - df['x_new'].shift(-shift))/shift
        df[f'y_speed_future_{shift}'] = (df['y_new'] - df['y_new'].shift(-shift))/shift
        df[f'ball_speed_future_{shift}'] = np.sqrt(df[f'x_speed_future_{shift}']**2 + df[f'y_speed_future_{shift}']**2)

        df[f'x_accel_past_{shift}'] = (df[f'x_speed_past_{shift}'] - df[f'x_speed_past_{shift}'].shift(-shift))/shift
        df[f'y_accel_past_{shift}'] = (df[f'y_speed_past_{shift}'] - df[f'y_speed_past_{shift}'].shift(-shift))/shift
        df[f'ball_accel_past_{shift}'] = np.sqrt(df[f'x_accel_past_{shift}']**2 + df[f'y_accel_past_{shift}']**2)

        df[f'x_accel_future_{shift}'] = (df[f'x_speed_future_{shift}'] - df[f'x_speed_future_{shift}'].shift(-shift))/shift
        df[f'y_accel_future_{shift}'] = (df[f'y_speed_future_{shift}'] - df[f'y_speed_future_{shift}'].shift(-shift))/shift
        df[f'ball_accel_future_{shift}'] = np.sqrt(df[f'x_accel_future_{shift}']**2 + df[f'y_accel_future_{shift}']**2)

        df[f'x_avg_speed_past_{shift}'] = df['x_speed_past_1'].rolling(shift, min_periods=shift).mean()
        df[f'y_avg_speed_past_{shift}'] = df['y_speed_past_1'].rolling(shift, min_periods=shift).mean()
        df[f'ball_avg_speed_past_{shift}'] = df['ball_speed_past_1'].rolling(shift, min_periods=shift).mean()

        df[f'x_avg_speed_future_{shift}'] = df['x_speed_future_1'].rolling(shift, min_periods=shift).mean()
        df[f'y_avg_speed_future_{shift}'] = df['y_speed_future_1'].rolling(shift, min_periods=shift).mean()
        df[f'ball_avg_speed_future_{shift}'] = df['ball_speed_future_1'].rolling(shift, min_periods=shift).mean()

        df[f'x_speed_delta_{shift}'] = df[f'x_avg_speed_future_{shift}'] - df[f'x_avg_speed_past_{shift}']
        df[f'y_speed_delta_{shift}'] = df[f'y_avg_speed_future_{shift}'] - df[f'y_avg_speed_past_{shift}']
        df[f'ball_speed_delta_{shift}'] = df[f'ball_avg_speed_future_{shift}'] - df[f'ball_avg_speed_past_{shift}']

    return df

from selenium import webdriver
from bs4 import BeautifulSoup
from time import sleep
import numpy as np
import pandas as pd

scores_A = np.array([], dtype=int)
scores_B = np.array([], dtype=int)
URL = "https://www.tennis24.com/match/vRTJeN3j/#/match-summary/point-by-point/0"

driver = webdriver.Chrome(r"venv-tennis/bin/chromedriver.exe")
driver.get(URL)
sleep(3)

soup = BeautifulSoup(driver.page_source, "html.parser")

divs = soup.find_all('div', {'class': 'matchHistoryRow__fifteens'})

for div in divs:
    spans = div.find_all('span', {'class': 'matchHistoryRow__fifteen'})
    if spans:
        for span in spans:
            score_str = span.text.strip()
            score_list = score_str.split(', ')
            print(score_list)

            scores_AB = np.array([score.split(':') for score in score_list if len(score) > 0], dtype=object)
            scores_AB = np.where(scores_AB == 'A', '41', scores_AB)
            scores_A = np.append(scores_A, scores_AB[:, 0])
            scores_B = np.append(scores_B, scores_AB[:, 1])

# Create a DataFrame from the two arrays
df = pd.DataFrame({'Player_A': scores_A, 'Player_B': scores_B})

# Add a new column 'point' with row numbers
df.insert(0, 'point', range(1, len(df) + 1))

# Add a new column 'game' with the game number
df['game'] = np.where((df['Player_A'] < df['Player_A'].shift(1)) | (df['Player_B'] < df['Player_B'].shift(1)), 1, 0)
df['game'] = df['game'].cumsum() + 1

final_points = df.groupby('game')['Player_A', 'Player_B'].max().reset_index()
print(final_points.dtypes)
#final_points['Player_A'] = np.where(final_points['Player_A'] > final_points['Player_B'], 45, final_points['Player_A'])
#final_points['Player_B'] = np.where(final_points['Player_B'] > final_points['Player_A'], 45, final_points['Player_B'])

new_df = pd.concat([df, final_points], axis = 0)

driver.quit()

new_df.to_csv('assets/outputs/pbp-points.csv')


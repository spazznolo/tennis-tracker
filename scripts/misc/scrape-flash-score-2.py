
from selenium import webdriver
from bs4 import BeautifulSoup
from time import sleep
import numpy as np
import pandas as pd

box_score = np.empty((100, 10), dtype=object)
scores_A = np.array([], dtype=int)
scores_B = np.array([], dtype=int)

URL = "https://www.tennis24.com/match/vRTJeN3j/#/match-summary/point-by-point/0"

driver = webdriver.Chrome(r"venv-tennis/bin/chromedriver.exe")
driver.get(URL)
sleep(3)

soup = BeautifulSoup(driver.page_source, "html.parser")

titles = soup.find_all('div', {'title': 'Serving player'})

i=0
for title in titles:
    if any("home" in s for s in title.parent['class']): box_score[i, 4] = 1
    else: box_score[i, 4] = 0
    i+=1


divs1 = soup.find_all('div', {'class': 'matchHistoryRow__score'})

i=0
for div in divs1:

    j = np.where(i % 2 == 1, int(i/2 - 0.45), int(i/2))

    if i % 2 == 0: box_score[j, 2] = div.text.strip()
    else: box_score[j, 3] = div.text.strip()
    i+=1


player_names = soup.find_all('div', {'class': "participant__participantName participant__overflow"})

i=0
for player_name in player_names:

    box_score[0:(len(titles)), i] = player_name.text.strip()
    i+=1


divs = soup.find_all('div', {'class': 'matchHistoryRow__fifteens'})

for div in divs:
    spans = div.find_all('span', {'class': 'matchHistoryRow__fifteen'})
    if spans:
        for span in spans:
            score_str = span.text.strip()
            score_list = score_str.split(', ')

            scores_AB = np.array([score.split(':') for score in score_list if len(score) > 0], dtype=object)
            scores_AB = np.where(scores_AB == 'A', '41', scores_AB)
            scores_A = np.append(scores_A, scores_AB[:, 0])
            scores_B = np.append(scores_B, scores_AB[:, 1])

# Create a DataFrame from the two arrays
df = pd.DataFrame({'Player_A': scores_A.astype(int), 'Player_B': scores_B.astype(int)})

# Add a new column 'game' with the game number
df['game'] = np.where((df['Player_A'] + 2 < df['Player_A'].shift(1)) | (df['Player_B'] + 2 < df['Player_B'].shift(1)), 1, 0)
df['game'] = df['game'].cumsum() + 1

df['point'] = range(1, len(box_score) + 1)

final_points = df.loc[df.groupby('game')['Player_A'].agg(lambda x: x.index.max())]

final_points['Player_A'] = np.where(final_points['Player_A'] > final_points['Player_B'], 45, final_points['Player_A'])
final_points['Player_B'] = np.where(final_points['Player_B'] > final_points['Player_A'], 45, final_points['Player_B'])

box_score = pd.concat([df, final_points], axis = 0)

# Add a new column 'point' with row numbers
box_score.insert(0, 'point', range(1, len(box_score) + 1))

print(box_score)


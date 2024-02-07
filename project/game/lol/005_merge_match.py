import pandas as pd

# winner.csv
win_file = '/content/match_winner_data_version1.csv'

columns_win = ['win', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald', 'towerKills', 'inhibitorKills', 'baronKills', 'dragonKills']
win_df = pd.read_csv(win_file)
df1 = win_df[columns_win]

df1.to_csv('match_winner.csv',index=False)

# loser.csv
lose_file = '/content/match_loser_data_version1.csv'

columns_lose = ['win', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald', 'towerKills', 'inhibitorKills', 'baronKills', 'dragonKills']
lose_df = pd.read_csv(lose_file)
df2 = lose_df[columns_lose]

df2.to_csv('match_loser.csv', index=False)

# 두 DataFrame을 합치기
merged_df = pd.merge(df1, df2, right_on='win', left_on='win')

# 합쳐진 DataFrame을 새로운 CSV 파일로 저장
merged_df.to_csv('match.csv', index=False)
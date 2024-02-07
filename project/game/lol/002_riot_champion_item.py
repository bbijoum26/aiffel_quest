from pandas.io.json import json_normalize
import requests
import pandas as pd
import numpy as np

req1 = requests.get('http://ddragon.leagueoflegends.com/cdn/13.11.1/data/ko_KR/champion.json')

# 챔피언 데이터에서 챔피언 이름을 추출하여 champ_ls에 저장
champ_ls = list(req1.json()['data'].keys())

champ_df = pd.DataFrame()

# 각 챔피언에 대해 챔피언 데이터를 추출하여 champ_df에 추가
for i in range(len(champ_ls)):
    # 챔피언 데이터를 JSON 정규화하여 DataFrame으로 변환
    pre_df = json_normalize(req1.json()['data'][champ_ls[i]])
    
    # 추출한 데이터를 champ_df에 추가
    champ_df = champ_df.append(pre_df)

# champ_df를 CSV 파일로 저장
champ_df.to_csv('riot_champion.csv', encoding='utf-8-sig')


req2 = requests.get('http://ddragon.leagueoflegends.com/cdn/13.11.1/data/ko_KR/item.json')

item_ls = []

# 아이템 데이터에서 유효한 아이템 ID를 찾아서 item_ls에 추가
for i in list(range(0, 10000)):
    try:
        a = req2.json()['data'][str(i)]
        item_ls.append(str(i))
    except:
        pass

item_table = pd.DataFrame()

# 유효한 아이템 ID를 기반으로 아이템 정보를 추출하여 item_table에 추가
for i in item_ls:
    item_id = i
    
    # 아이템 이름 추출
    try:
        name = req2.json()['data'][i]['name']
    except:
        name = np.nan
        
    # 조합 아이템 추출
    try:
        upper_item = req2.json()['data'][i]['into']
    except:
        upper_item = np.nan
    
    # 아이템 설명 추출
    try:
        explain = req2.json()['data'][i]['plaintext']
    except:
        explain = np.nan
    
    # 구매 가격 추출
    try:
        buy_price = req2.json()['data'][i]['gold']['base']
    except:
        buy_price = np.nan
    
    # 판매 가격 추출
    try:
        sell_price = req2.json()['data'][i]['gold']['sell']
    except:
        sell_price = np.nan
        
    # 아이템 태그 추출
    try:
        tag = req2.json()['data'][i]['tags'][0]
    except:
        tag = np.nan
    
    # 추출한 정보를 DataFrame에 추가
    pre_df = pd.DataFrame({
        'item_id': [item_id],
        'name': [name],
        'upper_item': [upper_item],
        'explain': [explain],
        'buy_price': [buy_price],
        'sell_price': [sell_price],
        'tag': [tag]
    })
    
    item_table = item_table.append(pre_df)

# item_table를 CSV 파일로 저장
item_table.to_csv('riot_item.csv', encoding='utf-8-sig')
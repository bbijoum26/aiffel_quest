import pandas as pd

df1 = pd.read_csv('C:\Users\fishr\OneDrive\바탕 화면\python\champion_statistics.csv').sort_values(by='챔피언')

# 원본 CSV 파일 경로
original_file = 'C:\Users\fishr\OneDrive\바탕 화면\python\riot_champion.csv'

# 추출할 열(column) 리스트
columns_to_extract = ['id', 'key', 'name', 'title', 'tags', 'info.difficulty']

# 원본 CSV 파일을 DataFrame으로 읽어오기
df = pd.read_csv(original_file)

# 필요한 열(column)만 추출하여 새로운 DataFrame 생성
df2 = df[columns_to_extract]

# name을 기준으로 정렬
df2 = df2.sort_values(by='name')

# 공통된 칼럼을 기준으로 두 DataFrame을 합치기
merged_df = pd.merge(df1, df2, left_on='챔피언', right_on='name')

# 합쳐진 DataFrame을 새로운 CSV 파일로 저장
merged_df.to_csv('champion.csv', index=False)

# name drop
merged_df.drop(['name'], axis=1, inplace=True)

# 챔피언 name 으로 바꾸기
merged_df.rename(columns={'챔피언': 'name'}, inplace=True)
merged_df.rename(columns={'info.difficulty': 'difficulty'}, inplace=True)

# 컬럼 순서 변경
new_order = ['name', 'id', 'key', 'difficulty', 'title', 'tags','플레이수', '픽률(%)', '승률(%)', '밴율(%)']  # 변경된 순서로 컬럼 이름을 나열m
merged_df = merged_df[new_order]

# 저장
merged_df.to_csv('champion.csv', encoding='utf-8-sig', index=False)
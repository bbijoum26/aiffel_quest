import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# entropy 함수 구현
def get_entropy(labels):
    unique_labels, label_counts = np.unique(labels, return_counts=True)       # 각 레이블의 개수를 계산
    probabilities = label_counts / len(labels)                                # 레이블의 개수를 전체 데이터 개수로 나누어 확률을 계산
    entropy = -np.sum(probabilities * np.log2(probabilities))                 # 확률에 로그를 적용하여 엔트로피를 계산
    return entropy

# Information Gain 함수 구현
def get_IG(feature, labels):
    feature_values = np.unique(feature)                                        # 특성의 고유한 값들을 가져옴
    total_entropy = get_entropy(labels)                                        # 전체 데이터의 엔트로피를 계산
    weighted_entropy = 0
    for value in feature_values:
        subset_labels = labels[feature == value]                               # 해당 특성 값에 해당하는 레이블만 추출
        subset_entropy = get_entropy(subset_labels)                            # 해당 특성 값에 대한 엔트로피 계산
        weighted_entropy += len(subset_labels) / len(labels) * subset_entropy  # 해당 특성 값의 가중 엔트로피 계산
    information_gain = total_entropy - weighted_entropy                        # 정보 이득 계산
    return information_gain

# CSV 파일 로드
register_golf_club = pd.read_csv('/Users/project/python/regression/register_golf_club.csv')

# 데이터 확인
print(f'register_golf_club shape = {register_golf_club.shape}') # (14, 6)
print(f'register_golf_club info \n{register_golf_club.info()}')
'''
golf_club의 가입 여부를 확인하는 데이터
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14 entries, 0 to 13
Data columns (total 6 columns):
 #   Column              Non-Null Count  Dtype
---  ------              --------------  -----
 0   Index               14 non-null     int64
 1   age                 14 non-null     object
 2   income              14 non-null     object
 3   married             14 non-null     object
 4   credit_score        14 non-null     object
 5   register_golf_club  14 non-null     object
dtypes: int64(1), object(5)
memory usage: 800.0+ bytes
'''

# 입력 특성과 타겟 분리
X = register_golf_club.iloc[:, 1:-1]    # 첫번째 열, 마지막 열 제외한 나머지 열은 입력 특성
y = register_golf_club.iloc[:, -1]      # 마지막 열은 타겟 변수

# 범주형 데이터를 숫자로 변환
X_encoded = pd.get_dummies(X, columns=['age', 'income', 'married', 'credit_score'])

# 타겟 변수를 숫자로 변화
y_encoded = y.map({'no': 0, 'yes': 1})

# Root node에서의 descriptive feature 선정과 Information Gain 계산
information_gains = []
for feature in X_encoded.columns:
    information_gain = get_IG(X_encoded[feature], y_encoded)        # 특성(feature)과 레이블(labels)을 기반으로 정보 이득을 계산
    information_gains.append((feature, information_gain))           # 특성과 정보 이득을 리스트에 추가

best_feature, best_IG = max(information_gains, key=lambda x: x[1])  # 정보 이득이 가장 큰 특성을 찾음
print(f'descriptive feature: {best_feature}')                       # 가장 큰 정보 이득을 가지는 특성 출력
print(f'Information Gain: {best_IG}')                               # 해당 특성의 정보 이득 출력

# 훈련 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded,
                                                    test_size=0.2, random_state=11)

# DecisionTreeClassifier 모델 생성
model = DecisionTreeClassifier(criterion='entropy')

# DecisionTreeClassifier 모델 속성 출력
for attr in dir(model):
    if not attr.startswith('_'): print(attr)

# 모델 훈련
model.fit(X_train, y_train)
print(f'depth = {model.get_depth()}\nger_n_leaves = {model.get_n_leaves()}')

# 테스트 데이터에 대한 정확도 계산
accuracy = model.score(X_test, y_test)
print(f'{accuracy = }')

# 그래프 그리기
plt.figure(figsize=(12, 8))
tree.plot_tree(model,
               feature_names=X_encoded.columns, # X_encoded의 열 이름 사용
               filled=True, rounded=True)
plt.show()

'''
Decision Tree 결과 분석
1. 모델의 구조
    - 트리의 깊이(depth): depth = 4
                         트리가 4단계까지 얕게 분할 되었음을 의미한다.
                         모델은 트리의 깊이가 얕아져서 각각의 결정 경계가 단순화 된다.
                         이는 모델이 더 일반적인 경향을 학습하게 되는데, 데이터의 노이즈나 이상에 덜 민감해진다는 것을 의미한다.
    - 리프 노드의 개수(get_n_leaves): ger_n_leaves = 5
                                     더 많은 데이터 포인트를 하나의 리프 노트로 묶을 수 있게 된다.
                                     적은 리프 노드 개수는 모델의 일반화 능력을 향상시키는데 도움을 줄 수 있다.
2. 모델 정확도
    - R-squared(결정계수): accuracy = 0.6666666666666666
    - 모델의 테스트 데이터의 변동이 약 66.67% 정도라고 설명할 수 있다.
3. 모델 총평
    - 낮은 깊이와 적은 리프 노드 개수는 모델이 단순해지고 과적합을 방지할 수 있는 효과를 가진다.
      복잡도가 낮아지면서 모델이 데이터의 노이즈나 이상치에 덜 민감해지고, 새로운 데이터에 대한 예측 성능이 개선될 수 있다.
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt

# CSV 파일 로드
bike_sharing = pd.read_csv('C:/python/regression/bike_sharing.csv')

# 데이터 확인
print(f'bike_sharing shape = {bike_sharing.shape}') # (17379, 17)
print(f'bike_sharing info \n{bike_sharing.info()}')
'''
2011-01-01 ~ 2012-12-31 자전거 대여수에 대한 데이터
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 17379 entries, 0 to 17378
Data columns (total 17 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   instant     17379 non-null  int64  
 1   dteday      17379 non-null  object 
 2   season      17379 non-null  int64  
 3   yr          17379 non-null  int64  
 4   mnth        17379 non-null  int64  
 5   hr          17379 non-null  int64  
 6   holiday     17379 non-null  int64  
 7   weekday     17379 non-null  int64  
 8   workingday  17379 non-null  int64  
 9   weathersit  17379 non-null  int64  
 10  temp        17379 non-null  float64
 11  atemp       17379 non-null  float64
 12  hum         17379 non-null  float64
 13  windspeed   17379 non-null  float64
 14  casual      17379 non-null  int64  
 15  registered  17379 non-null  int64  
 16  cnt         17379 non-null  int64  
dtypes: float64(4), int64(12), object(1)
첫번째 시도. dteday (object)를 drop 해도 될 듯 ?(약 99%)
두번째 시도. dteday (object)를 datetime으로 바꿔서 해보기? 
   why? yr (0, 1), mnth(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
=> X_train, X_test: 0 ~ 15: 입력 특성
=> y_train, y_test: 16: 타겟 변수
'''
# object(dteday) 수정
bike_sharing = bike_sharing.drop('dteday', axis=1)
print(f'after bkike_sharing info \n{bike_sharing.info()}')


# 입력 특성과 타겟 분리
X = bike_sharing.iloc[:, :-1]  # 마지막 열을 제외한 나머지 열은 입력 특성
y = bike_sharing.iloc[:, -1]   # 마지막 열은 타겟 변수

# 훈련 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, random_state=11)

# DecisionTreeRegressor 모델 생성
model = DecisionTreeRegressor()

# DecisionTreeRegressor 모델 속성 출력
for attr in dir(model):
    if not attr.startswith('_'): print(attr)

# 모델 훈련
model.fit(X_train, y_train)

print(f'depth = {model.get_depth()}\nger_n_leaves = {model.get_n_leaves()}')

# 테스트 데이터에 대한 정확도 계산
accuracy = model.score(X_test, y_test)
print(f'{accuracy = }')

# 각 feature 별 정확도 분석
feature_accuracies = {}  # 각 feature의 정확도를 저장하기 위한 딕셔너리

for feature in bike_sharing.columns:
    if feature != 'cnt':                                                # 'cnt' feature 제외
        X_train_feature = X_train.drop(feature, axis=1)                 # 현재 feature을 제외한 새로운 훈련 데이터 생성
        X_test_feature = X_test.drop(feature, axis=1)                   # 현재 feature을 제외한 새로운 테스트 데이터 생성

        model_feature = DecisionTreeRegressor()                         # DecisionTreeRegressor 모델 생성
        model_feature.fit(X_train_feature, y_train)                     # 수정된 훈련 데이터로 모델 훈련
        accuracy_feature = model_feature.score(X_test_feature, y_test)  # 수정된 테스트 데이터에 대한 정확도 계산

        feature_accuracies[feature] = accuracy_feature                  # feature_accuracies 딕셔너리에 feature과 해당 정확도 저장

# 결과 출력
for feature, accuracy in feature_accuracies.items():
    print(f'{feature}: {accuracy}')


# 그래프 그리기
plt.figure(figsize=(12, 8))
tree.plot_tree(model,
               feature_names=bike_sharing.columns,
               filled=True, rounded=True)
plt.show()

'''
Decision Tree 결과 분석
1. 모델의 구조
    - 트리의 깊이(depth): depth = 22
                         트리가 22단계까지 깊게 분할 되었음을 의미한다.
                         더 깊은 트리는 더 복잡한 모델을 나타내지만, 과적합 가능성이 높아질 수 있다.
    - 리프 노드의 개수(get_n_leaves): ger_n_leaves = 7296
                                     많은 리프노드는 모델이 데이터를 세분화하고 다양한 패턴을 학습할 수 있음을 나타낸다.
2. 모델 정확도
    - R-squared(결정계수): accuracy = 0.9992371166989799
    - 모델의 테스트 데이터의 변동이 약 99.92% 정도라고 설명할 수 있다.
        이는 모델이 입력된 특성을 기반으로 자전거 대여수를 매우 정확하게 예측할 수 있다는 것을 의미한다.
        하지만, 정확도가 100%에 가까울 경우 과적합 가능성이 있기 때문에 주의해야 한다.
3. 모델 총평
    - 이 모델은 매우 깊고 복잡한 결정 트리를 생성하여 훈련 데이터에 매우 잘 적합하지만,
      복잡성이 새로운 데이터에 대한 일반화 성능을 저하시킬 수 있다.
    - 모델 수정이 필요한 경우, 다른 기법을 통해 모델 성능을 평가해보는 게 좋을 수 있을 거 같다.
'''


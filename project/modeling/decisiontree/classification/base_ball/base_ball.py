import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# CSV 파일 로드
data = pd.read_csv('/Users/project/python/classification/base_ball_dataset.csv')

# 입력 특성과 타깃 분리
X = data.iloc[:, :-1]  # 마지막 열을 제외한 나머지 열은 입력 특성
y = data.iloc[:, -1]   # 마지막 열은 타깃 변수

# 범주형 데이터를 숫자로 변환
X_encoded = pd.get_dummies(X, columns=['weather', 'temperature', 'humidity', 'windy'])

# 학습 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, 
                                                    test_size=0.2, 
                                                    random_state=30)

# 결정 트리 모델 훈련
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print('depth: ', model.get_depth())
print('leef node: ', model.get_n_leaves())

# 테스트 데이터에 대한 정확도 계산
accuracy = model.score(X_test, y_test)
print(f'{accuracy = }')

# 시각화
fig = plt.figure(figsize=(12, 8))
tree.plot_tree(model, 
               feature_names=X_encoded.columns, 
               class_names=model.classes_, 
               filled=True, rounded=True)
plt.show()

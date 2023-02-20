import numpy as np
import pandas as pd
import os
import sys
import tqdm
from sklearn.model_selection import train_test_split

# sizekorea 8차 - 최종 볼륨계산 데이터 불러오기
data = pd.read_excel('data/size_korea_new/최종_선정컬럼_볼륨_계산_3가지_1222.xlsx')
data.info()
print(data.columns)

# 최종 볼륨 제외 볼륨 컬럼 삭제 -> 학습 데이터 셋 형성
data = data[['키', '몸무게', 'BMI', '허리_둘레', '가슴_둘레', '젖_가슴_둘레', '젖_가슴_아래_둘레', '젖_가슴_너비',
    '젖꼭지_사이_수평_길이','젖_가슴_높이', '젖_가슴_아래_높이','겨드랑_높이',
    '가슴_두께', '젖_가슴_두께', '젖_가슴_아래_두께','r1', 'r2', 'r3', 'r4', 'h1', 'h2',
    '유방_타원_원주', '평균r', '평균h', '유방_볼륨_계산_구']]
data = data.dropna(how='any', axis=0)
data.info()
data.to_excel('최종_볼륨_계산 데이터셋.xlsx', index=False)

# 학습 데이터 분리 : 컵_사이즈_신규(문자열) -> 머신러닝시 삭제
X = data[['키', '몸무게', 'BMI', '허리_둘레', '젖_가슴_너비', '젖꼭지_사이_수평_길이',
    '젖_가슴_높이', '젖_가슴_아래_높이','겨드랑_높이',
    '가슴_두께', '젖_가슴_두께', '젖_가슴_아래_두께','r1', 'r2', 'r3', 'r4', 'h1', 'h2',
    '유방_타원_원주', '평균r', '평균h']]
y=data[['유방_볼륨_계산_구']]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=550)

## 1) 랜덤포레스트 회귀
from sklearn.ensemble import RandomForestRegressor
rf_run = RandomForestRegressor(n_estimators=500, random_state=1234)
rf_run.fit(X_train, y_train)
# 모델 정확도
print('일반 모델 훈련 세트 정확도 : {:.3f}'.format(rf_run.score(X_train, y_train)))
print('일반 모델 테스트 세트 정확도 : {:.3f}'.format(rf_run.score(X_test, y_test)))
import pickle
import joblib
joblib.dump(rf_run, 'tools/size_korea_new/sizekorea_rf_model.pkl')
 
# 2) 의사결정나무 회귀
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(max_depth = 5)
dt_reg.fit(X_train, y_train)
# 모델 정확도
print('일반 모델 훈련 세트 정확도 : {:.3f}'.format(dt_reg.score(X_train, y_train)))
print('일반 모델 테스트 세트 정확도 : {:.3f}'.format(dt_reg.score(X_test, y_test)))

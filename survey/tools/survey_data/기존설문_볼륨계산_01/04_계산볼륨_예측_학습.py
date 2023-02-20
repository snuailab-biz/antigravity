import numpy as np
import pandas as pd
import os
import sys
import tqdm
import pickle
import joblib

## 가공 설문 데이터 불러오기 : (이상치 제거 1,248명)
data = pd.read_excel('data/qna_data/설문_가공_볼륨계산.xlsx')
data.info()

## 볼륨 계산 학습 진행 : 모두 존재 1,175명
data = data.dropna(axis=0)
data.info()
print(data.columns)

print("컵 사이즈 정수형 변환")
print("AA: 1, A: 2, B: 3, C: 4, D: 5, E: 6, F: 7")
for i in tqdm.tqdm(range(0, len(data))):
    if data['cup_size_now'].iloc[i]=='AA':
        data['cup_size_now'].iloc[i] = 1
    elif data['cup_size_now'].iloc[i]=='A':
        data['cup_size_now'].iloc[i] = 2
    elif data['cup_size_now'].iloc[i]=='B':
        data['cup_size_now'].iloc[i] = 3
    elif data['cup_size_now'].iloc[i]=='C':
        data['cup_size_now'].iloc[i] = 4
    elif data['cup_size_now'].iloc[i]=='D':
        data['cup_size_now'].iloc[i] = 5
    elif data['cup_size_now'].iloc[i]=='E':
        data['cup_size_now'].iloc[i] = 6
    elif data['cup_size_now'].iloc[i]=='F':
        data['cup_size_now'].iloc[i] = 7

X = data[['height', 'armpit_height', 'middle_breast_height',
       'bottom_breast_height', 'weight', 'BMI', 'wai_cir', 'band_size_now',
       'bottom_breast_round_설문', 'cup_size_now',
       'middle_breast_round', 'middle_breast_width', 'nipple_between_length',
       'top_breast_thickness', 'middle_breast_thickness',
       'bottom_breast_thickness', 'hook_num_now',
       'breast_space_fingers', 'good_bra_score_cup', 'bra_cup_bad_point',
       'good_bra_score_band', 'bra_band_bad_point', 'breast_shape',
       'breast_gol_shape', 'r1', 'r2', 'r3', 'r4', 'h1', 'h2', '유방_타원_원주',
       '평균r', '평균h']]
y=data['유방_볼륨_계산_구']

# 훈련 / 테스트 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=550)

## 1) 랜덤포레스트 회귀
from sklearn.ensemble import RandomForestRegressor
rf_run = RandomForestRegressor(n_estimators=500, random_state=1234)
rf_run.fit(X_train, y_train)
# 모델 정확도
print('일반 모델 훈련 세트 정확도 : {:.3f}'.format(rf_run.score(X_train, y_train)))
print('일반 모델 테스트 세트 정확도 : {:.3f}'.format(rf_run.score(X_test, y_test)))

# 모델 저장
#joblib.dump(rf_run, 'tools/size_korea_new/sizekorea_rf_model.pkl')
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
print(data.columns)

# 데이터 전체 열 이름 변경
data.columns = ['height', 'armpit_height', 'middle_breast_height',
       'bottom_breast_height', 'weight', 'BMI', 'wai_cir', 'band_size_now',
       'bottom_breast_round', 'bottom_breast_round_사이즈코리아',
       'middle_breast_round', 'middle_breast_width', 'nipple_between_length',
       'top_breast_thickness', 'middle_breast_thickness',
       'bottom_breast_thickness', 'cup_size_now', 'hook_num_now',
       'breast_space_fingers', 'good_bra_score_cup', 'bra_cup_bad_point',
       'good_bra_score_band', 'bra_band_bad_point', 'breast_shape',
       'breast_gol_shape', 'r1', 'r2', 'r3', 'r4', 'h1', 'h2', '유방_타원_원주',
       '평균r', '평균h', '유방_볼륨_계산_구']

total_data = data[['height', 'armpit_height', 'middle_breast_height',
        'bottom_breast_height', 'weight', 'BMI', 'wai_cir', 'band_size_now', 'bottom_breast_round',
        'middle_breast_round','middle_breast_width', 'nipple_between_length', 'top_breast_thickness', 'middle_breast_thickness', 'bottom_breast_thickness',
        'cup_size_now','hook_num_now', 'breast_space_fingers', 'good_bra_score_cup',
        'bra_cup_bad_point', 'good_bra_score_band', 'bra_band_bad_point',
        'breast_shape', 'breast_gol_shape']]
total_data.info()

## 반지름 4개 컬럼 생성 : r1 ~ r4
# 1) r1 : 젖_가슴_너비 - 젖꼭지_사이_수평_길이
total_data['r1'] = 0
for i in tqdm.tqdm(range(0,len(total_data))):
    total_data['r1'].iloc[i]=0.5*(total_data['middle_breast_width'].iloc[i] - total_data['nipple_between_length'].iloc[i])
print(total_data['r1'].unique())

# 2) r2 : 젖_가슴_높이 - 젖_가슴_아래_높이
total_data['r2'] = 0
for i in tqdm.tqdm(range(0,len(total_data))):
    total_data['r2'].iloc[i]=(total_data['middle_breast_height'].iloc[i] - total_data['bottom_breast_height'].iloc[i])
print(total_data['r2'].unique())

# 3) r3 : 0.5 * 젖꼭지_사이_수평_길이
total_data['r3'] = 0
for i in tqdm.tqdm(range(0,len(total_data))):
    total_data['r3'].iloc[i]=0.5*(total_data['nipple_between_length'].iloc[i])
print(total_data['r3'].unique())
    
# 4) r4 : 겨드랑_높이 - 젖_가슴_높이
total_data['r4'] = 0
for i in tqdm.tqdm(range(0,len(total_data))):
    total_data['r4'].iloc[i]=(total_data['armpit_height'].iloc[i] - total_data['middle_breast_height'].iloc[i])
print(total_data['r4'].unique())

## 높이(h) 컬럼 생성
# 1) h1 : 젖가슴두께 - 가슴두께
total_data['h1'] = 0
for i in tqdm.tqdm(range(0,len(total_data))):
    total_data['h1'].iloc[i]=(total_data['middle_breast_thickness'].iloc[i] - total_data['top_breast_thickness'].iloc[i])

# 2) h2 : 젖가슴두께 - 젖가슴아래두께
total_data['h2'] = 0
for i in tqdm.tqdm(range(0,len(total_data))):
    total_data['h2'].iloc[i]=(total_data['middle_breast_thickness'].iloc[i] - total_data['bottom_breast_thickness'].iloc[i])

# 반지름 컬럼(r1~r4), 높이컬럼 소수점 둘째자리로 반올림 (단위:cm)
r_columns =['r1','r2','r3','r4','h1','h2']
for i in tqdm.tqdm(range(0,len(r_columns))):
    total_data[r_columns[i]] = round(total_data[r_columns[i]],2) # 소수점 둘째자리로 반올림 (단위 : cm)

# 유방 1개 원주 계산 : breast_ellipse_circum(유방_타원_원주)
total_data['circum_01']=0 # 1) r1, r2
total_data['circum_02']=0 # 2) r2, r3
total_data['circum_03']=0 # 3) r3, r4
total_data['circum_04']=0 # 4) r1, r4
total_data['breast_ellipse_circum']=0 # 유방 1개 원주

for i in tqdm.tqdm(range(0,len(total_data))):
    total_data['circum_01'].iloc[i]=(0.25*1.5*3.14)*((total_data['r1'].iloc[i]+total_data['r2'].iloc[i])-((total_data['r1'].iloc[i]*total_data['r2'].iloc[i])**0.5))
    total_data['circum_02'].iloc[i]=(0.25*1.5*3.14)*((total_data['r2'].iloc[i]+total_data['r3'].iloc[i])-((total_data['r2'].iloc[i]*total_data['r3'].iloc[i])**0.5))
    total_data['circum_03'].iloc[i]=(0.25*1.5*3.14)*((total_data['r3'].iloc[i]+total_data['r4'].iloc[i])-((total_data['r3'].iloc[i]*total_data['r4'].iloc[i])**0.5))
    total_data['circum_04'].iloc[i]=(0.25*1.5*3.14)*((total_data['r4'].iloc[i]+total_data['r1'].iloc[i])-((total_data['r4'].iloc[i]*total_data['r1'].iloc[i])**0.5))
    total_data['breast_ellipse_circum'].iloc[i]=total_data['circum_01'].iloc[i]+total_data['circum_02'].iloc[i]+total_data['circum_03'].iloc[i]+total_data['circum_04'].iloc[i]
    total_data['breast_ellipse_circum'].iloc[i]=round(total_data['breast_ellipse_circum'].iloc[i],2)
total_data = total_data.drop(['circum_01','circum_02','circum_03','circum_04'],axis=1)
    
# 유방 1개 평균 반지름 계산 : r(평균r)
# 반지름(r) = 원주(circumference)/2π
total_data['r']=0
for i in tqdm.tqdm(range(0,len(total_data))):
    total_data['r'].iloc[i]=total_data['breast_ellipse_circum'].iloc[i]/6.284

# 유방 1개 평균 높이 계산 : h(평균h)
total_data['h']=0
for i in tqdm.tqdm(range(0,len(total_data))):
    total_data['h'].iloc[i]=(total_data['h1'].iloc[i]+total_data['h2'].iloc[i])/2

# 유방 평균 반지름, 높이 -> 하나의 구 형태로 볼륨계산 : volume(유방_볼륨_계산 - 타원구)
total_data['volume']=0
for i in tqdm.tqdm(range(0,len(total_data))):
    total_data['volume'].iloc[i]=(4*3.142*(total_data['r'].iloc[i]**2)*total_data['h'].iloc[i])/3

# 현재 밴드 사이즈 5단위 재범주화(후크 정보 반영 전)
for i in tqdm.tqdm(range(0, len(total_data))):
    if total_data['band_size_now'].iloc[i]<=70.0:
        total_data['band_size_now'].iloc[i]=70
    elif total_data['band_size_now'].iloc[i]>70.0 and total_data['band_size_now'].iloc[i]<=75.0:
        total_data['band_size_now'].iloc[i]=75
    elif total_data['band_size_now'].iloc[i]>75.0 and total_data['band_size_now'].iloc[i]<=80.0:
        total_data['band_size_now'].iloc[i]=80
    elif total_data['band_size_now'].iloc[i]>80.0 and total_data['band_size_now'].iloc[i]<=85.0:
        total_data['band_size_now'].iloc[i]=85
    elif total_data['band_size_now'].iloc[i]>85.0:
        total_data['band_size_now'].iloc[i]=90
print(total_data['band_size_now'].value_counts())
# 반지름 반올림
total_data['r'] = round(total_data['r'],2)
# volume 정수화
total_data['volume'] = round(total_data['volume'],0)
print(total_data.columns)

#print(total_data['r'].value_counts())
#print(total_data['h'].value_counts())
#print(total_data['middle_breast_width'].value_counts())
print(total_data.head())
# total_data.to_excel('data/qna_data/0131_설문_volumne.xlsx', index=False)

## 현재 질문 반영 제품 추천 로드맵
# anaconda_band_size(더핏브라 맞춤 밴드 사이즈) 컬럼 생성
total_data['anaconda_band_size'] = 0
# anaconda_cup_size(맞춤 컵 사이즈) 컬럼 생성
total_data['anaconda_cup_size'] = 0


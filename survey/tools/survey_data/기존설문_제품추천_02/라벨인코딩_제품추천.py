import numpy as np
import pandas as pd
import os
import sys
import tqdm

# 수치 변환 설문 데이터 불러오기 : 만족도 설문 한 사람/안한 사람 존재
# 전체 데이터 : 4,428개
data = pd.read_excel('data/qna_data/qna_기존_수치변환.xlsx')
data.info()

# 컬럼 순서 변경
data = data[['height', 'weight', 'BMI', 'wai_cir', 'band_size_now',
'cup_size_now', 'hook_num_now', 'breast_space_fingers',
'good_bra_score_cup', 'bra_cup_bad_point', 'good_bra_score_band',
'bra_band_bad_point', 'breast_shape', 'breast_gol_shape']]

## 제품 추천 진행 : 모두 존재 3,713명
data = data.dropna(axis=0)
data.info()
print(data.columns)

print(data['band_size_now'].value_counts())
# 현재 밴드 사이즈 5단위 재범주화(후크 정보 반영 전)
for i in tqdm.tqdm(range(0, len(data))):
    if data['band_size_now'].iloc[i]<=70.0:
        data['band_size_now'].iloc[i]=70
    elif data['band_size_now'].iloc[i]>70.0 and data['band_size_now'].iloc[i]<=75.0:
        data['band_size_now'].iloc[i]=75
    elif data['band_size_now'].iloc[i]>75.0 and data['band_size_now'].iloc[i]<=80.0:
        data['band_size_now'].iloc[i]=80
    elif data['band_size_now'].iloc[i]>80.0 and data['band_size_now'].iloc[i]<=85.0:
        data['band_size_now'].iloc[i]=85
    elif data['band_size_now'].iloc[i]>85.0:
        data['band_size_now'].iloc[i]=90
print(data['band_size_now'].value_counts())

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
print(data['cup_size_now'].value_counts())


## 현재 질문 반영 제품 추천 로드맵
# custom_product(맞춤 제품명) 컬럼 생성
data['custom_product'] = 0
# custom_band_size(맞춤 밴드 사이즈) 컬럼 생성
data['custom_band_size'] = 0
# custom_cup_size(맞춤 컵 사이즈) 컬럼 생성
data['custom_cup_size'] = 0

for i in tqdm.tqdm(range(0, len(data))):
    if data['breast_shape'].iloc[i]==4 or data['breast_shape'].iloc[i]==5 or data['breast_shape'].iloc[i]==6:
        if data['breast_space_fingers'].iloc[i]==3 or data['breast_space_fingers'].iloc[i]==4 or data['breast_space_fingers'].iloc[i]==5:
            if data['bra_cup_bad_point'].iloc[i]==1:
                data['custom_product'].iloc[i]='[도로시와] 노와이어 풀샷 브라'
                data['custom_band_size'].iloc[i]=data['band_size_now'].iloc[i]+5
                data['custom_cup_size'].iloc[i]=data['cup_size_now'].iloc[i]
            elif data['bra_cup_bad_point'].iloc[i]==2:
                data['custom_product'].iloc[i]='[아나콘다] 더핏브라'
                data['custom_band_size'].iloc[i]=data['band_size_now'].iloc[i]+5
                data['custom_cup_size'].iloc[i]=data['cup_size_now'].iloc[i]
            elif data['bra_cup_bad_point'].iloc[i]==3:
                data['custom_product'].iloc[i]='[해즈소울] 소울라이트'
                data['custom_band_size'].iloc[i]=data['band_size_now'].iloc[i]+10
                data['custom_cup_size'].iloc[i]=data['cup_size_now'].iloc[i]-2
        elif data['breast_space_fingers'].iloc[i]==0 or data['breast_space_fingers'].iloc[i]==1 or data['breast_space_fingers'].iloc[i]==2:
            data['custom_product'].iloc[i]='[에메필] 소프트 초모리 브라'
            data['custom_band_size'].iloc[i]=data['band_size_now'].iloc[i]
            data['custom_cup_size'].iloc[i]=data['cup_size_now'].iloc[i]+1
    elif data['breast_shape'].iloc[i]==1 or data['breast_shape'].iloc[i]==2 or data['breast_shape'].iloc[i]==3:
        if data['breast_space_fingers'].iloc[i]==3 or data['breast_space_fingers'].iloc[i]==4 or data['breast_space_fingers'].iloc[i]==5:
            if data['bra_cup_bad_point'].iloc[i]==1:
                data['custom_product'].iloc[i]='[도로시와] 볼륨메이커'
                data['custom_band_size'].iloc[i]=data['band_size_now'].iloc[i]+10
                data['custom_cup_size'].iloc[i]=data['cup_size_now'].iloc[i]-2
            else:
                data['custom_product'].iloc[i]='[속삭끄] 웨이브 브라'
                data['custom_band_size'].iloc[i]=data['band_size_now'].iloc[i]+10
                data['custom_cup_size'].iloc[i]=data['cup_size_now'].iloc[i]
        elif data['breast_space_fingers'].iloc[i]==0 or data['breast_space_fingers'].iloc[i]==1 or data['breast_space_fingers'].iloc[i]==2:
            data['custom_product'].iloc[i]='[에메필] 하프컵 초모리 브라'
            data['custom_band_size'].iloc[i]=data['band_size_now'].iloc[i]
            data['custom_cup_size'].iloc[i]=data['cup_size_now'].iloc[i]+1
            

# 예외 처리          
for i in tqdm.tqdm(range(0, len(data))):
    # 밴드 사이즈 90 이상 -> 90
    if data['custom_band_size'].iloc[i]>=90: 
        data['custom_band_size'].iloc[i]=90
    # 컵 사이즈 AA 미만 -> AA
    if data['custom_cup_size'].iloc[i]<1:
        data['custom_cup_size'].iloc[i]=1
    # 컵 사이즈 F 초과 -> F
    if data['custom_cup_size'].iloc[i]>7:
        data['custom_cup_size'].iloc[i]=7
        
print("컵 사이즈 문자형 변환")
print("1: AA, 2: A, 3: B, 4: C, 5: D, 6: E, 7: F")
for i in tqdm.tqdm(range(0, len(data))):
    if data['custom_cup_size'].iloc[i]==1:
        data['custom_cup_size'].iloc[i] = 'AA'
    elif data['custom_cup_size'].iloc[i]==2:
        data['custom_cup_size'].iloc[i] = 'A'
    elif data['custom_cup_size'].iloc[i]==3:
        data['custom_cup_size'].iloc[i] = 'B'
    elif data['custom_cup_size'].iloc[i]==4:
        data['custom_cup_size'].iloc[i] = 'C'
    elif data['custom_cup_size'].iloc[i]==5:
        data['custom_cup_size'].iloc[i] = 'D'
    elif data['custom_cup_size'].iloc[i]==6:
        data['custom_cup_size'].iloc[i] = 'E'
    elif data['custom_cup_size'].iloc[i]==7:
        data['custom_cup_size'].iloc[i] = 'F'
        
# custom_size(맞춤 사이즈) 컬럼 생성
data['custom_size'] = 0
for i in tqdm.tqdm(range(0, len(data))):
    data['custom_size'].iloc[i] = str(data['custom_band_size'].iloc[i])+data['custom_cup_size'].iloc[i]
print(data['custom_size'].value_counts())

data.head()
#data.to_excel('data/qna_data/설문_제품_추천.xlsx', index=False)
import numpy as np
import pandas as pd
import os
import sys
import tqdm

## 가공 설문데이터 셋 생성

# 수치 변환 설문 데이터 불러오기 : 만족도 설문 한 사람/안한 사람 존재
# 전체 데이터 : 4,428개
data = pd.read_excel('data/qna_data/qna_기존_수치변환.xlsx')
data.info()

# 컬럼 순서 변경
data = data[['height', 'weight', 'BMI', 'wai_cir', 'band_size_now',
'cup_size_now', 'hook_num_now', 'breast_space_fingers',
'good_bra_score_cup', 'bra_cup_bad_point', 'good_bra_score_band',
'bra_band_bad_point', 'breast_shape', 'breast_gol_shape']]

# 만족도 O 데이터 : 3,716개, 3개 데이터 결측 column 존재
# -> 총 3,713개 선별
data = data.dropna()
data.info()

# 컵 점수, 밴드 점수 둘다 7이상 데이터 선별 : 1,274명 데이터
# 컵 점수, 밴드 점수 7 미만 데이터 삭제
data = data[data['good_bra_score_cup']>=7]
data = data[data['good_bra_score_band']>=7]

# 이상치 제거 (허리둘레 이상치 : 14명, BMI 이상치 : 12명 제거)
data = data[data['wai_cir']>50.8]
data = data[data['BMI']>16]

# 데이터 저장
data.to_excel('data/qna_data/qna_기존_수치변환_7점이상_이상치_제거.xlsx')

# 사이즈 코리아 데이터 불러오기 : 여성 2,510명
sz_kr = pd.read_excel('data/size_korea_new/최종_볼륨_계산 데이터셋.xlsx')
sz_kr.info()

# 사이즈코리아 데이터 이상치 2명 제거 : 젖가슴 둘레 < 젖가슴 아래 둘레
# 이상치 2명 인덱스 outlier_idx 리스트에 추가
outlier_idx=[]
for i in tqdm.tqdm(range(0, len(sz_kr))):
    if sz_kr['젖_가슴_아래_둘레'].iloc[i]>sz_kr['젖_가슴_둘레'].iloc[i]:
       print('삭제 필요 인덱스 : ',i)
       outlier_idx.append(i)
# outlier_idx 인덱스 2명 사이즈코리아 데이터에서 제거 : 2,510명 -> 2,508명
for i in tqdm.tqdm(range(0, len(outlier_idx))):
    sz_kr = sz_kr.drop(i)

# 만족도 7이상 1,274명
# 1) 키 -> 겨드랑 높이, 젖가슴 높이, 젖가슴 아래 높이 : 292개 항목
# pt1_height : 사이즈 코리아 [키 - 겨드랑 높이/젖가슴 높이/젖가슴 아래 높이] 피벗 테이블
pt1_height = sz_kr.pivot_table(['겨드랑_높이','젖_가슴_높이','젖_가슴_아래_높이'],index=['키'])
pt1_height.reset_index(inplace=True)
print(pt1_height)
pt1_height.info()
df_OUTER_JOIN = pd.merge(data, pt1_height, left_on='height', right_on='키', how='left')
print(df_OUTER_JOIN)
print(df_OUTER_JOIN.info())

# 2) 사이즈코리아 허리둘레 기준 환산 젖가슴 둘레 환산
pt2 = sz_kr.pivot_table(['젖_가슴_둘레'],index=['젖_가슴_아래_둘레'])
pt2.reset_index(inplace=True)
print(pt2.columns)
pt2.info()
df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt2, left_on='band_size_now', right_on='젖_가슴_아래_둘레', how='left')
df_OUTER_JOIN['젖_가슴_둘레'] = df_OUTER_JOIN['젖_가슴_둘레'].round(1) # 소수점 첫째자리 반올림
print(df_OUTER_JOIN)

# 3-1) 설문데이터 기준 젖가슴 아래둘레 환산
#df_OUTER_JOIN['젖가슴_아래_둘레_설문'] = 0
#for i in tqdm.tqdm(range(0, len(data))):
#        df_OUTER_JOIN['젖가슴_아래_둘레_설문'].iloc[i] = df_OUTER_JOIN['band_size_now'].iloc[i]
#print(df_OUTER_JOIN)

# 3-2) 젖가슴 둘레 기준 젖가슴 아래둘레 환산
#pt3 = sz_kr.pivot_table(['젖_가슴_아래_둘레'],index=['젖_가슴_둘레'])
#pt3.reset_index(inplace=True)
#print(pt3)
#pt3.info()
#df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt3, left_on='젖_가슴_둘레', right_on='젖_가슴_둘레', how='left')
#df_OUTER_JOIN['젖_가슴_아래_둘레'] = df_OUTER_JOIN['젖_가슴_아래_둘레'].round(1) # 소수점 첫째자리 반올림

# 4) 젖가슴 둘레 기준 젖가슴 너비 환산
pt4 = sz_kr.pivot_table(['젖_가슴_너비'],index=['젖_가슴_둘레'])
pt4.reset_index(inplace=True)
print(pt4)
pt4.info()
df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt4, left_on='젖_가슴_둘레', right_on='젖_가슴_둘레', how='left')
df_OUTER_JOIN['젖_가슴_너비'] = df_OUTER_JOIN['젖_가슴_너비'].round(1) # 소수점 첫째자리 반올림

# 5) 젖가슴 둘레 기준 젖꼭지 사이 수평길이 환산
pt5 = sz_kr.pivot_table(['젖꼭지_사이_수평_길이'],index=['젖_가슴_둘레'])
pt5.reset_index(inplace=True)
print(pt5)
pt5.info()
df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt5, left_on='젖_가슴_둘레', right_on='젖_가슴_둘레', how='left')
df_OUTER_JOIN['젖꼭지_사이_수평_길이'] = df_OUTER_JOIN['젖꼭지_사이_수평_길이'].round(1) # 소수점 첫째자리 반올림

# 6) 젖가슴 둘레 기준 가슴 두께 환산
pt6 = sz_kr.pivot_table(['가슴_두께'],index=['젖_가슴_둘레'])
pt6.reset_index(inplace=True)
print(pt6)
pt6.info()
df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt6, left_on='젖_가슴_둘레', right_on='젖_가슴_둘레', how='left')
df_OUTER_JOIN['가슴_두께'] = df_OUTER_JOIN['가슴_두께'].round(1) # 소수점 첫째자리 반올림

# 7) 젖가슴 둘레 기준 젖가슴 두께 환산
pt7 = sz_kr.pivot_table(['젖_가슴_두께'],index=['젖_가슴_둘레'])
pt7.reset_index(inplace=True)
print(pt7)
pt7.info()
df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt7, left_on='젖_가슴_둘레', right_on='젖_가슴_둘레', how='left')
df_OUTER_JOIN['젖_가슴_두께'] = df_OUTER_JOIN['젖_가슴_두께'].round(1) # 소수점 첫째자리 반올림

# 8) 젖가슴 둘레 기준 가슴 두께 환산
pt8 = sz_kr.pivot_table(['젖_가슴_아래_두께'],index=['젖_가슴_둘레'])
pt8.reset_index(inplace=True)
print(pt8)
pt8.info()
df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt8, left_on='젖_가슴_둘레', right_on='젖_가슴_둘레', how='left')
df_OUTER_JOIN['젖_가슴_아래_두께'] = df_OUTER_JOIN['젖_가슴_아래_두께'].round(1) # 소수점 첫째자리 반올림

print(df_OUTER_JOIN)
print(df_OUTER_JOIN.info())
print(df_OUTER_JOIN.columns)
 
# 전체 열 이름 변경
df_OUTER_JOIN.columns = ['height', 'weight', 'BMI', 'wai_cir', 'band_size_now', 'cup_size_now',
       'hook_num_now', 'breast_space_fingers', 'good_bra_score_cup',
       'bra_cup_bad_point', 'good_bra_score_band', 'bra_band_bad_point',
       'breast_shape', 'breast_gol_shape', '키', 'armpit_height', 'middle_breast_height',
       'bottom_breast_height', '허리_둘레', 'middle_breast_round', 'bottom_breast_round_설문', 'bottom_breast_round_사이즈코리아',
       'middle_breast_width', 'nipple_between_length', 'top_breast_thickness', 'middle_breast_thickness', 'bottom_breast_thickness']

# 컬럼 순서 변경
total_data = df_OUTER_JOIN[['height', 'armpit_height', 'middle_breast_height',
       'bottom_breast_height', 'weight', 'BMI', 'wai_cir', 'band_size_now', 'bottom_breast_round_설문', 'bottom_breast_round_사이즈코리아',
       'middle_breast_round','middle_breast_width', 'nipple_between_length', 'top_breast_thickness', 'middle_breast_thickness', 'bottom_breast_thickness',
       'cup_size_now','hook_num_now', 'breast_space_fingers', 'good_bra_score_cup',
       'bra_cup_bad_point', 'good_bra_score_band', 'bra_band_bad_point',
       'breast_shape', 'breast_gol_shape']]
print(total_data.info())
#total_data.to_excel('data/qna_data/가공_설문_데이터.xlsx',index=False)



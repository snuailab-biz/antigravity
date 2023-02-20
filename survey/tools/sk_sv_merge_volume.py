import numpy as np
import pandas as pd
import os
import sys
import tqdm
from sklearn.model_selection import train_test_split

## [데이터]
## [안티그래비티] 설문 데이터 = data
## [사이즈코리아] 선별 2,510명 여성 치수 데이터 = sz_kr (사이즈코리아 14개 컬럼, 8개 계산 컬럼)
## [코드]
## 1) 사이즈코리아 - 설문 데이터 연동
## 2) 연동 데이터 여성 유방 볼륨 계산 (여성 유방 = 반 타원구 2개 -> 타원구 부피 계산 공식)

# 사이즈코리아 데이터 이상치 제거 : 젖가슴 둘레 < 젖가슴 아래 둘레
# 이상치 인원 인덱스 outlier_idx 리스트에 추가
# 사이즈코리아 = 키 소수점 단위로 존재 -> 예외 존재 -> 0.1cm 예외처리
def sz_kr_remove_outlier(sz_kr):
    print("사이즈코리아 데이터 이상치 제거 : '젖가슴 둘레 < 젖가슴 아래 둘레'인 경우")
    outlier_idx=[]
    for i in tqdm.tqdm(range(0, len(sz_kr))):
        if sz_kr['젖_가슴_아래_둘레'].iloc[i]>sz_kr['젖_가슴_둘레'].iloc[i]:
            outlier_idx.append(i)
    # outlier_idx 인덱스 존재 인원 사이즈코리아 데이터에서 제거
    print("이상치 {}명 수치 계산 데이터 제거".format(len(outlier_idx)))
    for i in tqdm.tqdm(range(0, len(outlier_idx))):
        sz_kr = sz_kr.drop(i)
    # 사이즈코리아 = 키 소수점 단위로 존재 -> 예외 존재 -> 0.1cm 예외처리
    print("키 소수점 단위로 존재 -> 예외 존재 -> 0.1cm 예외처리")
    for i in tqdm.tqdm(range(0, len(sz_kr))):
        if sz_kr['키'].iloc[i]==172.1:
            sz_kr['키'].iloc[i]=172
        elif sz_kr['키'].iloc[i]==173.8:
            sz_kr['키'].iloc[i]=174
        elif sz_kr['키'].iloc[i]==181:
            sz_kr['키'].iloc[i]=180
    return sz_kr

# 설문 데이터 - 사이즈코리아 8개 컬럼 연동
def survey_sizekorea(data, sz_kr):
    print("----설문-사이즈코리아 통합 데이터 생성을 시작합니다----")
    # 1) 키 -> 겨드랑 높이, 젖가슴 높이, 젖가슴 아래 높이 : 292개 항목    
    # pt1_height : 사이즈 코리아 [키 - 겨드랑 높이/젖가슴 높이/젖가슴 아래 높이] 피벗 테이블
    print("1) 키 -> 겨드랑 높이, 젖가슴 높이, 젖가슴 아래 높이 : 292개 항목")
    pt1_height = sz_kr.pivot_table(['겨드랑_높이','젖_가슴_높이','젖_가슴_아래_높이'],index=['키'])
    pt1_height.reset_index(inplace=True)
    print(pt1_height)
    df_OUTER_JOIN = pd.merge(data, pt1_height, left_on='height', right_on='키', how='left')
    print(df_OUTER_JOIN.head())

    # 2) 사이즈코리아 젖가슴 아래 기준 환산 젖가슴 둘레 환산
    print("2) 사이즈코리아 젖가슴 아래 기준 환산 젖가슴 둘레 환산")
    pt2 = sz_kr.pivot_table(['젖_가슴_둘레'],index=['젖_가슴_아래_둘레'])
    pt2.reset_index(inplace=True)
    print(pt2)
    df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt2, left_on='band_size_now', right_on='젖_가슴_아래_둘레', how='left')
    df_OUTER_JOIN['젖_가슴_둘레'] = df_OUTER_JOIN['젖_가슴_둘레'].round(1) # 소수점 첫째자리 반올림
    print(df_OUTER_JOIN.head())
        
    # 3) 젖가슴 둘레 기준 젖가슴 너비 환산
    pt3 = sz_kr.pivot_table(['젖_가슴_너비'],index=['젖_가슴_둘레'])
    pt3.reset_index(inplace=True)
    print(pt3)
    pt3.info()
    df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt3, left_on='젖_가슴_둘레', right_on='젖_가슴_둘레', how='left')
    df_OUTER_JOIN['젖_가슴_너비'] = df_OUTER_JOIN['젖_가슴_너비'].round(1) # 소수점 첫째자리 반올림
    print(df_OUTER_JOIN.head())

    # 4) 젖가슴 둘레 기준 젖꼭지 사이 수평길이 환산
    pt4 = sz_kr.pivot_table(['젖꼭지_사이_수평_길이'],index=['젖_가슴_너비'])
    pt4.reset_index(inplace=True)
    print(pt4)
    pt4.info()
    df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt4, left_on='젖_가슴_너비', right_on='젖_가슴_너비', how='left')
    df_OUTER_JOIN['젖꼭지_사이_수평_길이'] = df_OUTER_JOIN['젖꼭지_사이_수평_길이'].round(1) # 소수점 첫째자리 반올림
    print(df_OUTER_JOIN.head())

    # 5) 젖가슴 둘레 기준 가슴 두께 환산
    pt5 = sz_kr.pivot_table(['가슴_두께'],index=['젖_가슴_둘레'])
    pt5.reset_index(inplace=True)
    print(pt5)
    pt5.info()
    df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt5, left_on='젖_가슴_둘레', right_on='젖_가슴_둘레', how='left')
    df_OUTER_JOIN['가슴_두께'] = df_OUTER_JOIN['가슴_두께'].round(1) # 소수점 첫째자리 반올림
    print(df_OUTER_JOIN.head())

    # 6) 젖가슴 둘레 기준 젖가슴 두께 환산
    pt6 = sz_kr.pivot_table(['젖_가슴_두께'],index=['젖_가슴_둘레'])
    pt6.reset_index(inplace=True)
    print(pt6)
    pt6.info()
    df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt6, left_on='젖_가슴_둘레', right_on='젖_가슴_둘레', how='left')
    df_OUTER_JOIN['젖_가슴_두께'] = df_OUTER_JOIN['젖_가슴_두께'].round(1) # 소수점 첫째자리 반올림
    print(df_OUTER_JOIN.head())

    # 7) 젖가슴 둘레 기준 가슴 두께 환산
    pt7 = sz_kr.pivot_table(['젖_가슴_아래_두께'],index=['젖_가슴_둘레'])
    pt7.reset_index(inplace=True)
    print(pt7)
    pt7.info()
    df_OUTER_JOIN = pd.merge(df_OUTER_JOIN, pt7, left_on='젖_가슴_둘레', right_on='젖_가슴_둘레', how='left')
    df_OUTER_JOIN['젖_가슴_아래_두께'] = df_OUTER_JOIN['젖_가슴_아래_두께'].round(1) # 소수점 첫째자리 반올림
    print(df_OUTER_JOIN.head())

    # 조인 데이터 전체 열 이름 변경
    df_OUTER_JOIN.columns = ['height', 'weight', 'BMI', 'wai_cir', 'band_size_now', 'cup_size_now',
        'hook_num_now', 'breast_space_fingers', 'good_bra_score_cup',
        'bra_cup_bad_point', 'good_bra_score_band', 'bra_band_bad_point',
        'breast_shape', 'breast_gol_shape', '키', 'armpit_height', 'middle_breast_height',
        'bottom_breast_height', 'bottom_breast_round', 'middle_breast_round',
        'middle_breast_width', 'nipple_between_length', 'top_breast_thickness', 'middle_breast_thickness', 'bottom_breast_thickness']

    # 컬럼 순서 변경
    total_data = df_OUTER_JOIN[['height', 'armpit_height', 'middle_breast_height',
        'bottom_breast_height', 'weight', 'BMI', 'wai_cir', 'band_size_now', 'bottom_breast_round',
        'middle_breast_round','middle_breast_width', 'nipple_between_length', 'top_breast_thickness',
        'middle_breast_thickness', 'bottom_breast_thickness', 'cup_size_now','hook_num_now',
        'breast_space_fingers', 'good_bra_score_cup','bra_cup_bad_point',
        'good_bra_score_band', 'bra_band_bad_point','breast_shape', 'breast_gol_shape']]

    # 예외 null값 치환
    total_data["armpit_height"].fillna(119.4, inplace=True)
    total_data["middle_breast_height"].fillna(114.1, inplace=True)
    total_data["bottom_breast_height"].fillna(109.3, inplace=True)
    total_data["middle_breast_round"].fillna(95.8, inplace=True)
    total_data["middle_breast_width"].fillna(33, inplace=True)
    total_data["nipple_between_length"].fillna(17, inplace=True)
    total_data["top_breast_thickness"].fillna(21.5, inplace=True)
    total_data["middle_breast_thickness"].fillna(25.8, inplace=True)
    total_data["bottom_breast_thickness"].fillna(23.5, inplace=True)

    # 젖가슴 아래 둘레 : 밴드 사이즈로 재환산
    total_data['bottom_breast_round'] = total_data['band_size_now']
    print("----설문-사이즈코리아 통합 데이터 생성이 완료되었습니다----")
    print(total_data.head())
    
    # 열 이름 및 순서 변경 데이터 선별
    return total_data

# 반지름(r1 ~ r4), 깊이(h1,h2) -> 여성 유방 볼륨 계산
def volume_calculation(total_data): 
    print("----통합 데이터 유방 볼륨 계산을 시작합니다----")
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
    #total_data['r'] = round(total_data['r'],3) # 소수점 셋째자리로 반올림 (단위 : cm)

    # 유방 1개 평균 높이 계산 : h(평균h)
    total_data['h']=0
    for i in tqdm.tqdm(range(0,len(total_data))):
        total_data['h'].iloc[i]=(total_data['h1'].iloc[i]+total_data['h2'].iloc[i])/2
    #total_data['h'] = round(total_data['h'],3) # 소수점 셋째자리로 반올림 (단위 : cm)
    print("----통합 데이터 유방 볼륨 관련 수치 계산이 완료되었습니다----")
    
    return total_data

def train_test_data_split(total_data):
    # 유방 평균 반지름, 높이 -> 하나의 구 형태로 볼륨계산 : volume(유방_볼륨_계산 - 타원구)
    total_data['volume']=0
    for i in tqdm.tqdm(range(0,len(total_data))):
        total_data['volume'].iloc[i]=(4*3.142*(total_data['r'].iloc[i]**2)*total_data['h'].iloc[i])/3
    print("----통합 데이터 유방 볼륨 계산이 완료되었습니다----")
    
    # ★ train data : 컵/밴드 만족도 7점 이상 데이터 선별 ★
    # 컵 점수, 밴드 점수 둘 다 7이상 데이터 선별 : 1,248명 데이터
    print("----컵/밴드 만족도 7점 이상 데이터로 Train/Test 데이터 분리를 시작합니다----")
    total_data = total_data[total_data['good_bra_score_cup']>=7]
    total_data = total_data[total_data['good_bra_score_band']>=7]
    
    # 결측치 행 삭제 후 Train/Test 데이터 분리 및 학습 진행 예정
    print("----결측치 행 삭제 후 Train/Test 데이터 분리 및 학습 진행합니다----")
    original_length = len(total_data)
    total_data = total_data.dropna()
    print(total_data.info())
    
    # 고객 유방 볼륨(volume) 수치 정수로 반올림
    print("----고객 유방 볼륨(volume) 수치 정수로 반올림 진행합니다----")
    total_data['volume'] = total_data['volume'].astype('int') # 정수로 반올림
    print("----",(original_length-len(total_data)),"개 결측 데이터 삭제 후 Train/Test 데이터 분리 진행----")
    
    X = total_data[['height', 'armpit_height', 'middle_breast_height',
       'bottom_breast_height', 'weight', 'BMI', 'wai_cir', 'band_size_now',
       'bottom_breast_round', 'middle_breast_round', 'middle_breast_width',
       'nipple_between_length', 'top_breast_thickness',
       'middle_breast_thickness', 'bottom_breast_thickness', 'cup_size_now',
       'hook_num_now', 'breast_space_fingers', 'good_bra_score_cup',
       'bra_cup_bad_point', 'good_bra_score_band', 'bra_band_bad_point',
       'breast_shape', 'breast_gol_shape', 'r1', 'r2', 'r3', 'r4', 'h1', 'h2',
       'breast_ellipse_circum', 'r', 'h']]
    y = total_data[['volume']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=550)
    
    print("----컵/밴드 만족도 7점 이상 데이터로 Train/Test 데이터 분리가 완료 되었습니다----")
    return X_train, X_test, y_train, y_test
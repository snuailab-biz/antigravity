import numpy as np
import pandas as pd
import os
import sys
import tqdm

## [안티그래비티] 설문 데이터 = data

# 설문 데이터 불러오기 : 만족도 설문 한 사람/안한 사람 존재
# 설문데이터 기존 컬럼 컬럼명 변경
def kor2eng(data):
    data.columns = ['band_size_now', 'cup_size_now', 'hook_num_now', 'breast_space_fingers',
    'good_bra_score_cup', 'bra_cup_bad_point', 'good_bra_score_band',
    'bra_band_bad_point', 'breast_shape', 'breast_gol_shape', 'height', 'weight', 'wai_cir']
    return data

# 설문데이터 기존 컬럼 데이터 전처리 : 수치형 변환 및 일부 계산
def text2int(data):
    # 1) 후크 번호(hook_num_now) : 정수형 변환
    # '1번(가장 안쪽)' : 1, '2번(중간)' : 2, '3번(가장 바깥쪽)' : 3 변환
    print("1) 후크 번호(hook_num_now) : 정수형 변환")
    print("'1번(가장 안쪽)' : 1, '2번(중간)' : 2, '3번(가장 바깥쪽)' : 3 변환")
    for i in tqdm.tqdm(range(0, len(data))):
        if data['hook_num_now'].iloc[i]=='1번(가장 안쪽)':
            data['hook_num_now'].iloc[i] = 1
        elif data['hook_num_now'].iloc[i]=='2번(중간)':
            data['hook_num_now'].iloc[i] = 2
        elif data['hook_num_now'].iloc[i]=='3번(가장 바깥쪽)':
            data['hook_num_now'].iloc[i] = 3
            
    # 2) 후크 번호 -> 젖가슴아랫둘레(band_size_now) 추가 계산
    # 후크번호 1 : -1.2cm / 2 : 0 / 3 : +1.2cm 
    print("2) 후크 번호 -> 젖가슴아랫둘레(band_size_now) 추가 계산")
    print("후크번호 1 : -1.2cm / 2 : 0 / 3 : +1.2cm")
    for i in tqdm.tqdm(range(0, len(data))):
        if data['hook_num_now'].iloc[i]==1:
            data['band_size_now'].iloc[i] = data['band_size_now'].iloc[i]-1.2
        elif data['hook_num_now'].iloc[i]==2:
            data['band_size_now'].iloc[i] = data['band_size_now'].iloc[i]
        elif data['hook_num_now'].iloc[i]==3:
            data['band_size_now'].iloc[i] = data['band_size_now'].iloc[i]+1.2

    # 3) 유방 간 간격(breast_space_fingers) : 정수형 변환
    # '거의 붙어있음' : 0, '손가락 1개 정도' : 1, '손가락 2개 정도' : 2, '손가락 3개 정도' : 3, '손가락 4개 정도' : 4, '손가락 5개 이상' : 5
    print("3) 유방 간 간격(breast_space_fingers) : 정수형 변환")
    print("'거의 붙어있음' : 0, '손가락 1개 정도' : 1, '손가락 2개 정도' : 2, '손가락 3개 정도' : 3, '손가락 4개 정도' : 4, '손가락 5개 이상' : 5")
    for i in tqdm.tqdm(range(0, len(data))):
        if data['breast_space_fingers'].iloc[i]=='거의 붙어있음':
            data['breast_space_fingers'].iloc[i] = 0
        elif data['breast_space_fingers'].iloc[i]=='손가락 1개 정도':
            data['breast_space_fingers'].iloc[i] = 1
        elif data['breast_space_fingers'].iloc[i]=='손가락 2개 정도':
            data['breast_space_fingers'].iloc[i] = 2
        elif data['breast_space_fingers'].iloc[i]=='손가락 3개 정도':
            data['breast_space_fingers'].iloc[i] = 3
        elif data['breast_space_fingers'].iloc[i]=='손가락 4개 정도':
            data['breast_space_fingers'].iloc[i] = 4
        elif data['breast_space_fingers'].iloc[i]=='손가락 5개 이상':
            data['breast_space_fingers'].iloc[i] = 5

    # 4) 컵 불편한 점(bra_cup_bad_point) : 정수형 변환
    # '패드가 작아요' : 1, '적당하며, 괜찮아요' : 2, '패드가 커요' : 3 변환
    print("4) 컵 불편한 점(bra_cup_bad_point) : 정수형 변환")
    print("'패드가 작아요' : 1, '적당하며, 괜찮아요' : 2, '패드가 커요' : 3 변환")
    for i in tqdm.tqdm(range(0, len(data))):
        if data['bra_cup_bad_point'].iloc[i]=='패드가 작아요':
            data['bra_cup_bad_point'].iloc[i] = 1
        elif data['bra_cup_bad_point'].iloc[i]=='적당하며, 괜찮아요':
            data['bra_cup_bad_point'].iloc[i] = 2
        elif data['bra_cup_bad_point'].iloc[i]=='패드가 커요':
            data['bra_cup_bad_point'].iloc[i] = 3

    # 5) 밴드 불편한 점(bra_band_bad_point) : 정수형 변환
    # '밴드가 압박감을 주거나, 주변으로 살이 튀어나와요.' : 1, '적당하며, 괜찮아요' : 2, '밴드가 등 위로 올라가거나 헐렁해요' : 3" 변환
    print("5) 밴드 불편한 점(bra_band_bad_point) : 정수형 변환")
    print("'밴드가 압박감을 주거나, 주변으로 살이 튀어나와요.' : 1, '적당하며, 괜찮아요' : 2, '밴드가 등 위로 올라가거나 헐렁해요' : 3")
    for i in tqdm.tqdm(range(0, len(data))):
        if data['bra_band_bad_point'].iloc[i]=='밴드가 압박감을 주거나, 주변으로 살이 튀어나와요.':
            data['bra_band_bad_point'].iloc[i] = 1
        elif data['bra_band_bad_point'].iloc[i]=='적당하며, 괜찮아요':
            data['bra_band_bad_point'].iloc[i] = 2
        elif data['bra_band_bad_point'].iloc[i]=='밴드가 등 위로 올라가거나 헐렁해요':
            data['bra_band_bad_point'].iloc[i] = 3

    # 6) 가슴 형태(breast_shape) : 정수형 변환
    # "작은 가슴' : 1, '종모양 가슴' : 2, '전체적으로 둥근형' : 3, '큰가슴' : 4, '약간 처지고 옆으로 벌어진 가슴' : 5, '많이 처진 가슴' : 6"
    print("6) 가슴 형태(breast_shape) : 정수형 변환")
    print("'작은 가슴' : 1, '종모양 가슴' : 2, '전체적으로 둥근형' : 3, '큰가슴' : 4, '약간 처지고 옆으로 벌어진 가슴' : 5, '많이 처진 가슴' : 6")
    for i in tqdm.tqdm(range(0, len(data))):
        if data['breast_shape'].iloc[i]=='작은 가슴':
            data['breast_shape'].iloc[i] = 1
        elif data['breast_shape'].iloc[i]=='종모양 가슴':
            data['breast_shape'].iloc[i] = 2
        elif data['breast_shape'].iloc[i]=='전체적으로 둥근형':
            data['breast_shape'].iloc[i] = 3
        elif data['breast_shape'].iloc[i]=='큰가슴':
            data['breast_shape'].iloc[i] = 4
        elif data['breast_shape'].iloc[i]=='약간 처지고 옆으로 벌어진 가슴':
            data['breast_shape'].iloc[i] = 5
        elif data['breast_shape'].iloc[i]=='많이 처진 가슴':
            data['breast_shape'].iloc[i] = 6

    # 7) BMI(bmi) 컬럼 신규 생성
    print("7) BMI(체질량지수)컬럼 신규 생성")
    data['BMI'] = 0
    for i in tqdm.tqdm(range(0,len(data))):
        data['BMI'].iloc[i] = data['weight'].iloc[i]/((data['height'].iloc[i]/100)**2)
    data['BMI'] = round(data['BMI'], 1) # 소수점 첫째자리 반올림 

    # 8) 허리둘레(wai_cir) : 기존 inch -> cm 변환 = 60 미만 까지
    print("8) 허리둘레 cm 단위 통일 (inch → cm, cm → cm)")
    for i in tqdm.tqdm(range(0,len(data))):
        if data['wai_cir'].iloc[i]<60:
            data['wai_cir'].iloc[i] = data['wai_cir'].iloc[i]*2.54
        else :
            data['wai_cir'].iloc[i] = data['wai_cir'].iloc[i]
    data['wai_cir'] = round(data['wai_cir'], 1) # 소수점 첫째자리 반올림
    
    # 9) 컵 사이즈 정수형 변환
    print("9) 컵 사이즈 정수형 변환")
    print("AA: 1, A: 2, B: 3, C: 4, D: 5, E: 6, F: 7")
    for i in tqdm.tqdm(range(0, len(data))):
        print('asd',data['cup_size_now'])
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
    
    # 10) 설문 결측치 삭제 후 Train/Test 데이터 분리 및 학습 진행 예정
    original_length = len(data)
    data = data.dropna()
    print(data.info())
    print("----",(original_length-len(data)),"개 결측 데이터 삭제 후 Train/Test 데이터 분리 진행----")
    
    # 11) 설문 데이터 컬럼 Type 정수형 변환
    print("10) 설문 데이터 컬럼 Type 정수형 변환 (object -> int64)")
    obj2int_columns = ['cup_size_now', 'hook_num_now', 'breast_space_fingers',
                       'bra_cup_bad_point','bra_band_bad_point','breast_shape']
    for i in tqdm.tqdm(range(0, len(obj2int_columns))):
        data[obj2int_columns[i]] = data[obj2int_columns[i]].astype('int')
    
    # 기존 컬럼 데이터 전처리 완료
    print("설문데이터 기존 컬럼 데이터 전처리 완료")
    return data

# 설문 데이터 이상치 제거 : 허리둘레, BMI 이상치 제거
# 이상치 인원 인덱스 outlier_idx 리스트에 추가
def survey_remove_outlier(data):
    print("설문 데이터 이상치 제거 : 허리둘레, BMI 이상치 제거")
    # 이상치 제거 (허리둘레 이상치 : 14명, BMI 이상치 : 12명, 키 : 453 1명 제거)
    data = data[data['wai_cir']>50.8]
    data = data[data['BMI']>16]
    data = data[data['height']<200]
    return data

# 설문 데이터 컬럼 순서 변경
def column_order_change(data):
    data = data[['height', 'weight', 'BMI', 'wai_cir', 'band_size_now',
    'cup_size_now', 'hook_num_now', 'breast_space_fingers',
    'good_bra_score_cup', 'bra_cup_bad_point', 'good_bra_score_band',
    'bra_band_bad_point', 'breast_shape', 'breast_gol_shape']]
    return data
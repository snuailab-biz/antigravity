import numpy as np
import pandas as pd
import os
import sys
import tqdm
from sklearn.model_selection import train_test_split

# 설문 데이터 불러오기 : 만족도 설문 한 사람/안한 사람 존재
# 전체 데이터 : 4,428개
# -> 만족도 O 데이터 : 3,716개 
data = pd.read_excel('data/qna_data/qna_anti_4000.xlsx')
data.info()
print(data.columns)

# 설문데이터 기존 컬럼 컬럼명 변경
data.columns = ['band_size_now', 'cup_size_now', 'hook_num_now', 'breast_space_fingers',
'good_bra_score_cup', 'bra_cup_bad_point', 'good_bra_score_band',
'bra_band_bad_point', 'breast_shape', 'breast_gol_shape', 'height', 'weight', 'wai_cir']
data.info()

# 설문데이터 기존 컬럼 데이터 전처리 : 수치형 변환 및 일부 계산
# 1) 후크 번호(hook_num_now) : 정수형 변환
#'1번(가장 안쪽)' : 1, '2번(중간)' : 2, '3번(가장 바깥쪽)' : 3 변환
for i in tqdm.tqdm(range(0, len(data))):
    if data['hook_num_now'].iloc[i]=='1번(가장 안쪽)':
        data['hook_num_now'].iloc[i] = 1
    elif data['hook_num_now'].iloc[i]=='2번(중간)':
        data['hook_num_now'].iloc[i] = 2
    elif data['hook_num_now'].iloc[i]=='3번(가장 바깥쪽)':
        data['hook_num_now'].iloc[i] = 3
        
# 2) 후크 번호 -> 젖가슴아랫둘레(band_size_now) 추가 계산
# 후크번호 1 : -1.2cm / 2 : 0 / 3 : +1.2cm 
for i in tqdm.tqdm(range(0, len(data))):
    if data['hook_num_now'].iloc[i]==1:
        data['band_size_now'].iloc[i] = data['band_size_now'].iloc[i]-1.2
    elif data['hook_num_now'].iloc[i]==2:
        data['band_size_now'].iloc[i] = data['band_size_now'].iloc[i]
    elif data['hook_num_now'].iloc[i]==3:
        data['band_size_now'].iloc[i] = data['band_size_now'].iloc[i]+1.2

# 3) 유방 간 간격(breast_space_fingers) : 정수형 변환
# '거의 붙어있음' : 0, '손가락 1개 정도' : 1, '손가락 2개 정도' : 2, '손가락 3개 정도' : 3, '손가락 4개 정도' : 4, '손가락 5개 이상' : 5
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
for i in tqdm.tqdm(range(0, len(data))):
    if data['bra_cup_bad_point'].iloc[i]=='패드가 작아요':
        data['bra_cup_bad_point'].iloc[i] = 1
    elif data['bra_cup_bad_point'].iloc[i]=='적당하며, 괜찮아요':
        data['bra_cup_bad_point'].iloc[i] = 2
    elif data['bra_cup_bad_point'].iloc[i]=='패드가 커요':
        data['bra_cup_bad_point'].iloc[i] = 3

# 5) 밴드 불편한 점(bra_band_bad_point) : 정수형 변환
# "밴드가 압박감을 주거나, 주변으로 살이 튀어나와요.' : 1, '적당하며, 괜찮아요' : 2, '밴드가 등 위로 올라가거나 헐렁해요' : 3" 변환
for i in tqdm.tqdm(range(0, len(data))):
    if data['bra_band_bad_point'].iloc[i]=='밴드가 압박감을 주거나, 주변으로 살이 튀어나와요.':
        data['bra_band_bad_point'].iloc[i] = 1
    elif data['bra_band_bad_point'].iloc[i]=='적당하며, 괜찮아요':
        data['bra_band_bad_point'].iloc[i] = 2
    elif data['bra_band_bad_point'].iloc[i]=='밴드가 등 위로 올라가거나 헐렁해요':
        data['bra_band_bad_point'].iloc[i] = 3

# 6) 가슴 형태(breast_shape) : 정수형 변환
# "작은 가슴' : 1, '종모양 가슴' : 2, '전체적으로 둥근형' : 3, '큰가슴' : 4, '약간 처지고 옆으로 벌어진 가슴' : 5, '많이 처진 가슴' : 6"
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
data['BMI'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    data['BMI'].iloc[i] = data['weight'].iloc[i]/((data['height'].iloc[i]/100)**2)
data['BMI'] = round(data['BMI'], 1) # 소수점 첫째자리 반올림 

# 8) 허리둘레(wai_cir) : 기존 inch -> cm 변환 = 60 미만 까지
for i in tqdm.tqdm(range(0,len(data))):
    if data['wai_cir'].iloc[i]<60:
        data['wai_cir'].iloc[i] = data['wai_cir'].iloc[i]*2.54
    else :
        data['wai_cir'].iloc[i] = data['wai_cir'].iloc[i]
data['wai_cir'] = round(data['wai_cir'], 1) # 소수점 첫째자리 반올림 
print(data.head())

# 컵 점수, 밴드 점수 7 미만 데이터 삭제
data = data[data['good_bra_score_cup']>=7]
data = data[data['good_bra_score_band']>=7]
# 컵/밴드 점수 7점이상 설문 데이터 셋 저장
data.to_excel('data/qna_data/qna_컬럼명변경_7점이상.xlsx',index=False)
data.info()
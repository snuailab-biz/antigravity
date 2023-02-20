import numpy as np
import pandas as pd
import os
import sys
import tqdm

# sizekorea 8차 직접측정 데이터 불러오기
data = pd.read_excel('data/size_korea_new/size_korea_02.xlsx')
data.info()

# 컬럼 순서 변경 : 기본 신체 정보 -> 목 ~ 배 ~ 다리 순
data = data[['HUMAN_ID', '성별', '조사년도', '조사일', '나이', '측정복_젖_가슴_둘레',
'측정복_상의_사이즈', '측정복_배꼽수준_허리_둘레', '키', '몸무게',
'목_둘레', '목밑_둘레', '목밑뒤_길이', '목뒤_등뼈위_겨드랑_수준_길이',
'목옆_젖꼭지_길이','목뒤_젖꼭지_길이','목뒤_젖꼭지_허리_둘레선_길이',
'목옆_젖꼭지_허리_둘레선_길이',
'어깨_길이','벽면_어깨_수평_길이','위팔_둘레_팔굽힌','위팔_사이_너비',
'손목_둘레', '팔꿈치_높이_팔굽힌','팔꿈치_둘레_팔굽힌','팔꿈치_사이_너비_팔굽힌',
'겨드랑_앞벽_사이_길이','겨드랑_앞접힘_사이_길이','겨드랑_뒤벽_사이_길이',
'겨드랑_뒤접힘_사이_길이','겨드랑_높이','겨드랑_둘레','겨드랑_두께',
'가슴_둘레','가슴_너비','가슴_두께','가슴_두께_벽면','젖_가슴_둘레',
'젖_가슴_너비','젖_가슴_두께','젖꼭지_사이_수평_길이','젖_가슴_아래_둘레',
'허리_둘레','허리_너비','허리_두께','배꼽수준_허리_둘레','배꼽수준_허리_너비',
'배꼽수준_허리_두께','허리_기준선_둘레','배꼽수준_앞중심_길이','배꼽수준_등_길이',
'배꼽수준_샅앞뒤_길이','배_둘레','앉은_배_두께','앉은_엉덩이배_두께',
'몸통_수직_길이','엉덩이_높이','엉덩이_둘레','배돌출점기준_엉덩이_둘레',
'엉덩이_너비','엉덩이_두께','엉덩이_수직_길이','엉덩이_옆_길이',
'앉은_엉덩이_너비','넙다리_직선_길이','넙다리_둘레','넙다리_중간_둘레','무릎_둘레']]

data.info()

# 컬럼 변경 파일 저장 : 최종 사이즈코리아 데이터 셋
#data.to_excel('data/size_korea_new/size_korea_01.xlsx',index=False)

# [신규 공식] 유방 볼륨 계산 : 유방_볼륨_스누
# : (4/3) X (0.5 X (젖_가슴_너비_3 - 젖꼭지_사이_수평_길이_2))² X (젖_가슴_두께_4 - 겨드랑_두께_1)
data['유방_볼륨_스누'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    data['유방_볼륨_스누'].iloc[i] = (4/3) * (0.5 * (data['젖_가슴_너비'].iloc[i]/10 - data['젖꼭지_사이_수평_길이'].iloc[i]/10)) * (0.5 * (data['젖_가슴_너비'].iloc[i]/10 - data['젖꼭지_사이_수평_길이'].iloc[i]/10)) * (data['젖_가슴_두께'].iloc[i]/10 - data['겨드랑_두께'].iloc[i]/10)

# [기존 논문] 유방 볼륨 계산 : 유방_볼륨_논문
# : (4/3) X (0.5 X (젖_가슴_너비_3 - 젖꼭지_사이_수평_길이_2))² X (젖_가슴_두께_4 - 겨드랑_두께_1)
data['유방_볼륨_논문'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    data['유방_볼륨_논문'].iloc[i] = (17.67*(data['젖_가슴_둘레'].iloc[i]/10))-(24.29*(data['젖_가슴_아래_둘레'].iloc[i]/10))+(16.31*(data['목옆_젖꼭지_허리_둘레선_길이'].iloc[i]/10))+(22.83*(data['젖_가슴_너비'].iloc[i]/10))+(12.22*(data['허리_두께'].iloc[i]/10))-(8.34*(data['겨드랑_앞벽_사이_길이'].iloc[i]/10))-611.30

# [신규 컬럼] 가슴 높이 : 젖가슴_두께 - 겨드랑_두께
data['가슴_높이_신규'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    data['가슴_높이_신규'].iloc[i] = data['젖_가슴_두께'].iloc[i] - data['겨드랑_두께'].iloc[i]

# [신규 컬럼] 컵 기준치 : 젖_가슴_둘레 - 젖_가슴_아래_둘레
data['컵_기준치_신규'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    data['컵_기준치_신규'].iloc[i] = data['젖_가슴_둘레'].iloc[i] - data['젖_가슴_아래_둘레'].iloc[i]

# [신규 컬럼] 컵 사이즈 : 컵_기준치_신규 기준
data['컵_사이즈_신규'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    if data['컵_기준치_신규'].iloc[i]<62.5 :
        data['컵_사이즈_신규'].iloc[i] = 'AAA'
    elif data['컵_기준치_신규'].iloc[i]>= 62.5 and data['컵_기준치_신규'].iloc[i]<87.5 :
        data['컵_사이즈_신규'].iloc[i] = 'AA'
    elif data['컵_기준치_신규'].iloc[i]>= 87.5 and data['컵_기준치_신규'].iloc[i]<112.5 :
        data['컵_사이즈_신규'].iloc[i] = 'A'
    elif data['컵_기준치_신규'].iloc[i]>= 112.5 and data['컵_기준치_신규'].iloc[i]<137.5 :
        data['컵_사이즈_신규'].iloc[i] = 'B'
    elif data['컵_기준치_신규'].iloc[i]>= 137.5 and data['컵_기준치_신규'].iloc[i]<162.5 :
        data['컵_사이즈_신규'].iloc[i] = 'C'
    elif data['컵_기준치_신규'].iloc[i]>= 162.5 and data['컵_기준치_신규'].iloc[i]<187.5 :
        data['컵_사이즈_신규'].iloc[i] = 'D'
    elif data['컵_기준치_신규'].iloc[i]>= 187.5 and data['컵_기준치_신규'].iloc[i]<212.5 :
        data['컵_사이즈_신규'].iloc[i] = 'E'
    elif data['컵_기준치_신규'].iloc[i]>= 212.5 and data['컵_기준치_신규'].iloc[i]<237.5 :
        data['컵_사이즈_신규'].iloc[i] = 'F'
    elif data['컵_기준치_신규'].iloc[i]>= 237.5 and data['컵_기준치_신규'].iloc[i]<262.5 :
        data['컵_사이즈_신규'].iloc[i] = 'G'
    else:
        data['컵_사이즈_신규'].iloc[i] = 'H'
        
print(data.iloc[:30])
print(data['컵_사이즈_신규'].value_counts())

# volume추가 데이터 생성
data.to_excel('data/size_korea_new/size_korea_02_volume.xlsx',index=False)

# 기존/신규 유방볼륨 상관성 파악
data_corr = pd.DataFrame(data.corr())
data_corr.to_excel('data/size_korea_new/size_korea_02_corr.xlsx',index=False)


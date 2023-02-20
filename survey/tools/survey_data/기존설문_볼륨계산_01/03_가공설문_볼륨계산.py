import numpy as np
import pandas as pd
import os
import sys
import tqdm

# 가공 설문 데이터 불러오기 : 7점 이상, 사이즈코리아 컬럼 연동
# 전체 데이터 : 1,274개 -> 1,179개
volume_column_df = pd.read_excel('data/qna_data/가공_설문_데이터.xlsx')
volume_column_df.info()

## 반지름 4개 컬럼 생성 : r1 ~ r4
# 1) r1 : 젖_가슴_너비 - 젖꼭지_사이_수평_길이
volume_column_df['r1'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['r1'].iloc[i]=0.5*(volume_column_df['middle_breast_width'].iloc[i] - volume_column_df['nipple_between_length'].iloc[i])
print(volume_column_df['r1'].unique())

# 2) r2 : 젖_가슴_높이 - 젖_가슴_아래_높이
volume_column_df['r2'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['r2'].iloc[i]=(volume_column_df['middle_breast_height'].iloc[i] - volume_column_df['bottom_breast_height'].iloc[i])
print(volume_column_df['r2'].unique())

# 3) r3 : 0.5 * 젖꼭지_사이_수평_길이
volume_column_df['r3'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['r3'].iloc[i]=0.5*(volume_column_df['nipple_between_length'].iloc[i])
print(volume_column_df['r3'].unique())
    
# 4) r4 : 겨드랑_높이 - 젖_가슴_높이
volume_column_df['r4'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['r4'].iloc[i]=(volume_column_df['armpit_height'].iloc[i] - volume_column_df['middle_breast_height'].iloc[i])
print(volume_column_df['r4'].unique())

## 높이(h) 컬럼 생성
# 1) h1 : 젖가슴두께 - 가슴두께
volume_column_df['h1'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['h1'].iloc[i]=(volume_column_df['middle_breast_thickness'].iloc[i] - volume_column_df['top_breast_thickness'].iloc[i])
    
# 2) h2 : 젖가슴두께 - 젖가슴아래두께
volume_column_df['h2'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['h2'].iloc[i]=(volume_column_df['middle_breast_thickness'].iloc[i] - volume_column_df['bottom_breast_thickness'].iloc[i])
    
# 반지름 컬럼(r1~r4), 높이컬럼 소수점 둘째자리로 반올림 (단위:cm)
r_columns =['r1','r2','r3','r4','h1','h2']
for i in tqdm.tqdm(range(0,len(r_columns))):
    volume_column_df[r_columns[i]] = round(volume_column_df[r_columns[i]],2) # 소수점 둘째자리로 반올림 (단위 : cm)

# 유방 1개 원주 계산 : 유방_원주(circumference)
volume_column_df['circum_01']=0 # 1) r1, r2
volume_column_df['circum_02']=0 # 2) r2, r3
volume_column_df['circum_03']=0 # 3) r3, r4
volume_column_df['circum_04']=0 # 4) r1, r4
volume_column_df['유방_타원_원주']=0 # 유방 1개 원주

for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['circum_01'].iloc[i]=(0.25*1.5*3.14)*((volume_column_df['r1'].iloc[i]+volume_column_df['r2'].iloc[i])-((volume_column_df['r1'].iloc[i]*volume_column_df['r2'].iloc[i])**0.5))
    volume_column_df['circum_02'].iloc[i]=(0.25*1.5*3.14)*((volume_column_df['r2'].iloc[i]+volume_column_df['r3'].iloc[i])-((volume_column_df['r2'].iloc[i]*volume_column_df['r3'].iloc[i])**0.5))
    volume_column_df['circum_03'].iloc[i]=(0.25*1.5*3.14)*((volume_column_df['r3'].iloc[i]+volume_column_df['r4'].iloc[i])-((volume_column_df['r3'].iloc[i]*volume_column_df['r4'].iloc[i])**0.5))
    volume_column_df['circum_04'].iloc[i]=(0.25*1.5*3.14)*((volume_column_df['r4'].iloc[i]+volume_column_df['r1'].iloc[i])-((volume_column_df['r4'].iloc[i]*volume_column_df['r1'].iloc[i])**0.5))
    volume_column_df['유방_타원_원주'].iloc[i]=volume_column_df['circum_01'].iloc[i]+volume_column_df['circum_02'].iloc[i]+volume_column_df['circum_03'].iloc[i]+volume_column_df['circum_04'].iloc[i]
    volume_column_df['유방_타원_원주'].iloc[i]=round(volume_column_df['유방_타원_원주'].iloc[i],2)
volume_column_df = volume_column_df.drop(['circum_01','circum_02','circum_03','circum_04'],axis=1)
    
# 유방 1개 평균 반지름 계산 : 평균r
# 반지름(r) = 원주(circumference)/2π
volume_column_df['평균r']=0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['평균r'].iloc[i]=volume_column_df['유방_타원_원주'].iloc[i]/6.284

# 유방 1개 평균 높이 계산 : 평균h
volume_column_df['평균h']=0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['평균h'].iloc[i]=(volume_column_df['h1'].iloc[i]+volume_column_df['h2'].iloc[i])/2

# 유방 평균 반지름, 높이 -> 하나의 구 형태로 볼륨계산 : 유방_볼륨_계산_구
volume_column_df['유방_볼륨_계산_구']=0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['유방_볼륨_계산_구'].iloc[i]=(4*3.142*(volume_column_df['평균r'].iloc[i]**2)*volume_column_df['평균h'].iloc[i])/3

volume_column_df['유방_볼륨_계산_구'] = volume_column_df['유방_볼륨_계산_구'].astype('int') # 소수점 첫째자리로 반올림
print(volume_column_df['유방_볼륨_계산_구'].unique())
print(volume_column_df['유방_볼륨_계산_구'].value_counts())

print(volume_column_df.info())
print(volume_column_df.head())

# 데이터 저장
volume_column_df.to_excel('data/qna_data/0201_설문_가공_볼륨계산.xlsx', index=False)

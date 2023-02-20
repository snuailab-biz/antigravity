import numpy as np
import pandas as pd
import os
import sys
import tqdm

# sizekorea 8차 3D 측정 데이터 불러오기
data = pd.read_excel('data/size_korea_new/size_korea_02_volume.xlsx')
data.info()
print(data.columns)

# 기존 데이터에 BMI 추가 -> 분리 데이터도 가져가기
data['BMI'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    data['BMI'].iloc[i] = data['몸무게'].iloc[i]/((data['키'].iloc[i]/1000)**2)
data['BMI'] = round(data['BMI'], 1) # 소수점 첫째자리 반올림 

# 데이터 분리 -> 컬럼 축소
volume_column_df = data[['키','몸무게','BMI','허리_둘레','가슴_둘레','젖_가슴_둘레','젖_가슴_아래_둘레','젖_가슴_너비', '젖꼭지_사이_수평_길이', '겨드랑_높이','젖_가슴_높이', '젖_가슴_아래_높이','가슴_두께','젖_가슴_두께', '젖_가슴_아래_두께','목옆_젖꼭지_허리_둘레선_길이','허리_두께','겨드랑_앞벽_사이_길이','겨드랑_두께']]

# 수치 컬럼 단위 cm화
cm_columns = ['키','허리_둘레','가슴_둘레','젖_가슴_둘레','젖_가슴_아래_둘레','젖_가슴_너비', '젖꼭지_사이_수평_길이', '겨드랑_높이','젖_가슴_높이', '젖_가슴_아래_높이','가슴_두께','젖_가슴_두께', '젖_가슴_아래_두께','목옆_젖꼭지_허리_둘레선_길이','허리_두께','겨드랑_앞벽_사이_길이','겨드랑_두께']
for i in tqdm.tqdm(range(0,len(cm_columns))):
    volume_column_df[cm_columns[i]] = volume_column_df[cm_columns[i]]/10
    volume_column_df[cm_columns[i]] = round(volume_column_df[cm_columns[i]], 2) # 소수점 둘째자리로 반올림 (단위 : cm)
    
# [신규 컬럼] 컵 기준치 : 젖_가슴_둘레 - 젖_가슴_아래_둘레
volume_column_df['컵_기준치_신규'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    volume_column_df['컵_기준치_신규'].iloc[i] = volume_column_df['젖_가슴_둘레'].iloc[i] - volume_column_df['젖_가슴_아래_둘레'].iloc[i]
    volume_column_df['컵_기준치_신규'].iloc[i] = round(volume_column_df['컵_기준치_신규'].iloc[i], 1) # 소수점 첫째자리로 반올림 (단위 : cm)

# [신규 컬럼] 컵 사이즈 : 컵_기준치_신규 기준
volume_column_df['컵_사이즈_신규'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    if volume_column_df['컵_기준치_신규'].iloc[i]<6.25 :
        volume_column_df['컵_사이즈_신규'].iloc[i] = 'AAA'
    elif volume_column_df['컵_기준치_신규'].iloc[i]>= 6.25 and volume_column_df['컵_기준치_신규'].iloc[i]<8.75 :
        volume_column_df['컵_사이즈_신규'].iloc[i] = 'AA'
    elif volume_column_df['컵_기준치_신규'].iloc[i]>= 8.75 and volume_column_df['컵_기준치_신규'].iloc[i]<11.25 :
        volume_column_df['컵_사이즈_신규'].iloc[i] = 'A'
    elif volume_column_df['컵_기준치_신규'].iloc[i]>= 11.25 and volume_column_df['컵_기준치_신규'].iloc[i]<13.75 :
        volume_column_df['컵_사이즈_신규'].iloc[i] = 'B'
    elif volume_column_df['컵_기준치_신규'].iloc[i]>= 13.75 and volume_column_df['컵_기준치_신규'].iloc[i]<16.25 :
        volume_column_df['컵_사이즈_신규'].iloc[i] = 'C'
    elif volume_column_df['컵_기준치_신규'].iloc[i]>= 16.25 and volume_column_df['컵_기준치_신규'].iloc[i]<18.75 :
        volume_column_df['컵_사이즈_신규'].iloc[i] = 'D'
    elif volume_column_df['컵_기준치_신규'].iloc[i]>= 18.75 and volume_column_df['컵_기준치_신규'].iloc[i]<21.25 :
        volume_column_df['컵_사이즈_신규'].iloc[i] = 'E'
    elif volume_column_df['컵_기준치_신규'].iloc[i]>= 21.25 and volume_column_df['컵_기준치_신규'].iloc[i]<23.75 :
        volume_column_df['컵_사이즈_신규'].iloc[i] = 'F'
    elif volume_column_df['컵_기준치_신규'].iloc[i]>= 23.75 and volume_column_df['컵_기준치_신규'].iloc[i]<26.25 :
        volume_column_df['컵_사이즈_신규'].iloc[i] = 'G'
    else:
        volume_column_df['컵_사이즈_신규'].iloc[i] = 'H'
        
## 반지름 4개 컬럼 생성 : r1 ~ r4
# 1) r1 : 젖_가슴_너비 - 젖꼭지_사이_수평_길이
volume_column_df['r1'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['r1'].iloc[i]=0.5*(volume_column_df['젖_가슴_너비'].iloc[i] - volume_column_df['젖꼭지_사이_수평_길이'].iloc[i])
print(volume_column_df['r1'].unique())

# 2) r2 : 젖_가슴_높이 - 젖_가슴_아래_높이
volume_column_df['r2'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['r2'].iloc[i]=(volume_column_df['젖_가슴_높이'].iloc[i] - volume_column_df['젖_가슴_아래_높이'].iloc[i])
print(volume_column_df['r2'].unique())
    
# 3) r3 : 0.5 * 젖꼭지_사이_수평_길이
volume_column_df['r3'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['r3'].iloc[i]=0.5*(volume_column_df['젖꼭지_사이_수평_길이'].iloc[i])
print(volume_column_df['r3'].unique())
    
# 4) r4 : 겨드랑_높이 - 젖_가슴_높이
volume_column_df['r4'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['r4'].iloc[i]=(volume_column_df['겨드랑_높이'].iloc[i] - volume_column_df['젖_가슴_높이'].iloc[i])
print(volume_column_df['r4'].unique())

## 높이(h) 컬럼 생성
# 1) h1 : 젖가슴두께 - 가슴두께
volume_column_df['h1'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['h1'].iloc[i]=(volume_column_df['젖_가슴_두께'].iloc[i] - volume_column_df['가슴_두께'].iloc[i])
    
# 2) h2 : 젖_가슴_두께 - 젖_가슴_아래_두께
volume_column_df['h2'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['h2'].iloc[i]=(volume_column_df['젖_가슴_두께'].iloc[i] - volume_column_df['젖_가슴_아래_두께'].iloc[i])
    
# 반지름 컬럼(r1~r4), 높이컬럼 소수점 둘째자리로 반올림 (단위:cm)
r_columns =['r1','r2','r3','r4','h1','h2']
for i in tqdm.tqdm(range(0,len(r_columns))):
    volume_column_df[r_columns[i]] = round(volume_column_df[r_columns[i]],2) # 소수점 둘째자리로 반올림 (단위 : cm)

# 이상치 존재 -> 젖가슴높이, 가슴높이 > 겨드랑 높이, 젖가슴두께 < 젖가슴아래두께 -> 15명 제거

# [신규 공식] 유방 볼륨 계산 : 유방_볼륨_스누
# : (4/3) X (0.5 X (젖_가슴_너비_3 - 젖꼭지_사이_수평_길이_2))² X (젖_가슴_두께_4 - 겨드랑_두께_1)
volume_column_df['유방_볼륨_스누'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['유방_볼륨_스누'].iloc[i] = (4/3) * (0.5 * (volume_column_df['젖_가슴_너비'].iloc[i] - volume_column_df['젖꼭지_사이_수평_길이'].iloc[i])) * (0.5 * (volume_column_df['젖_가슴_너비'].iloc[i] - volume_column_df['젖꼭지_사이_수평_길이'].iloc[i])) * (volume_column_df['젖_가슴_두께'].iloc[i] - volume_column_df['겨드랑_두께'].iloc[i])
    volume_column_df['유방_볼륨_스누'].iloc[i] = round(volume_column_df['유방_볼륨_스누'].iloc[i],2)
# [기존 논문] 유방 볼륨 계산 : 유방_볼륨_논문
# : (4/3) X (0.5 X (젖_가슴_너비_3 - 젖꼭지_사이_수평_길이_2))² X (젖_가슴_두께_4 - 겨드랑_두께_1)
volume_column_df['유방_볼륨_논문'] = 0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['유방_볼륨_논문'].iloc[i] = (17.67*(volume_column_df['젖_가슴_둘레'].iloc[i]))-(24.29*(volume_column_df['젖_가슴_아래_둘레'].iloc[i]))+(16.31*(volume_column_df['목옆_젖꼭지_허리_둘레선_길이'].iloc[i]))+(22.83*(volume_column_df['젖_가슴_너비'].iloc[i]))+(12.22*(volume_column_df['허리_두께'].iloc[i]))-(8.34*(volume_column_df['겨드랑_앞벽_사이_길이'].iloc[i]))-611.30
    volume_column_df['유방_볼륨_논문'].iloc[i] = round(volume_column_df['유방_볼륨_논문'].iloc[i],2)

# 유방_볼륨_타원구 : 1/3 * π * h * ((r1*r2) + (r2*r3) + (r1*r4) + (r3*r4))
# 1/3 * π = 1.0473
# 1/6 * π = 0.786
volume_column_df['유방_볼륨_타원구']=0
for i in tqdm.tqdm(range(0,len(volume_column_df))):
    volume_column_df['유방_볼륨_타원구'].iloc[i]=0.786 * volume_column_df['h1'].iloc[i] * ((volume_column_df['r1'].iloc[i]*volume_column_df['r2'].iloc[i]) + (volume_column_df['r2'].iloc[i]*volume_column_df['r3'].iloc[i]) + (volume_column_df['r1'].iloc[i]*volume_column_df['r4'].iloc[i]) + (volume_column_df['r3'].iloc[i]*volume_column_df['r4'].iloc[i]))

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

print(volume_column_df.info())
print(volume_column_df.head())
# 데이터 저장
volume_column_df.to_excel('data/size_korea_new/최종_선정컬럼_볼륨_계산_3가지_1222.xlsx', index=False)
# 상관계수 데이터 저장
corr = volume_column_df.corr()
#corr.to_excel('data/size_korea_new/최종_선정컬럼_볼륨_계산_3가지_상관계수_1222.xlsx', index=False)
#volume_column_df.to_excel('data/size_korea_new/반지름_볼륨계산.xlsx', index=False)
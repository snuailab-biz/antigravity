import numpy as np
import pandas as pd
import os
import sys
import tqdm

# main.py -> volume 계산 사이즈 표 확인 목적 데이터 가공
data = pd.read_excel('data/volume_사이즈 비교_0202.xlsx')
data.info()

print(data['band_size_now'].unique())

# 밴드 사이즈 정규화
for i in tqdm.tqdm(range(0, len(data))):
    if data['band_size_now'].iloc[i]<=66.2:
        data['band_size_now'].iloc[i]=65
    elif data['band_size_now'].iloc[i]>66.2 and data['band_size_now'].iloc[i]<=72:
        data['band_size_now'].iloc[i]=70
    elif data['band_size_now'].iloc[i]>72 and data['band_size_now'].iloc[i]<=76.2:
        data['band_size_now'].iloc[i]=75
    elif data['band_size_now'].iloc[i]>76.2 and data['band_size_now'].iloc[i]<=81.2:
        data['band_size_now'].iloc[i]=80
    elif data['band_size_now'].iloc[i]>81.2 and data['band_size_now'].iloc[i]<=86.2:
        data['band_size_now'].iloc[i]=85
    elif data['band_size_now'].iloc[i]>86.2 :
        data['band_size_now'].iloc[i]=90
print(data['band_size_now'].value_counts())

# 밴드 사이즈 컬럼 정수화
data['band_size_now'] = data['band_size_now'].astype('int')

# custom_size(맞춤 사이즈) 컬럼 생성
data['custom_size'] = 0
for i in tqdm.tqdm(range(0, len(data))):
    data['custom_size'].iloc[i] = str(data['band_size_now'].iloc[i])+data['cup_size_now'].iloc[i]
#print(data['custom_size'].value_counts())

data.to_excel('data/volume_사이즈 비교_0202.xlsx', index=False)

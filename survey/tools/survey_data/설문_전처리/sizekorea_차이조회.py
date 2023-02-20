import numpy as np
import pandas as pd
import os
import sys
import tqdm

# 신규 사이즈코리아 데이터 불러오기
# 전체 데이터 : 2,525개
data = pd.read_excel('data/size_korea_new/최종_선정컬럼_볼륨_계산_3가지_1222.xlsx')
data.info()

# 가슴둘레/젖가슴둘레 크기 차이 비교
problem_person = 0
for i in tqdm.tqdm(range(0, len(data))):
    if data['가슴_둘레'].iloc[i]>data['젖_가슴_둘레'].iloc[i]:
        problem_person+=1
print(problem_person)

# 젖가슴둘레/젖가슴아래둘레 크기 차이 비교
problem_person = 0
for i in tqdm.tqdm(range(0, len(data))):
    if data['젖_가슴_아래_둘레'].iloc[i]>data['젖_가슴_둘레'].iloc[i]:
        problem_person+=1
print(problem_person)
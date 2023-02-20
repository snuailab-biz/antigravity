import numpy as np
import pandas as pd
import os
import sys
import tqdm
from sklearn.model_selection import train_test_split

# sizekorea 안티그래비티 데이터 불러오기
data = pd.read_excel('data/size_korea_anti/sizekorea_major_columns_re.xlsx')
data.info()

# [신규 공식] 유방 볼륨 계산 : 유방_볼륨_스누
# : (4/3) X (0.5 X (젖_가슴_너비_3 - 젖꼭지_사이_수평_길이_2))² X (젖_가슴_두께_4 - 겨드랑_두께_1)
data['유방_볼륨_스누'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    data['유방_볼륨_스누'].iloc[i] = (4/3) * (0.5 * (data['젖_가슴_너비_3'].iloc[i]/10 - data['젖꼭지_사이_수평_길이_2'].iloc[i]/10)) * (0.5 * (data['젖_가슴_너비_3'].iloc[i]/10 - data['젖꼭지_사이_수평_길이_2'].iloc[i]/10)) * (data['젖_가슴_두께_4'].iloc[i]/10 - data['겨드랑_두께_1'].iloc[i]/10)

# [기존 논문] 유방 볼륨 계산 : 유방_볼륨_논문
# : (4/3) X (0.5 X (젖_가슴_너비_3 - 젖꼭지_사이_수평_길이_2))² X (젖_가슴_두께_4 - 겨드랑_두께_1)
data['유방_볼륨_논문'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    data['유방_볼륨_논문'].iloc[i] = (17.67*(data['젖가슴_둘레'].iloc[i]/10))-(24.29*(data['젖가슴_아래_둘레'].iloc[i]/10))+(16.31*(data['목옆_허리_둘레선_길이'].iloc[i]/10))+(22.83*(data['젖_가슴_너비_3'].iloc[i]/10))+(12.22*(data['허리_두께'].iloc[i]/10))-(8.34*(data['겨드랑_앞벽_사이_길이'].iloc[i]/10))-611.30

# [신규 컬럼] 가슴 높이 : 젖가슴_두께 - 겨드랑_두께
data['가슴_높이_신규'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    data['가슴_높이_신규'].iloc[i] = data['젖_가슴_두께_4'].iloc[i] - data['겨드랑_두께_1'].iloc[i]

# [신규 컬럼] 컵 기준치 : 젖_가슴_둘레 - 젖_가슴_아래_둘레
data['컵_기준치_신규'] = 0
for i in tqdm.tqdm(range(0,len(data))):
    data['컵_기준치_신규'].iloc[i] = data['젖가슴_둘레'].iloc[i] - data['젖가슴_아래_둘레'].iloc[i]

## 1) clo 결측 행 삭제 : 1074명 존재
data = data.dropna(subset=['유방_볼륨_CLO'], how='any', axis=0)

## 2) 몸무게 결측 1명 삭제
data = data.dropna(subset=['몸무게'], how='any', axis=0)
print(data.info())

# 컬럼 소수점 첫째자리 반올림
data['BMI']= round(data['BMI'], 1)
data['유방_볼륨_CLO']= round(data['유방_볼륨_CLO'], 1)
data['유방_볼륨_스누']= round(data['유방_볼륨_스누'], 1)
data['유방_볼륨_논문']= round(data['유방_볼륨_논문'], 1)
print(data.head())
print(data.columns)

# 일부컬럼 정수형 변환
int_columns = ['목옆_젖꼭지_길이', '목뒤_젖꼭지_길이',
       '목옆_허리_둘레선_길이', '위팔_둘레', '위팔_사이_너비', '팔꿈치_사이_너비', '겨드랑_둘레', '겨드랑_두께_1',
       '겨드랑_앞벽_사이_길이', '가슴_둘레', '가슴_너비', '가슴_두께', '젖가슴_둘레', '젖가슴_아래_둘레',
       '젖꼭지_사이_수평_길이_2', '젖_가슴_너비_3', '젖_가슴_두께_4', '허리_둘레', '허리_너비', '허리_두께',
       '배꼽수준_허리_둘레', '배꼽수준_허리_너비', '배꼽수준_허리_두께', '배_둘레', '앉은_배_두께',
       '앉은_엉덩이배_두께', '엉덩이_둘레', '엉덩이_두께', '배돌출점기준_엉덩이_둘레', '엉덩이_돌출점_배_돌출점_두께',
       '벽면_몸통_두께', '넙다리_둘레']
for i in tqdm.tqdm(range(0, len(int_columns))):
    data[int_columns[i]] = data[int_columns[i]].astype('int64')
    
## 신규 컬럼 5개 추가
# 1) 가슴_젖가슴_둘레 비율 = 클 수록 작은 컵
data['가슴_젖가슴_둘레'] = 0
for i in tqdm.tqdm(range(0, len(data))):
    data['가슴_젖가슴_둘레'].iloc[i] = data['가슴_둘레'].iloc[i] / data['젖가슴_둘레'].iloc[i]
data['가슴_젖가슴_둘레']= round(data['가슴_젖가슴_둘레'], 2) # 소수점 둘째자리로
# 2) 가슴_밑가슴_둘레 비율
data['가슴_밑가슴_둘레'] = 0
for i in tqdm.tqdm(range(0, len(data))):
    data['가슴_밑가슴_둘레'].iloc[i] = data['가슴_둘레'].iloc[i] / data['젖가슴_아래_둘레'].iloc[i]
data['가슴_밑가슴_둘레'] = round(data['가슴_밑가슴_둘레'], 2) # 소수점 둘째자리로
# 3) 젖가슴_밑가슴_둘레 비율
data['젖가슴_밑가슴_둘레'] = 0
for i in tqdm.tqdm(range(0, len(data))):
    data['젖가슴_밑가슴_둘레'].iloc[i] = data['젖가슴_둘레'].iloc[i] / data['젖가슴_아래_둘레'].iloc[i]
data['젖가슴_밑가슴_둘레'] = round(data['젖가슴_밑가슴_둘레'], 2) # 소수점 둘째자리로
# 4) 수평길이_너비 비율
data['젖_수평길이_너비'] = 0
for i in tqdm.tqdm(range(0, len(data))):
    data['젖_수평길이_너비'].iloc[i] = data['젖꼭지_사이_수평_길이_2'].iloc[i] / data['젖_가슴_너비_3'].iloc[i]
data['젖_수평길이_너비'] = round(data['젖_수평길이_너비'], 2) # 소수점 둘째자리로
# 5) 가슴 두께 비율
data['가슴_젖가슴_두께'] = 0
for i in tqdm.tqdm(range(0, len(data))):
    data['가슴_젖가슴_두께'].iloc[i] = data['젖_가슴_두께_4'].iloc[i] / data['가슴_두께'].iloc[i]
data['가슴_젖가슴_두께'] = round(data['가슴_젖가슴_두께'], 2) # 소수점 둘째자리로
print(data.info())

# 훈련/테스트 데이터 7:3 분리, 정답 정수형 변환
data['유방_볼륨_CLO'] = data['유방_볼륨_CLO'].astype('int64')
X = data[['키', '몸무게', 'BMI', '체지방량', '체지방율', '지방조절', '목옆_젖꼭지_길이', '목뒤_젖꼭지_길이',
       '목옆_허리_둘레선_길이', '위팔_둘레', '위팔_사이_너비', '팔꿈치_사이_너비', '겨드랑_둘레', '겨드랑_두께_1',
       '겨드랑_앞벽_사이_길이', '가슴_둘레', '가슴_너비', '가슴_두께', '젖가슴_둘레', '젖가슴_아래_둘레',
       '젖꼭지_사이_수평_길이_2', '젖_가슴_너비_3', '젖_가슴_두께_4', '허리_둘레', '허리_너비', '허리_두께',
       '배꼽수준_허리_둘레', '배꼽수준_허리_너비', '배꼽수준_허리_두께', '배_둘레', '앉은_배_두께',
       '앉은_엉덩이배_두께', '엉덩이_둘레', '엉덩이_두께', '배돌출점기준_엉덩이_둘레', '엉덩이_돌출점_배_돌출점_두께',
       '벽면_몸통_두께', '넙다리_둘레', '유방_볼륨_스누', '유방_볼륨_논문', '가슴_높이_신규', '컵_기준치_신규',
       '가슴_젖가슴_둘레', '가슴_밑가슴_둘레','젖가슴_밑가슴_둘레', '젖_수평길이_너비', '가슴_젖가슴_두께']]
y = data[['유방_볼륨_CLO']]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=550)

# 랜덤포레스트 회귀모델 학습
from sklearn.ensemble import RandomForestRegressor
rf_run = RandomForestRegressor(n_estimators=500, random_state=1234)
rf_run.fit(X_train, y_train)

# 랜덤포레스트 회귀모델 정확도
print('일반 모델 훈련 세트 정확도 : {:.3f}'.format(rf_run.score(X_train, y_train)))
print('일반 모델 테스트 세트 정확도 : {:.3f}'.format(rf_run.score(X_test, y_test)))
# 특성 중요도
feature_importance = pd.DataFrame(rf_run.feature_importances_.reshape((1, -1)), columns=X_train.columns, index=['feature_importance'])
print(feature_importance)

## CLO 미존재 데이터 로드, 전처리 후 예측
predict = pd.read_excel('data/size_korea_anti/size_korea_volume_fill_clo_5column.xlsx')
# 볼륨 제외 컬럼 미존재 데이터 삭제
dropna_columns = ['몸무게', 'BMI', '체지방량', '체지방율', '지방조절', '목옆_젖꼭지_길이', '목뒤_젖꼭지_길이',
                  '목옆_허리_둘레선_길이', '겨드랑_둘레', '젖가슴_둘레', '젖가슴_아래_둘레','젖꼭지_사이_수평_길이_2',
                  '배꼽수준_허리_둘레', '배_둘레']
for i in tqdm.tqdm(range(0, len(dropna_columns))):
    predict = predict.dropna(subset=[dropna_columns[i]], how='any', axis=0)
# 일부컬럼 정수형 변환
int_columns = ['목옆_젖꼭지_길이', '목뒤_젖꼭지_길이',
       '목옆_허리_둘레선_길이', '위팔_둘레', '위팔_사이_너비', '팔꿈치_사이_너비', '겨드랑_둘레', '겨드랑_두께_1',
       '겨드랑_앞벽_사이_길이', '가슴_둘레', '가슴_너비', '가슴_두께', '젖가슴_둘레', '젖가슴_아래_둘레',
       '젖꼭지_사이_수평_길이_2', '젖_가슴_너비_3', '젖_가슴_두께_4', '허리_둘레', '허리_너비', '허리_두께',
       '배꼽수준_허리_둘레', '배꼽수준_허리_너비', '배꼽수준_허리_두께', '배_둘레', '앉은_배_두께',
       '앉은_엉덩이배_두께', '엉덩이_둘레', '엉덩이_두께', '배돌출점기준_엉덩이_둘레', '엉덩이_돌출점_배_돌출점_두께',
       '벽면_몸통_두께', '넙다리_둘레']
for i in tqdm.tqdm(range(0, len(int_columns))):
    predict[int_columns[i]] = predict[int_columns[i]].astype('int64')
# CLO 예측 진행 : 'predict' 컬럼 생성
X = predict[['키', '몸무게', 'BMI', '체지방량', '체지방율', '지방조절', '목옆_젖꼭지_길이', '목뒤_젖꼭지_길이',
       '목옆_허리_둘레선_길이', '위팔_둘레', '위팔_사이_너비', '팔꿈치_사이_너비', '겨드랑_둘레', '겨드랑_두께_1',
       '겨드랑_앞벽_사이_길이', '가슴_둘레', '가슴_너비', '가슴_두께', '젖가슴_둘레', '젖가슴_아래_둘레',
       '젖꼭지_사이_수평_길이_2', '젖_가슴_너비_3', '젖_가슴_두께_4', '허리_둘레', '허리_너비', '허리_두께',
       '배꼽수준_허리_둘레', '배꼽수준_허리_너비', '배꼽수준_허리_두께', '배_둘레', '앉은_배_두께',
       '앉은_엉덩이배_두께', '엉덩이_둘레', '엉덩이_두께', '배돌출점기준_엉덩이_둘레', '엉덩이_돌출점_배_돌출점_두께',
       '벽면_몸통_두께', '넙다리_둘레', '유방_볼륨_스누', '유방_볼륨_논문', '가슴_높이_신규','컵_기준치_신규',
       '가슴_젖가슴_둘레', '가슴_밑가슴_둘레','젖가슴_밑가슴_둘레', '젖_수평길이_너비', '가슴_젖가슴_두께']]
y = predict[['유방_볼륨_CLO']]
for col in predict.columns.tolist():
    predict[col] = pd.DataFrame(predict[col])
predict["CLO_predict"] = pd.DataFrame(data=rf_run.predict(X), index=predict.index)
predict.to_excel('data/size_korea_anti/clo_predict_rf_reg_5column.xlsx',index=False)

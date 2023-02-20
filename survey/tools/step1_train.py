from sklearn.ensemble import RandomForestRegressor
import joblib
# from 파일명 import 클래스/함수명
import json
import pandas as pd
from survey_preprocess import kor2eng, text2int, survey_remove_outlier, column_order_change
from sk_sv_merge_volume import sz_kr_remove_outlier, survey_sizekorea, volume_calculation, train_test_data_split
from sklearn.ensemble import RandomForestRegressor


def train(X_train, X_test, y_train, y_test):
    print("\n---------------------------------------------")
    print(f"\n유방 볼륨 계산에 대해 학습을 시작합니다.")
    rf_run = RandomForestRegressor(n_estimators=500, random_state=1234)
    rf_run.fit(X_train, y_train)
    # 모델 정확도
    print('일반 모델 훈련 세트 정확도 : {:.3f}'.format(rf_run.score(X_train, y_train)))
    print('일반 모델 테스트 세트 정확도 : {:.3f}'.format(rf_run.score(X_test, y_test)))

    # 모델 저장
    joblib.dump(rf_run, 'tools/ML_model/고객_설문_유방볼륨_수치_예측모델.pkl')
    
    result = '고객 설문 데이터 유방 볼륨 계산에 대한 학습이 완료되었습니다.'
    return result

    
if __name__=="__main__":
    # get config setting
    config_dir = "tools/config_js.json"
    #config_dir = r"F:\안티그래비티\python_code\tools\test\config.json"
    #config_dir = '/media/hwi/One Touch/안티그래비티/python_code/tools/test/config linux.json'
    
    with open(config_dir, 'r', encoding='utf-8') as json_file:
        config_data = json.load(json_file)

    # 1) 설문 데이터 로드 및 전처리
    survey_data_dir = config_data['survey_data_dir']
    data = pd.read_excel(survey_data_dir)
    data = kor2eng(data)
    data = text2int(data)
    data - survey_remove_outlier(data)
    survey_preprocess_data = column_order_change(data)
    
    # 2) 사이즈코리아 데이터 로드 및 전처리
    sizekorea_data_dir = config_data['sizekorea_data_dir']
    sz_kr = pd.read_excel(sizekorea_data_dir)
    sz_kr = sz_kr_remove_outlier(sz_kr)
    sz_kr = survey_sizekorea(survey_preprocess_data, sz_kr)
    sk_sv_merge_volume_data = volume_calculation(sz_kr)
    # sk_sv_merge_volume_data = volume_calculation(survey_sizekorea(survey_preprocess_data, remove_outlier(sz_kr)))
    X_train, X_test, y_train, y_test = train_test_data_split(sk_sv_merge_volume_data)
    
    print("Train / Test 데이터 분리 완료")
    result = train(X_train, X_test, y_train, y_test)
    print(result)
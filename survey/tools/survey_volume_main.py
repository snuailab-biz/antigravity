# from 파일명 import 클래스/함수명
import json
import pandas as pd
from tools.step2_inference import inference
from tools.survey_preprocess import kor2eng, text2int, survey_remove_outlier, column_order_change
from tools.sk_sv_merge_volume import sz_kr_remove_outlier, survey_sizekorea, volume_calculation, train_test_data_split
from sklearn.ensemble import RandomForestRegressor

class AntiSurvey(object):
    def __init__(self):
        # SizeKorea Data Merge 용
        config_dir = "tools/config_js.json"
        with open(config_dir, 'r', encoding='utf-8') as json_file:
            self.config_data = json.load(json_file)

    def predict(self, data):
        # 1) 설문 데이터 로드 및 전처리
        data = pd.DataFrame([data['data']])
        data = kor2eng(data)
        data = text2int(data)
        data - survey_remove_outlier(data)
        survey_preprocess_data=column_order_change(data)
        
        # 2) 사이즈코리아 데이터 로드 및 전처리
        sizekorea_data_dir = self.config_data['sizekorea_data_dir']
        sz_kr = pd.read_excel(sizekorea_data_dir)
        sz_kr = sz_kr_remove_outlier(sz_kr)
        sz_kr = survey_sizekorea(survey_preprocess_data, sz_kr)
        sk_sv_merge_volume_data = volume_calculation(sz_kr)
        
        isInference = self.config_data['isInference']
        inference_data = sk_sv_merge_volume_data
        result = inference(inference_data)
        return result
    
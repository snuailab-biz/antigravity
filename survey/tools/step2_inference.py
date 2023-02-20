import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import pandas as pd
from tools.survey_preprocess import kor2eng, text2int, survey_remove_outlier, column_order_change
from tools.sk_sv_merge_volume import sz_kr_remove_outlier, survey_sizekorea, volume_calculation, train_test_data_split
from sklearn.ensemble import RandomForestRegressor


def inference(inference_data):
    print("inference")
    
    #inference_data["Volume_predict"] = pd.DataFrame(data=rf_run.predict(X), index=inference_data.index)
    # for col in inference_data.columns.tolist():
    #     inference_data[col] = pd.DataFrame(inference_data[col])
    #     inference_data["volume"] = pd.DataFrame(data=rf_run.predict(X), index=predict.index)

    model_name = "tools/ML_model/고객_설문_유방볼륨_수치_예측모델.pkl"
    rf_run = joblib.load(model_name)
    inference_data["volume"] = pd.DataFrame(data=rf_run.predict(inference_data), index=inference_data.index)
    
    # dataframe 행, 열 변경해주셔야함
    inference_data = inference_data.T
    
    # dataframe -> string json -> json
    string_json = inference_data.to_json()
    real_json = json.loads(string_json)
    return real_json

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
    inference_data = sk_sv_merge_volume_data    
    inference(inference_data)
from tools.survey_volume_main import AntiSurvey
import json
    
if __name__=="__main__":
    with open(r"./data/sample/people1_json.json", 'rb') as jsondata:
        data = json.load(jsondata)
    # print(data) # sample Data
    
    survey = AntiSurvey()
    result = survey.predict(data)
    print("Volume: ",result['0']['volume'])
    
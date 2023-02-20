
  <h1 align="center">Diagram : Antigravity Volume Estimation</h1>

#### [Antigravity Bitbucket : FIT-AI-VOLUME](https://bitbucket.org/antigravity_dev/fit-ai-volume/src/master/)
---

---
main_GT.py
---
전체 프로세스 흐름도
main_GT.py
```mermaid
flowchart LR
PreProcessing-->Train
PreProcessing-->Inference
```
---
Image, Depth(G.T.) 로드 순서  
main.py 
```mermaid
flowchart LR
PreProcessing-->Train
PreProcessing-->Inference
subgraph PreProcessing["Survey Preprocessing"]
    subgraph inp [input]
        style inp stroke-dasharray: 5 5
        survey_data(바디미러 고객 설문 데이터)
        size_korea(사이즈코리아 데이터)

    end
    subgraph pre [설문 전처리]
        survey_data(바디미러 고객 설문 데이터)-->
        korea2eng-->text2int-->survey_remove_outlier-->column_order_change-->res(설문 전처리 결과)
    end
    subgraph sizekorea [사이즈 코리아 전처리]
        size_korea(사이즈코리아 데이터)-->sz_kr_remove_outlier-->survey_sizekorea
        res-->survey_sizekorea--> volume_cal(부피 계산)
    end
    subgraph out [output]
        style out stroke-dasharray: 5 5
        volume_cal(부피 계산)-->df(dataframe)
    end
end
```

---

- Survey Train
```mermaid
flowchart TB
PreProcessing-->Train
PreProcessing-->Inference
subgraph Train["Survey Train"]
    subgraph inp [input]
        style inp stroke-dasharray: 5 5
        df(Dataframe)
    end
    df-->process-->model
    subgraph out [output]
        style out stroke-dasharray: 5 5
        model
    end
end
```

---
Survey Inference

```mermaid
flowchart TB
subgraph Infer["Survey Inference"]
    subgraph inp [input]
        style inp stroke-dasharray: 5 5
        new(New Data)
    end
    new-->PreProcessing-->model-->Volume
    PreProcessing-->variable_data(sizekorea데이터에 수치)
    subgraph out [output]
        style out stroke-dasharray: 5 5
        Volume
    end
end
```



# [FIT-AI-SNUAILAB](https://bitbucket.org/antigravity_dev/fit-ai-snuailab/src/master/)

Antigravity, SNUAILAB 브라 사이즈 추천 시스템 프로젝트!!

```bash
conda create -n FIT-AI python=3.7
conda activate FIT-AI

pip install -r requirements.txt
```

## Projects Structure

```bash
FIT-AI-SNUAILAB
├── .git
├── .gitignore
├── README.md
├── requirements.txt
├── reference.md
│
├── conmmon
│   ├── __init__ 
│   └── common.py  : 프로젝트 전반에 사용되는 공용 함수
│
├── dataset ( 원본 파일 ) 
│   ├── azure_kinect_pad 
│   │   ├── calibration     : .pickle 파일 형식
│   │   ├── depths 
│   │   ├── images 
│   │   ├── npy    
│   │   └── segmentations
│   └── test_samples: 외주 파일 (test set)
│
├── detector : segmentation & landmark  ------ @leejj
│   
├── estimator: depth estimator 
│   ├── dataset : train/test dataset
│   ├── models
│   ├── __init__.py
│   ├── bts.py              : bts models
│   ├── anti_args.py         : train/test config file
│   ├── estimator_main.py   : estimator main 
│   ├── step0_bts_augumentation.py
│   ├── step1_bts_trian.py
│   ├── step2_bts_predict.py
│   └── step3_bts_visualize.py
│
├── imgporc : 전처리/후처리 
│   ├── __init__.py
│   ├── postprocessing.py    : voloume rendering 후처리 알고리즘
│   └── preprocessing.py     : image 전처리 알고리즘
│ 
├── reconsturctor : tsdf/mesh 
│   ├── __init__.py
│   └── reconstructor_main.py : reconstructor main
│   
└── tool : 각종 utils 
│   ├── __init__.py
│   ├── utils.py             : depth to pointcloud visualize 관련 함수 (현재)
│   └── vtk_utils.py         : vtk관련 utils 구현 함수  
```
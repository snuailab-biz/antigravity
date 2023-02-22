from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, qRgb
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel, QTableWidget,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QInputDialog)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
import glob
import os
import cv2
import numpy as np
import math

from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QLineEdit, QTableView, QPushButton, QLabel, \
							QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant, QModelIndex

import pandas as pd
pd.set_option('mode.chained_assignment',  None) # 경고 off

from PyQt5 import QtCore, QtGui, QtWidgets


type_enum = {
    '1': 
    "끈 가림\n패드 부분을 찾는 과정에서 끈이 가리고 있어 안쪽의 패드 영역을 못찾음. \n \
        - 학습데이터로부터 학습된 모델은 Large feature로 보면 둥그런 모양을 먼저 찾게되는데 그 부분에서 둥글지 못하게 나누어지고 medium, small 단계로 가면서 나머지 패드에 대해 못찾게 됨. \n \
        - 반대로 가려진 부분까지 패드로 인식하게 되는 경우도 존재함. \n \
        - 끈이 가리게 되어 Landmark부분을 찾기 어려워짐. \n\n \
    ### 개선 방향 \n \
        : 데이터 수집 환경을 제한(제약을 걸어)하여 수집한 데이터에 대해서 진행.  \n \
        또는 이렇게 말린 ‘학습’ 데이터를 많이 늘려 학습 진행. " ,
    '2': 
    "찌그러짐\n이 경우에는 패드를 충분히 잘 찾게되어 큰 문제가 되지 않지만 심하게 찌그러진 경우 다음과 같은 문제가 발생할 수 있음.\n \
          - 모양이 너무 심하게 찌그러진 경우 학습데이터에서 항상 보던 동그란 형태의 패드가 아닌 급격하게 꺾이는 모양을 갖게 되어 바깥 면이 보이는 경우 패드로 인식.(Large Feature에서 동그랗게 분리하기 때문)\n \
      (학습데이터에서는 전혀 볼 수 없는 부분 )\n \
          - 패드 면적이 좁아지며 찌그러진 부분의 depth 정보가 손실됨.\n\n \
    ### 개선 방향\n \
          : 데이터 수집 환경을 제한(제약을 걸어)하여 수집한 데이터에 대해서 진행. \n \
            이 부분은 학습데이터 annotation에서 패드만 잘 따서 학습을 하게 된다면 안쪽 패드만 잘 찾게 될 수 있지만, depth 정보가 손실되는 문제가 발생할 수 있으므로 <수집 조건 제약> 하는 쪽으로 가야함.",
    '3': 
    '끈 연결부 포함\n이 경우에는 Volume Estimation을 진행하는데 있어 depth값 중 높이 솟아 있는 것이 있고 type8 고스트 노이즈가 발생할수 있음.\n \
        크리티컬한 부분은 아니지만, 학습데이터에서도 끈까지 포함되어 있다면 더 나아가 모델이 끈 위쪽까지 찾게 될 수도 있음\n\n \
    ### 개선 방향\n \
        Ground Truth 수집 시 끈을 제외하고 annotation을 진행. \n \
        annotation을 디테일하게 한다면 일반화(generalization)이 좋아짐.',
    '4': 
    '면 정의 불가\n이 형태는 면정의 불가 판정 뿐만 아니라 모델의 결과도 잘못 나온 경우가 많습니다. \n \
        왼쪽 패드, 오른쪽 패드가 겹쳐 있어 구분짓기 어려우며 가운데 landmark에 대해서도 구분이 어려운 데이터. \n\n \
    ### 성능 개선 방향 \n \
    ### DeepLearning Model \n \
        : 충분히 성능을 올리기 위한 방법은 1. 모델의 클래스를 구분지어 학습, 2. 데이터로 일반화시키기. \n\n \
    ### Volume Estimate \n \
        : 데이터를 굴곡지게 만들어 재정의가 필요함.  \n \
        얼마나 굴곡지게 할지에 따라 부피가 달라지겠지만 그에 따라 패드 면적이 좁아짐에 따라 부피는 유사할 것으로 보임.', 
    '5': 
    'Over Segment & No Detect Landmark\n이 경우는 학습되지 않은 데이터 형태에서 발생하며 유사한 데이터라고 할지여도 학습데이터 양이 현저히 부족하여 일반화가 전혀 이루어지지 않음. - overfitting 발생 \n \
    ### Overfitting \n \
        - 데이터 양 : 적은 데이터의 양을 학습시키면 **절대적**으로 overffiting이 발생함 (deep learning 관점) \n \
        - Domain : 현재 학습용 데이터셋 domain 자체가 overfitting이 발생하기 쉬움. \n \
    (동일한 각도, 동일한 조도, 동일한 거리, 동일한 object 개수, 동일한 annotation) \n \
    다양한 색상, 다양한 모양이 있지만 그 모양, 색상에 대한 데이터가 매우 부족함. \n\n \
    ### 성능 개선 방향 \n \
        - 많은 데이터 수집 및 데이터 품질 증가. \n \
        - 딥러닝 모델은 엄청나게 많은 파라미터를 가지고 있어 데이터 개수가 적을 경우에는 모델 파라미터 한 개 한 개가 fitting이 될 수 밖에 없다. \n \
        - 데이터 양은 모델의 크기, 데이터 형태, 데이터 분포에 따라 다르지만 클래스(모양이나 색상), 도메인(배경이나, 각도, 거리) 당 기본 1,000개 이상은 되어야하며, 그 이하의 데이터에서는 **어떠한 결과 분석**을 할 수가 없다. \n \
        - 다른 조건(사진 하나에 object가 여러 개 있는 경우 등),',
    '6': 
    'Dragged effect\n물체 경계 부분 depth 정보가 불확실하게 나오며, 경계부에 대해 sharp하게 나누어도 depth가 아래로 꺼지는 형상이 발생함. (Anti depth camera GT에 대해서도 이 현상이 발생함.) \n\n \
    ### 개선 방향 \n \
        : 환경 통제 및 카메라 파라미터 튜닝.',
    '7': 
    '조도 문제\n',
    '8': 
    'Diff Domain\n각도, 거리, 보이는 정도 차이 : 현재 학습에 사용되는 데이터의 object가 일정 거리에 고정되어 있어 패드의 크기가 전체 이미지에서 차지하는 비율이 어느정도 고정되어 있습니다. \n\n \
    ### 개선 방향 \n \
    : 다양한 도메인의 학습 데이터 \n \
        수집 데이터를 학습 도메인에 맞추기. ( 수집 환경 제한 ) \n \
        수집 환경을 제한하는 것도 필요하겠지만, 현재 학습 데이터양이 매우 적어 정확하게 동일하지 않으면 틀릴 수 있으므로, 학습 데이터 양을 늘리면서 다양한 데이터를 수집하는 것이 현실적', 
    '9': 'Good',
    '10': "Best"
}



# root = '/home/ljj/data/anti/valid_2'
class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()
        self.ok_data = False
        self.ok_img = False
        self.ok_csv = False

        self.setObjectName("MainWii")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)
        # df = pd.read_csv(os.path.join(root, 'valid.csv'))
        # ddf = df.drop(columns='ID')
        # img_lst = []
        # image_lst = df['Image'].values.tolist()
        # for img in image_lst:
        #     img_lst.append(img[4:])

        # img_lst.sort()
        # self.img_list = []
        # self.mask_list = []
        # for img_path in img_lst:
        #     row = df[df['Image'].str.contains(img_path)]
        #     self.img_list.append(os.path.join(root,row.Landmark.values[0]))
        #     self.mask_list.append(os.path.join(root,row.Mask.values[0]))


        self.printer = QPrinter()
        self.width = 2520
        self.height = 1420

        self.setWindowTitle("Image Viewer")
        self.resize(self.width, self.height)

        self.pushButton_save = QtWidgets.QPushButton(self.centralwidget, clicked=self.save_csv)

        self.pushButton_save.setGeometry(QtCore.QRect(1700, 1400, 650, 100))
        self.pushButton_save.setObjectName("save_button")

        # self.pushButton_apply = QtWidgets.QPushButton(self.centralwidget, clicked=self.apply_valid)
        # self.pushButton_apply.setGeometry(QtCore.QRect(1500, 880, 981, 71))
        # self.table.setGeometry(QtCore.QRect(1500, 90, 981, 761))


        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        QtCore.QMetaObject.connectSlotsByName(self)

        self.gray_color_table = [qRgb(i, i, i) for i in range(256)]

        self.pos = 0

        self.mask_name = QtWidgets.QLabel(self.centralwidget)
        self.mask_name.setGeometry(QtCore.QRect(640, 12, 67, 17))
        self.mask_name.setObjectName("label")
        self.mask_name.setFont(QtGui.QFont("",15))
        # self.mask_name.setStyleSheet("border: 15px;")
        self.maskLabel = QLabel(self.centralwidget)
        self.maskLabel.setBackgroundRole(QPalette.Base)
        self.maskLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.maskLabel.setGeometry(QtCore.QRect(30, 50, 0, 0))
        self.maskLabel.setScaledContents(False)
        # self.mask_chbox = QtWidgets.QCheckBox(self.centralwidget)
        # self.mask_chbox.setGeometry(QtCore.QRect(1360, 320, 131, 41))
        # self.mask_chbox.setObjectName("checkBox")
        # self.mask_chbox.setStyleSheet("QCheckBox::indicator" "{" "width :40px;" "height : 40px;" "}")

        self.mask_group = QtWidgets.QGroupBox(self.centralwidget)
        self.mask_group.setGeometry(QtCore.QRect(1320, 470, 161, 151))
        self.mask_radio_t = QtWidgets.QRadioButton(self.mask_group)
        self.mask_radio_t.setGeometry(QtCore.QRect(10, 40, 112, 23))
        self.mask_radio_f = QtWidgets.QRadioButton(self.mask_group)
        self.mask_radio_f.setGeometry(QtCore.QRect(10, 100, 121, 41))

        self.point_name = QtWidgets.QLabel(self.centralwidget)
        self.point_name.setGeometry(QtCore.QRect(640, 792, 67, 17))
        self.point_name.setObjectName("label")
        self.point_name.setFont(QtGui.QFont("",15))
        self.pointLabel = QLabel(self.centralwidget)
        self.pointLabel.setBackgroundRole(QPalette.Base)
        self.pointLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.pointLabel.setGeometry(QtCore.QRect(30, 820, 0, 0))
        self.pointLabel.setObjectName("label_2")
        self.pointLabel.setScaledContents(True)

        self.point_group = QtWidgets.QGroupBox(self.centralwidget)
        self.point_group.setGeometry(QtCore.QRect(1320, 1000, 161, 151))
        self.point_radio_t = QtWidgets.QRadioButton(self.point_group)
        self.point_radio_t.setGeometry(QtCore.QRect(10, 40, 112, 23))
        self.point_radio_f = QtWidgets.QRadioButton(self.point_group)
        self.point_radio_f.setGeometry(QtCore.QRect(10, 100, 121, 41))

        # Label
        self.label_image_path = QtWidgets.QLabel(self.centralwidget)
        self.label_image_path.setGeometry(QtCore.QRect(1520, 575, 87, 17))
        self.label_imagename = QtWidgets.QLabel(self.centralwidget)
        self.label_imagename.setGeometry(QtCore.QRect(1620, 575, 367, 17))

        self.label_type = QtWidgets.QLabel(self.centralwidget)
        self.label_type.setGeometry(QtCore.QRect(1520, 650, 67, 17))
        self.label_type_value = QtWidgets.QLabel(self.centralwidget)
        self.label_type_value.setGeometry(QtCore.QRect(1640, 650, 67, 17))

        self.label_volume = QtWidgets.QLabel(self.centralwidget)
        self.label_volume.setGeometry(QtCore.QRect(1520, 700, 111, 17))
        self.label_volume_value = QtWidgets.QLabel(self.centralwidget)
        self.label_volume_value.setGeometry(QtCore.QRect(1640, 700, 67, 17))
        self.label_volume_left = QtWidgets.QLabel(self.centralwidget)
        self.label_volume_left.setGeometry(QtCore.QRect(1520, 750, 111, 17))
        self.label_volume_left_value = QtWidgets.QLabel(self.centralwidget)
        self.label_volume_left_value.setGeometry(QtCore.QRect(1640, 750, 67, 17))
        self.label_volume_right = QtWidgets.QLabel(self.centralwidget)
        self.label_volume_right.setGeometry(QtCore.QRect(1520, 800, 111, 17))
        self.label_volume_right_value = QtWidgets.QLabel(self.centralwidget)
        self.label_volume_right_value.setGeometry(QtCore.QRect(1640, 800, 67, 17))

        self.label_pad = QtWidgets.QLabel(self.centralwidget)
        self.label_pad.setGeometry(QtCore.QRect(1520, 850, 111, 17))
        self.label_pad_value = QtWidgets.QLabel(self.centralwidget)
        self.label_pad_value.setGeometry(QtCore.QRect(1640, 850, 67, 17))
        self.label_pad_left = QtWidgets.QLabel(self.centralwidget)
        self.label_pad_left.setGeometry(QtCore.QRect(1520, 900, 111, 17))
        self.label_pad_left_value = QtWidgets.QLabel(self.centralwidget)
        self.label_pad_left_value.setGeometry(QtCore.QRect(1640, 900, 67, 17))
        self.label_pad_right = QtWidgets.QLabel(self.centralwidget)
        self.label_pad_right.setGeometry(QtCore.QRect(1520, 950, 111, 17))
        self.label_pad_right_value = QtWidgets.QLabel(self.centralwidget)
        self.label_pad_right_value.setGeometry(QtCore.QRect(1640, 950, 67, 17))


        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(1720, 630, 731, 500))


        self.type_group = QtWidgets.QGroupBox(self.centralwidget)
        self.type_group.setGeometry(QtCore.QRect(1520, 1150, 910, 101))
        self.type_radio_1 = QtWidgets.QRadioButton(self.type_group)
        a = 90
        b = 30
        self.type_radio_1.setGeometry(QtCore.QRect(50, 40, 112, 23))
        self.type_radio_2 = QtWidgets.QRadioButton(self.type_group)
        self.type_radio_2.setGeometry(QtCore.QRect(b+a*1, 40, 112, 23))
        self.type_radio_3 = QtWidgets.QRadioButton(self.type_group)
        self.type_radio_3.setGeometry(QtCore.QRect(b+a*2, 40, 112, 23))
        self.type_radio_4 = QtWidgets.QRadioButton(self.type_group)
        self.type_radio_4.setGeometry(QtCore.QRect(b+a*3, 40, 112, 23))
        self.type_radio_5 = QtWidgets.QRadioButton(self.type_group)
        self.type_radio_5.setGeometry(QtCore.QRect(b+a*4, 40, 112, 23))
        self.type_radio_6 = QtWidgets.QRadioButton(self.type_group)
        self.type_radio_6.setGeometry(QtCore.QRect(b+a*5, 40, 112, 23))
        self.type_radio_7 = QtWidgets.QRadioButton(self.type_group)
        self.type_radio_7.setGeometry(QtCore.QRect(b+a*6, 40, 112, 23))
        self.type_radio_8 = QtWidgets.QRadioButton(self.type_group)
        self.type_radio_8.setGeometry(QtCore.QRect(b+a*7, 40, 112, 23))
        self.type_radio_9 = QtWidgets.QRadioButton(self.type_group)
        self.type_radio_9.setGeometry(QtCore.QRect(b+a*8, 40, 112, 23))
        self.type_radio_10 = QtWidgets.QRadioButton(self.type_group)
        self.type_radio_10.setGeometry(QtCore.QRect(b+a*9, 40, 112, 23))

        self.createActions()
        self.initTable()
        self.retranslateUi()
    
    def LoadCsv(self):
        
        self.urlSource = self.dataSourceField.text() +'/valid.csv'
        self.root = self.dataSourceField.text()
        # self.urlSource = '/home/ljj/data/anti/valid_2/valid.csv'
        df = pd.read_csv(self.urlSource)
        df.fillna('')
        img_lst = []
        rows=[]
        image_lst = df['Image'].values.tolist()
        for img in image_lst:
            img_lst.append(img[4:])

        img_lst.sort()
        for img_path in img_lst:
            row = df[df['Image'].str.contains(img_path)]
            rows.append(row.values.tolist()[0])
        
        df = pd.DataFrame(rows, columns=['ID', 'Image', 'Mask', 'Valid Mask', 'Landmark', 'Valid Landmark', 'Rendering', 'Volume', "Left Volume", "Right Volume", 'Valid Volume', 'Pad Volume', 'Pad Left Volume', 'Pad Right Volume', 'Valid Pad Volume', "Type"])
        # df = df.drop('ID', axis=1)
        # df = df.drop('Image', axis=1)
        # df = df.drop('Volume', axis=1)
        # df = df.drop('Left Volume', axis=1)
        # df = df.drop('Right Volume', axis=1)
        # df = df.drop('Valid Volume', axis=1)
        # df = df.drop('Rendering', axis=1)
        self.df = df.copy()
        self.model=PandasModel(self.df)
        self.table.setModel(self.model)

        self.comboColumns.clear()
        self.comboColumns.addItems(self.df.columns)


        self.table = QTableView(self.centralwidget)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionsMovable(True)

        self.load_images() 

        mask = cv2.imread(self.mask_list[self.pos])
        point = cv2.imread(self.img_list[self.pos])
        self.openImage(image=self.toQImage(point))
        self.openImage(image=self.toQImage(mask), type='mask')          
        self.ok_data = True  
    
    def load_images(self):
        self.img_list=[]
        self.mask_list=[]
        for i in range(len(self.df.Mask.values.tolist())):
            self.mask_list.append(os.path.join(self.root, self.df.Mask.values.tolist()[i]))
            self.img_list.append(os.path.join(self.root, self.df.Landmark.values.tolist()[i]))
        
        self.total = len(self.img_list)

        self.ok_img = True

    
    def initTable(self):
        self.csv_path = QtWidgets.QLabel(self.centralwidget)
        self.csv_path.setGeometry(QtCore.QRect(1520, 60, 67, 17))
        self.dataSourceField = QtWidgets.QLineEdit(self.centralwidget)
        self.dataSourceField.setGeometry(QtCore.QRect(1600, 40, 760, 40))
        self.pushButton_load = QtWidgets.QPushButton(self.centralwidget, clicked=self.LoadCsv)
        self.pushButton_load.setGeometry(QtCore.QRect(2400, 40, 80, 40))
        self.table = QtWidgets.QTableView(self.centralwidget)
        self.table.setGeometry(QtCore.QRect(1500, 90, 981, 461))
        self.comboColumns = QtWidgets.QComboBox(self.centralwidget)

    def apply_valid(self):
        if self.ok_data:
            if self.mask_radio_t.isChecked():
                self.df['Valid Mask'][self.df[self.df.Mask.str.contains(self.mask_list[self.pos].split('/')[-1])].index] = 'O'
            elif self.mask_radio_f.isChecked():
                self.df['Valid Mask'][self.df[self.df.Mask.str.contains(self.mask_list[self.pos].split('/')[-1])].index] = 'X'
            if self.point_radio_t.isChecked():
                self.df['Valid Landmark'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] = 'O'
            elif self.point_radio_f.isChecked():
                self.df['Valid Landmark'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] = 'X'
            if self.type_radio_1.isChecked():
                self.df['Type'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] ='1'
            elif self.type_radio_2.isChecked():
                self.df['Type'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] ='2'
            elif self.type_radio_3.isChecked():
                self.df['Type'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] ='3'
            elif self.type_radio_4.isChecked():
                self.df['Type'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] ='4'
            elif self.type_radio_5.isChecked():
                self.df['Type'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] ='5'
            elif self.type_radio_6.isChecked():
                self.df['Type'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] ='6'
            elif self.type_radio_7.isChecked():
                self.df['Type'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] ='7'
            elif self.type_radio_8.isChecked():
                self.df['Type'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] ='8'
            elif self.type_radio_9.isChecked():
                self.df['Type'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] ='9'
            elif self.type_radio_10.isChecked():
                self.df['Type'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] ='10'

            self.model=PandasModel(self.df)
            self.table.setModel(self.model)

            self.comboColumns.clear()
            self.comboColumns.addItems(self.df.columns)


            self.table = QTableView(self.centralwidget)
            self.table.setSortingEnabled(True)
            self.table.horizontalHeader().setSectionsMovable(True)
            self.ok_csv = True
    
    def save_csv(self):
        if self.ok_csv:
            self.df.to_csv(self.urlSource, index=False)

    
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self._translate = QtCore.QCoreApplication.translate
        self.pushButton_save.setText(_translate("MainWindow", "Save"))
        # self.pushButton_apply.setText(_translate("MainWindow", "Apply"))
        self.pushButton_load.setText(_translate("MainWindow", "Load"))
        self.mask_name.setText(_translate("MainWindow", "Mask"))
        self.point_name.setText(_translate("MainWindow", "Point"))
        self.csv_path.setText(_translate("MainWindow", "Load Path : "))

        self.label_type.setText(_translate("MainWindow", "Type :"))
        self.label_volume.setText(_translate("MainWindow", "Volume : "))
        self.label_volume_left.setText(_translate("MainWindow", "Left: "))
        self.label_volume_right.setText(_translate("MainWindow", "right : "))
        self.label_pad.setText(_translate("MainWindow", "Pad Volume : "))
        self.label_pad_left.setText(_translate("MainWindow", "Pad Left : "))
        self.label_pad_right.setText(_translate("MainWindow", "Pad Right :"))

        self.label_image_path.setText(_translate("MainWindow", "Image Path :"))

        self.mask_group.setTitle(_translate("MainWindow", "Mask Valid"))
        self.mask_radio_f.setText(_translate("MainWindow", "False"))
        self.mask_radio_t.setText(_translate("MainWindow", "True"))
        self.point_group.setTitle(_translate("MainWindow", "Point Valid"))
        self.point_radio_f.setText(_translate("MainWindow", "False"))
        self.point_radio_t.setText(_translate("MainWindow", "True"))
        
        self.type_group.setTitle(_translate("MainWindow", "Type Select"))
        self.type_radio_1.setText(_translate("MainWindow", "1"))
        self.type_radio_2.setText(_translate("MainWindow", "2"))
        self.type_radio_3.setText(_translate("MainWindow", "3"))
        self.type_radio_4.setText(_translate("MainWindow", "4"))
        self.type_radio_5.setText(_translate("MainWindow", "5"))
        self.type_radio_6.setText(_translate("MainWindow", "6"))
        self.type_radio_7.setText(_translate("MainWindow", "7"))
        self.type_radio_8.setText(_translate("MainWindow", "8"))
        self.type_radio_9.setText(_translate("MainWindow", "9"))
        self.type_radio_10.setText(_translate("MainWindow", "10"))
    
    def normalSize(self):
        self.maskLabel.adjustSize()
        self.pointLabel.adjustSize()

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()
        self.updateActions()
        
    def createActions(self):
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S",
                enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False,
                checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)
        
    def updateActions(self):
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())
       
    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))
    def keyPressEvent(self, e):
        if self.ok_img:
            if e.key() == 65:
                self.apply_valid()
                if not self.pos == 0:
                    self.pos -= 1
                    point = cv2.imread(self.img_list[self.pos])
                    mask = cv2.imread(self.mask_list[self.pos])
                    """
                    이미지 처리
                    """

                    self.openImage(image=self.toQImage(point))
                    self.openImage(image=self.toQImage(mask), type='mask')            
                    landmark_tf = self.df['Valid Landmark'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index]
                    mask_tf = self.df['Valid Mask'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index]
                    if type(landmark_tf) != float:
                        if landmark_tf.values[0]=='O':
                            self.point_radio_t.setChecked(True)
                        elif landmark_tf.values[0]=='X':
                            self.point_radio_f.setChecked(True)
                        if mask_tf.values[0]=='O':
                            self.mask_radio_t.setChecked(True)
                        elif mask_tf.values[0]=='X':
                            self.mask_radio_f.setChecked(True)
                    
                    self.label_type_value.setText(self._translate("MainWindow", str(self.df['Type'][self.pos])))
                    self.label_volume_value.setText(self._translate("MainWindow", str(self.df['Volume'][self.pos])))
                    self.label_volume_left_value.setText(self._translate("MainWindow", str(self.df['Left Volume'][self.pos])))
                    self.label_volume_right_value.setText(self._translate("MainWindow", str(self.df['Right Volume'][self.pos])))
                    self.label_pad_value.setText(self._translate("MainWindow", str(self.df['Pad Volume'][self.pos])))
                    self.label_pad_left_value.setText(self._translate("MainWindow", str(self.df['Pad Left Volume'][self.pos])))
                    self.label_pad_right_value.setText(self._translate("MainWindow", str(self.df['Pad Right Volume'][self.pos])))
                    self.label_imagename.setText(self._translate("MainWindow", str(self.df['Image'][self.pos])))
                    type_num = str(self.df['Type'][self.pos])[0]
                    if type_num in type_enum.keys():
                        self.label_type_value.setText(self._translate("MainWindow", type_num))
                        self.textBrowser.setText(type_enum[type_num])
                    else:
                        self.textBrowser.clear()
                        self.label_type_value.setText(self._translate("MainWindow", 'Non Type'))

                                                    
            elif e.key() == 68:
                self.apply_valid()
                self.pos += 1
                if self.total == self.pos:
                    self.pos -= 1
                point = cv2.imread(self.img_list[self.pos])
                mask = cv2.imread(self.mask_list[self.pos])
                """
                이미지 처리
                """
                self.openImage(image=self.toQImage(point))            
                self.openImage(image=self.toQImage(mask), type='mask')            
                landmark_tf = self.df['Valid Landmark'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index]
                mask_tf = self.df['Valid Mask'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index]
                if type(landmark_tf) != float:
                    if landmark_tf.values[0]=='O':
                        self.point_radio_t.setChecked(True)
                    elif landmark_tf.values[0]=='X':
                        self.point_radio_f.setChecked(True)

                    if mask_tf.values[0]=='O':
                        self.mask_radio_t.setChecked(True)
                    elif mask_tf.values[0]=='X':
                        self.mask_radio_f.setChecked(True)
                
                self.label_type_value.setText(self._translate("MainWindow", str(self.df['Type'][self.pos])))
                self.label_volume_value.setText(self._translate("MainWindow", str(self.df['Volume'][self.pos])))
                self.label_volume_left_value.setText(self._translate("MainWindow", str(self.df['Left Volume'][self.pos])))
                self.label_volume_right_value.setText(self._translate("MainWindow", str(self.df['Right Volume'][self.pos])))
                self.label_pad_value.setText(self._translate("MainWindow", str(self.df['Pad Volume'][self.pos])))
                self.label_pad_left_value.setText(self._translate("MainWindow", str(self.df['Pad Left Volume'][self.pos])))
                self.label_pad_right_value.setText(self._translate("MainWindow", str(self.df['Pad Right Volume'][self.pos])))
                self.label_imagename.setText(self._translate("MainWindow", str(self.df['Image'][self.pos])))
                type_num = str(self.df['Type'][self.pos])[0]
                if type_num in type_enum.keys():
                    self.label_type_value.setText(self._translate("MainWindow", type_num))
                    self.textBrowser.setText(type_enum[type_num])
                else:
                    self.textBrowser.clear()
                    self.label_type_value.setText(self._translate("MainWindow", 'Non Type'))

            elif e.key() == 49:
                self.mask_radio_t.setChecked(True)
            elif e.key() == 50:
                self.mask_radio_f.setChecked(True)
            elif e.key() == 51:
                self.point_radio_t.setChecked(True)
            elif e.key() == 52:
                self.point_radio_f.setChecked(True)




    def openImage(self, image=None, fileName=None, type='img'):
            if image == None:
                image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return
            if type=='img':
                self.pointLabel.setPixmap(QPixmap.fromImage(image))
            elif type=='mask':
                self.maskLabel.setPixmap(QPixmap.fromImage(image))
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()
            if not self.fitToWindowAct.isChecked():
                self.maskLabel.adjustSize()
                self.pointLabel.adjustSize()

    def toQImage(self, im, copy=False):
        if im is None:
            return QImage()
        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(self.gray_color_table)
                return qim.copy() if copy else qim
            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
                    return qim.copy() if copy else qim

class PandasModel(QAbstractTableModel):
	def __init__(self, df=pd.DataFrame(), parent=None):
		super().__init__(parent)
		self._df = df

	def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
		if role != Qt.ItemDataRole.DisplayRole:
			return QVariant()
		if orientation == Qt.Orientation.Horizontal:
			try:
				return self._df.columns.tolist()[section]
			except IndexError:
				return QVariant()
		elif orientation == Qt.Orientation.Vertical:
			try:
				return self._df.index.tolist()[section]
			except IndexError:
				return QVariant()

	def rowCount(self, parent=QModelIndex()):
		return self._df.shape[0]

	def columnCount(self, parent=QModelIndex()):
		return self._df.shape[1]

	def data(self, index, role=Qt.ItemDataRole.DisplayRole):
		if role != Qt.ItemDataRole.DisplayRole:
			return QVariant()
		if not index.isValid():
			return QVariant()
		return QVariant(str(self._df.iloc[index.row(), index.column()]))



if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
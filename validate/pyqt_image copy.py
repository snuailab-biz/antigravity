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

        self.pushButton_save.setGeometry(QtCore.QRect(2320, 1290, 141, 141))
        self.pushButton_save.setObjectName("save_button")

        self.pushButton_apply = QtWidgets.QPushButton(self.centralwidget, clicked=self.apply_valid)
        self.pushButton_apply.setGeometry(QtCore.QRect(1500, 880, 981, 71))
        # self.table.setGeometry(QtCore.QRect(1500, 90, 981, 761))
        self.pushButton_apply.setObjectName("apply_button")


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


        # self.point_chbox = QtWidgets.QCheckBox(self.centralwidget)
        # self.point_chbox.setGeometry(QtCore.QRect(1340, 1320, 131, 41))
        # self.point_chbox.setObjectName("checkBox_2")
        # self.point_chbox.setStyleSheet("QCheckBox::indicator" "{" "width :40px;" "height : 40px;" "}")

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
        
        df = pd.DataFrame(rows, columns=['ID', 'Image', 'Mask', 'Valid Mask', 'Landmark', 'Valid Landmark', 'Rendering', 'Volume', "Left Volume", "Right Volume", 'Valid Volume', 'Pad Volume', 'Pad Left Volume', 'Pad Right Volume', 'Valid Pad Volume'])
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
        self.table.setGeometry(QtCore.QRect(1500, 90, 981, 761))
        self.comboColumns = QtWidgets.QComboBox(self.centralwidget)

    def apply_valid(self):
        if self.ok_data:
            if self.mask_radio_t.isChecked():
                self.df['Valid Mask'][self.df[self.df.Mask.str.contains(self.mask_list[self.pos].split('/')[-1])].index] = 'O'
            elif self.mask_radio_f.isChecked():
                self.df['Valid Mask'][self.df[self.df.Mask.str.contains(self.mask_list[self.pos].split('/')[-1])].index] = 'X'
            # if self.mask_chbox.isChecked():
            #     self.df['Valid Mask'][self.df[self.df.Mask.str.contains(self.mask_list[self.pos].split('/')[-1])].index] = 'O'
            if self.point_radio_t.isChecked():
                self.df['Valid Landmark'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] = 'O'
            elif self.point_radio_f.isChecked():
                self.df['Valid Landmark'][self.df[self.df.Landmark.str.contains(self.img_list[self.pos].split('/')[-1])].index] = 'X'

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
        self.pushButton_save.setText(_translate("MainWindow", "Save"))
        self.pushButton_apply.setText(_translate("MainWindow", "Apply"))
        self.pushButton_load.setText(_translate("MainWindow", "Load"))
        self.mask_name.setText(_translate("MainWindow", "Mask"))
        self.point_name.setText(_translate("MainWindow", "Point"))
        self.csv_path.setText(_translate("MainWindow", "Load Path : "))
        # self.mask_chbox.setText(_translate("MainWindow", "Mask Valid"))
        # self.point_chbox.setText(_translate("MainWindow", "Point Valid"))

        self.mask_group.setTitle(_translate("MainWindow", "Mask Valid"))
        self.mask_radio_f.setText(_translate("MainWindow", "False"))
        self.mask_radio_t.setText(_translate("MainWindow", "True"))
        self.point_group.setTitle(_translate("MainWindow", "Point Valid"))
        self.point_radio_f.setText(_translate("MainWindow", "False"))
        self.point_radio_t.setText(_translate("MainWindow", "True"))
    
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
            if e.key() == 72:
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



                    # if self.mask_group.isChecked():
                    # self.mask_radio_t.setChecked(False)
                    # self.point_radio_f.setChecked(False)
                    # self.point_radio_t.setChecked(False)
                    # if self.point_self.radio_f.isChecked():
                    #     self.point_radio_f.toggle()
                    # elif self.point_radio_t.isChecked():
                    #     self.point_radio_t.toggle()
                    # print('\r' + self.img_list[self.pos], end="")
                                                    
            elif e.key() == 76:
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
            elif e.key() == 49:
                self.mask_radio_t.setChecked(True)
            elif e.key() == 50:
                self.mask_radio_f.setChecked(True)
            elif e.key() == 51:
                self.point_radio_t.setChecked(True)
            elif e.key() == 52:
                self.point_radio_f.setChecked(True)




                # if self.point_chbox.isChecked():
                #     self.point_chbox.toggle()
                # if self.mask_chbox.isChecked():
                #     self.mask_chbox.toggle()
                # print('\r' + self.img_list[self.pos], end="")

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
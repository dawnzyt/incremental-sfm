# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sfm_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1285, 699)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(190, 110, 651, 371))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.match_layout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.match_layout.setContentsMargins(0, 0, 0, 0)
        self.match_layout.setObjectName("match_layout")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(860, 110, 401, 371))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.points_layout = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.points_layout.setContentsMargins(0, 0, 0, 0)
        self.points_layout.setObjectName("points_layout")
        self.match_title_btn = QtWidgets.QPushButton(self.centralwidget)
        self.match_title_btn.setGeometry(QtCore.QRect(432, 80, 121, 28))
        self.match_title_btn.setObjectName("match_title_btn")
        self.points_title_btn = QtWidgets.QPushButton(self.centralwidget)
        self.points_title_btn.setGeometry(QtCore.QRect(972, 80, 141, 28))
        self.points_title_btn.setObjectName("points_title_btn")
        self.open_btn = QtWidgets.QPushButton(self.centralwidget)
        self.open_btn.setGeometry(QtCore.QRect(50, 20, 93, 28))
        self.open_btn.setObjectName("open_btn")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 80, 171, 401))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.localBA_checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.localBA_checkBox.setGeometry(QtCore.QRect(10, 10, 91, 19))
        self.localBA_checkBox.setObjectName("localBA_checkBox")
        self.globalBA_checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.globalBA_checkBox.setGeometry(QtCore.QRect(10, 40, 101, 21))
        self.globalBA_checkBox.setObjectName("globalBA_checkBox")
        self.match_factor_text = QtWidgets.QLineEdit(self.groupBox)
        self.match_factor_text.setGeometry(QtCore.QRect(10, 130, 113, 21))
        self.match_factor_text.setObjectName("match_factor_text")
        self.match_factor_hint_btn = QtWidgets.QPushButton(self.groupBox)
        self.match_factor_hint_btn.setGeometry(QtCore.QRect(10, 100, 111, 28))
        self.match_factor_hint_btn.setObjectName("match_factor_hint_btn")
        self.ransac_t_hint_btn = QtWidgets.QPushButton(self.groupBox)
        self.ransac_t_hint_btn.setGeometry(QtCore.QRect(10, 160, 141, 28))
        self.ransac_t_hint_btn.setObjectName("ransac_t_hint_btn")
        self.ransac_t_text = QtWidgets.QLineEdit(self.groupBox)
        self.ransac_t_text.setGeometry(QtCore.QRect(10, 190, 113, 21))
        self.ransac_t_text.setObjectName("ransac_t_text")
        self.downscale_factor_hint_btn = QtWidgets.QPushButton(self.groupBox)
        self.downscale_factor_hint_btn.setGeometry(QtCore.QRect(10, 220, 141, 28))
        self.downscale_factor_hint_btn.setObjectName("downscale_factor_hint_btn")
        self.downscale_factor_text = QtWidgets.QLineEdit(self.groupBox)
        self.downscale_factor_text.setGeometry(QtCore.QRect(10, 250, 113, 21))
        self.downscale_factor_text.setObjectName("downscale_factor_text")
        self.set_default_para_btn = QtWidgets.QPushButton(self.groupBox)
        self.set_default_para_btn.setGeometry(QtCore.QRect(10, 370, 71, 28))
        self.set_default_para_btn.setObjectName("set_default_para_btn")
        self.save_parm_btn = QtWidgets.QPushButton(self.groupBox)
        self.save_parm_btn.setGeometry(QtCore.QRect(90, 370, 71, 28))
        self.save_parm_btn.setObjectName("save_parm_btn")
        self.select_checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.select_checkBox.setGeometry(QtCore.QRect(10, 70, 101, 21))
        self.select_checkBox.setObjectName("select_checkBox")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(190, 490, 1071, 151))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.log_layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.log_layout.setContentsMargins(0, 0, 0, 0)
        self.log_layout.setObjectName("log_layout")
        self.step_btn = QtWidgets.QPushButton(self.centralwidget)
        self.step_btn.setGeometry(QtCore.QRect(20, 490, 71, 28))
        self.step_btn.setObjectName("step_btn")
        self.run_btn = QtWidgets.QPushButton(self.centralwidget)
        self.run_btn.setGeometry(QtCore.QRect(100, 490, 71, 28))
        self.run_btn.setObjectName("run_btn")
        self.save_ply_btn = QtWidgets.QPushButton(self.centralwidget)
        self.save_ply_btn.setGeometry(QtCore.QRect(40, 570, 111, 28))
        self.save_ply_btn.setObjectName("save_ply_btn")
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(560, 10, 341, 91))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.logo_layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.logo_layout.setContentsMargins(0, 0, 0, 0)
        self.logo_layout.setObjectName("logo_layout")
        self.para_hint_btn = QtWidgets.QPushButton(self.centralwidget)
        self.para_hint_btn.setGeometry(QtCore.QRect(50, 60, 93, 28))
        self.para_hint_btn.setObjectName("para_hint_btn")
        self.stop_btn = QtWidgets.QPushButton(self.centralwidget)
        self.stop_btn.setGeometry(QtCore.QRect(40, 530, 111, 28))
        self.stop_btn.setObjectName("stop_btn")
        self.watermark_btn = QtWidgets.QPushButton(self.centralwidget)
        self.watermark_btn.setGeometry(QtCore.QRect(190, 37, 161, 51))
        self.watermark_btn.setStyleSheet("border:none;\n"
"font-size:40px;\n"
"font-family: STXingkai;\n"
"\n"
"\n"
"color:qradialgradient(spread:repeat, cx:0.5, cy:0.5, radius:0.077, fx:0.5, fy:0.5, stop:0 rgba(0, 169, 255, 147), stop:0.497326 rgba(0, 0, 0, 147), stop:1 rgba(0, 169, 255, 147))")
        self.watermark_btn.setObjectName("watermark_btn")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1285, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.match_title_btn.setText(_translate("MainWindow", "match view"))
        self.points_title_btn.setText(_translate("MainWindow", "point cloud"))
        self.open_btn.setText(_translate("MainWindow", "open"))
        self.localBA_checkBox.setText(_translate("MainWindow", "local BA"))
        self.globalBA_checkBox.setText(_translate("MainWindow", "global BA"))
        self.match_factor_hint_btn.setText(_translate("MainWindow", "match factor"))
        self.ransac_t_hint_btn.setText(_translate("MainWindow", "ransac threshold"))
        self.downscale_factor_hint_btn.setText(_translate("MainWindow", "downscale factor"))
        self.set_default_para_btn.setText(_translate("MainWindow", "default"))
        self.save_parm_btn.setText(_translate("MainWindow", "save"))
        self.select_checkBox.setText(_translate("MainWindow", "select"))
        self.step_btn.setText(_translate("MainWindow", "step"))
        self.run_btn.setText(_translate("MainWindow", "run"))
        self.save_ply_btn.setText(_translate("MainWindow", "save ply"))
        self.para_hint_btn.setText(_translate("MainWindow", "para set"))
        self.stop_btn.setText(_translate("MainWindow", "stop"))
        self.watermark_btn.setText(_translate("MainWindow", "dawnzyt"))

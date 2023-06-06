import sys

import cv2
import numpy as np
import qtawesome
from PyQt5.QtCore import pyqtSlot, QThread, QObject, pyqtSignal, Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap, QDoubleValidator, QFont, QIcon, QTextCursor
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QGridLayout, QLabel, QScrollArea, \
    QPlainTextEdit, QFileDialog, QMessageBox
from pyqtgraph.opengl import GLViewWidget
import pyqtgraph.opengl as gl
from sfm_ui import Ui_MainWindow
import logging
import sfm


# 单步sfm线程
class StepWorker(QThread, QObject):
    sinEnd = pyqtSignal()  # step线程结束信号

    def __init__(self, obj, model):
        super().__init__()
        self.obj = obj  # 主窗口对象
        self.model = model  # sfm模型对象, 均是引用传递

    def run(self) -> None:
        self.obj.is_running = True
        self.model.step(if_local_BA=self.obj.if_local_BA, if_global_BA=self.obj.if_global_BA,
                        if_select=self.obj.if_select, match_factor=self.obj.match_factor, threshold=self.obj.threshold)
        self.sinEnd.emit()
        self.obj.is_running = False


# 连续运行sfm线程
class RunWorker(QThread, QObject):
    sinStepEnd = pyqtSignal()  # 单步sfm线程结束信号

    def __init__(self, obj, model):
        super().__init__()
        self.obj = obj
        self.model = model

    def run(self) -> None:
        # sfm执行
        self.obj.is_running = True
        while self.model.check_sfm_completed() is False:
            self.model.step(if_local_BA=self.obj.if_local_BA, if_global_BA=self.obj.if_global_BA,
                            if_select=self.obj.if_select, match_factor=self.obj.match_factor,
                            threshold=self.obj.threshold)
            self.sinStepEnd.emit()
        self.obj.is_running = False


class QPlainTextEditLogger(logging.Handler):
    '''
    logging的处理器,多添加一个QPlainTextEdit,以记录日志传来时同步更新QPlainTextEdit以display。
    '''

    def __init__(self, parent):  # 这里的parent其实就是主窗口mainwindow
        super().__init__()
        self.widget = QPlainTextEdit(parent)
        self.widget.setFont(QFont("楷体", 9, QFont.Normal))
        self.widget.setReadOnly(True)
        self.widget.textChanged.connect(self.on_text_changed)

    # 当日志超过下端时,保证日志显示在最下方,即光标一直在最后
    def on_text_changed(self):
        self.widget.moveCursor(QTextCursor.End)

    def emit(self, record):
        '''
        重写logging.Handler的emit函数,使日志的处理器将记录器产生的日志记录发送到QPlainTextEdit里。

        :param record:接收的日志
        :return:
        '''
        msg = self.format(record)
        self.widget.appendPlainText(msg)


class MainUi(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        # 继承pyqt designer生成的ui类
        self.setupUi(self)
        # 初始化sfm相关默认参数
        self.if_local_BA = False
        self.if_global_BA = False
        self.if_select = True
        self.match_factor = 0.7
        self.threshold = 0.4
        self.downscale_factor = 1.0

        # 一些控制变量
        self.is_running = False  # 是否正在运行
        self.step_worker = None  # 用于执行单步sfm算法的线程
        self.run_worker = None  # 用于执行连续sfm算法的线程

        # 初始化所有控件
        self.initUI()
        self.initSlot()
        self.initLog()
        self.initStyle()

        logging.info('----------------------------------------')
        logging.info('Inplementation of increment sfm based on opencv and pyqt5')
        logging.info('Author:dawnzyt(github), qq: 1207068927')
        logging.info('Email:ziyu@bupt.edu.cn')
        logging.info('ps:本程序仅供学习交流使用...')
        logging.info('----------------------------------------')
        logging.info('')
        self.set_default_para_btn.click()

    def initUI(self):
        # 禁止最大化按钮
        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        # 禁止拉伸窗口大小
        self.setFixedSize(self.width(), self.height())
        # # logo label
        # self.logo_label = QLabel()
        # self.logo_layout.addWidget(self.logo_label)
        # self.set_logo(r'D:\JetBrains\Toolbox\PacharmProject\sfm-gui\imgs\bupt.png')

        # match_label: 用来展示匹配结果
        self.match_label = QLabel()
        self.match_layout.addWidget(self.match_label)
        self.set_match_label(np.zeros((100, 100, 3), dtype=np.uint8))

        # GLView_widget: 用来展示点云
        self.GLView_widget = GLViewWidget()
        self.points_layout.addWidget(self.GLView_widget)

        # 限定match_factor_text和threshold_text为double类型
        self.match_factor_text.setValidator(QDoubleValidator())
        self.ransac_t_text.setValidator(QDoubleValidator())
        self.downscale_factor_text.setValidator(QDoubleValidator())

        # init tooltips
        self.match_factor_hint_btn.setToolTip('sift匹配因子,(0,1), 越大匹配点越多; 建议分辨率高的图片设置小一点')
        self.ransac_t_hint_btn.setToolTip('ransac阈值,(0,1), 越大内点越多, 精确度越低, 建议遵循默认值')
        self.downscale_factor_hint_btn.setToolTip('降采样因子,大于等于1.0, 用来加速, 越大精确度越低')
        self.open_btn.setToolTip('打开图片集文件夹')
        self.run_btn.setToolTip('连续运行sfm算法')
        self.step_btn.setToolTip('单步运行sfm算法')
        self.stop_btn.setToolTip('停止运行sfm算法')
        self.save_parm_btn.setToolTip('保存参数')
        self.save_ply_btn.setToolTip('保存点云')
        self.set_default_para_btn.setToolTip('设置默认参数')
        self.localBA_checkBox.setToolTip('局部BA')
        self.globalBA_checkBox.setToolTip('全局BA')
        self.select_checkBox.setToolTip('剔除global异常点')

    def initStyle(self):
        self.setWindowTitle('Incremental SfM')
        self.setWindowOpacity(0.9)
        # sfm window icon
        self.setWindowIcon(qtawesome.icon('fa5s.camera-retro', color='red'))
        # self.setWindowIcon(QIcon('D:\JetBrains\Toolbox\PacharmProject\sfm-gui\imgs\Icon.ico'))
        # 不同objectName的控件设置不同的样式, 用于区分
        self.para_hint_btn.setObjectName('label_btn')
        self.operator_hint_btn.setObjectName('label_btn')
        self.match_factor_hint_btn.setObjectName('label_btn')
        self.ransac_t_hint_btn.setObjectName('label_btn')
        self.downscale_factor_hint_btn.setObjectName('label_btn')
        self.match_title_btn.setObjectName('label_btn')
        self.points_title_btn.setObjectName('label_btn')


        # 可操作的btn
        self.open_btn.setObjectName('btn')
        self.run_btn.setObjectName('btn')
        self.step_btn.setObjectName('btn')
        self.save_parm_btn.setObjectName('btn')
        self.set_default_para_btn.setObjectName('btn')
        self.save_ply_btn.setObjectName('btn')
        self.stop_btn.setObjectName('btn')

        self.setStyleSheet('''
            background:white;
        ''')
        self.centralwidget.setStyleSheet(
            '''
            QPushButton{color:black;border:none;}
            QPushButton#btn:hover{
                color:black;
                border-left:4px solid red;
                font-weight:700;
                border-right:3px solid black;
            }
            QPushButton#btn{
                color:darkGray;
                font-size:14px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            QPushButton#label_btn{
                font-size:14px;
                font-weight:500;
                font-family: "Helvetica Neue",Helvetica,Arial,"Microsoft Yahei","Hiragino Sans GB","Heiti SC","WenQuanYiMicro Hei",sans-serif;
            }
            '''
        )
        # 设置btn的图标
        # operator btn
        self.open_btn.setIcon(qtawesome.icon('fa.folder-open', color='red'))
        self.set_default_para_btn.setIcon(qtawesome.icon('fa.refresh', color='red'))
        self.save_parm_btn.setIcon(qtawesome.icon('ei.wrench', color='red'))
        self.save_ply_btn.setIcon(qtawesome.icon('fa.save', color='red'))
        self.run_btn.setIcon(qtawesome.icon('fa.play', color='red'))
        self.step_btn.setIcon(qtawesome.icon('fa.step-forward', color='red'))
        self.stop_btn.setIcon(qtawesome.icon('fa.stop', color='red'))

        # label btn
        self.match_title_btn.setIcon(qtawesome.icon('fa.image', color='red'))
        self.points_title_btn.setIcon(qtawesome.icon('fa.cubes', color='red'))
        self.para_hint_btn.setIcon(qtawesome.icon('msc.settings-gear', color='red'))
        self.operator_hint_btn.setIcon(qtawesome.icon('fa5s.tools', color='red'))
        self.match_factor_hint_btn.setIcon(qtawesome.icon('fa.question-circle', color='red'))
        self.ransac_t_hint_btn.setIcon(qtawesome.icon('fa.question-circle', color='red'))
        self.downscale_factor_hint_btn.setIcon(qtawesome.icon('fa.question-circle', color='red'))
        self.watermark_btn.setIcon(qtawesome.icon('fa.github', color='black'))

        # 设置cursor为Qt.PointingHandCursor
        self.open_btn.setCursor(Qt.PointingHandCursor)
        self.set_default_para_btn.setCursor(Qt.PointingHandCursor)
        self.save_parm_btn.setCursor(Qt.PointingHandCursor)
        self.save_ply_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.step_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setCursor(Qt.PointingHandCursor)

    def initLog(self):
        '''
        初始化log相关方法;下面为发送日志;具体反应回hander.emit
        logging.debug('damn, a bug')
        logging.info('something to remember')
        logging.warning('that\'s not right')
        logging.error('foobar')
        :return:
        '''
        self.logTextEdit = QPlainTextEditLogger(self)
        # 日志格式为时间+message
        # logging的fmt包括:
        # %(asctime)s:时间
        # %(levelname)s:日志级别名称即logging.info(),logging.debug()等
        # %(message)s:日志信息
        self.logTextEdit.setFormatter(
            logging.Formatter(fmt='[%(asctime)s]-%(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logging.getLogger().addHandler(self.logTextEdit)
        # You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)
        self.log_layout.addWidget(self.logTextEdit.widget)

    def initSlot(self):
        self.save_parm_btn.clicked.connect(self.save_parm_btn_clicked)
        self.open_btn.clicked.connect(self.open_btn_clicked)
        self.set_default_para_btn.clicked.connect(self.set_default_para_btn_clicked)

        self.step_btn.clicked.connect(self.step_btn_clicked)
        self.run_btn.clicked.connect(self.run_btn_clicked)
        self.stop_btn.clicked.connect(self.stop_btn_clicked)
        self.save_ply_btn.clicked.connect(self.save_ply_btn_clicked)

    def set_logo(self, img_path):
        img = cv2.imread(img_path)
        zoom_img = cv2.cvtColor(cv2.resize(img, (350, 128)), cv2.COLOR_BGR2RGB)
        qImage = QImage(zoom_img.data, zoom_img.shape[1], zoom_img.shape[0], zoom_img.shape[1] * 3,
                        QImage.Format_RGB888)
        self.logo_label.setPixmap(QPixmap.fromImage(qImage))

    def set_match_label(self, img):
        zoom_img = cv2.cvtColor(cv2.resize(img, (649, 369)), cv2.COLOR_BGR2RGB)
        qImage = QImage(zoom_img.data, zoom_img.shape[1], zoom_img.shape[0], zoom_img.shape[1] * 3,
                        QImage.Format_RGB888)
        self.match_label.setPixmap(QPixmap.fromImage(qImage))

    @pyqtSlot()
    def save_parm_btn_clicked(self):
        self.if_local_BA = self.localBA_checkBox.isChecked()
        self.if_global_BA = self.globalBA_checkBox.isChecked()
        self.if_select = self.select_checkBox.isChecked()
        match_factor = self.match_factor_text.text().strip()
        threshold = self.ransac_t_text.text().strip()
        downscale_factor = self.downscale_factor_text.text().strip()
        if not match_factor or not threshold or not downscale_factor:
            logging.warning('match_factor or threshold or downscale_factor is empty')
            QMessageBox.warning(self, '警告', '请填写完整参数')
            return
        self.match_factor = float(match_factor)
        self.threshold = float(threshold)
        self.downscale_factor = float(downscale_factor)
        logging.info('save parameters success')

    @pyqtSlot()
    def open_btn_clicked(self):
        self.img_path = QFileDialog.getExistingDirectory(self, '选择数据集(注意:目录下包含.png/.jpg格式的图像集和内参K.txt)', r'./')
        if not self.img_path:
            return
        # 清空log text
        self.logTextEdit.widget.clear()
        logging.info('open images dataset %s' % self.img_path)
        # 锁定downscale_factor_text
        self.downscale_factor_text.setReadOnly(True)
        # 将match_label清空, GLView_widget清空
        self.set_match_label(np.zeros((100, 100, 3), dtype=np.uint8))
        self.GLView_widget.clear()
        # 初始化新的sfm model
        if self.model is not None:
            del self.model
        self.model = sfm.IncrementalSFM(self.img_path, downscale_factor=self.downscale_factor,
                                        log_handler=self.logTextEdit)

    @pyqtSlot()
    def set_default_para_btn_clicked(self):
        # 设置默认参数, 包括if_local_BA, if_global_BA, match_factor, threshold, downscale_factor
        self.localBA_checkBox.setChecked(False)
        self.globalBA_checkBox.setChecked(False)
        self.select_checkBox.setChecked(True)
        self.match_factor_text.setText('0.7')
        self.ransac_t_text.setText('0.4')
        self.downscale_factor_text.setText('1.0')
        logging.info('set default parameters success')

    @pyqtSlot()
    def step_btn_clicked(self):
        # step单步执行sfm
        if self.model is None:
            logging.warning('please open images dataset first')
            QMessageBox.warning(self, '警告', '请先打开图片集')
            return
        if self.model.check_sfm_completed():
            logging.info('sfm finished')
            QMessageBox.information(self, '提示', 'sfm已经结束')
            return
        # 定义单步执行线程
        self.step_worker = StepWorker(self, self.model)
        self.step_worker.sinEnd.connect(self.step_worker_finished)
        self.step_worker.start()
        # self.model.step(if_local_BA=self.if_local_BA, if_global_BA=self.if_global_BA, if_select=self.if_select,
        #                 match_factor=self.match_factor, threshold=self.threshold)

    @pyqtSlot()
    def run_btn_clicked(self):
        if self.model is None:
            logging.warning('please open images dataset first')
            QMessageBox.warning(self, '警告', '请先打开图片集')
            return
        if self.model.check_sfm_completed():
            logging.info('sfm finished')
            QMessageBox.information(self, '提示', 'sfm已经结束')
            return
        # sfm执行
        self.run_worker = RunWorker(self, self.model)
        self.run_worker.sinStepEnd.connect(self.step_worker_finished)
        self.run_worker.start()

    @pyqtSlot()
    def save_ply_btn_clicked(self):
        # 保存点云
        path = QFileDialog.getSaveFileName(self, '保存点云', './', '*.ply')
        if not path[0]:
            logging.warning('path is None')
            QMessageBox.warning(self, '警告', '请先设置path')
            return
        self.model.save(path[0])
        logging.info('save point cloud success')

    @pyqtSlot()
    def stop_btn_clicked(self):
        """
        强行停止sfm的单步线程或者连续运行线程

        :return:
        """
        if self.is_running is False:
            logging.warning('sfm is not running')
            QMessageBox.warning(self, '警告', 'sfm未运行')
            return
        if self.step_worker is not None:
            self.step_worker.terminate()
        if self.run_worker is not None:
            self.run_worker.terminate()
        self.is_running = False
        logging.info('sfm stopped')
        # 提示重新打开图片集
        QMessageBox.information(self, '提示', '请重新打开图片集')

    @pyqtSlot()
    def step_worker_finished(self):
        match = self.model.img_matches
        if match is not None:
            self.set_match_label(match)
        if self.model.new_points is not None:
            new_points = self.model.new_points
            plot = gl.GLScatterPlotItem()  # 创建显示对象
            plot.setData(pos=new_points, color=(1, 1, 1, 1), size=0.001, pxMode=False)  # 设置显示数据
            self.GLView_widget.addItem(plot)  # 显示点云
        if self.model.state == 3:
            self.downscale_factor_text.setReadOnly(False)

    # 重写事件过滤器, 保证在sfm运行时, 按钮不可点击, 除了stop_btn
    def eventFilter(self, obj, event):
        # 当self.is_running为True时, 拦截按钮单击事件
        if self.is_running and event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if obj.objectName() == 'btn':
                # 如果是stop_btn可以点击
                if obj == self.stop_btn:
                    pass
                else:
                    logging.warning('please wait for sfm finished')
                    QMessageBox.warning(self, '警告', '请等待sfm结束')
        return super().eventFilter(obj, event)  # 其他事件正常处理


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MainUi()
    demo.show()
    app.installEventFilter(demo)  # 为app安装事件过滤器
    sys.exit(app.exec_())

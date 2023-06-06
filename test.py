import sys
import open3d as o3d
import numpy as np
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QPushButton, QWidget, QFileDialog
from pyqtgraph.opengl import GLViewWidget


class PyQtGraphicDemo(QWidget):
    def __init__(self, parent=None):
        super(PyQtGraphicDemo, self).__init__(parent)

        # 点云显示控件
        self.graphicsView = GLViewWidget(self)
        # 按钮
        self.pushButton = QPushButton(self)
        self.pushButton.setText("PushButton")
        self.pushButton.clicked.connect(self.showCloud)
        # 布局
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.addWidget(self.graphicsView)
        self.verticalLayout.addWidget(self.pushButton)
        self.setLayout(self.verticalLayout)

    def showCloud(self):
        fileName, filetype = QFileDialog.getOpenFileName(self, "请选择点云：", '.', "cloud Files(*pcd *ply)")
        if fileName != '':
            pcd = o3d.io.read_point_cloud(fileName) #读取点云
            np_points = np.asarray(pcd.points)  #获取Numpy数组
            plot = gl.GLScatterPlotItem() #创建显示对象
            plot.setData(pos=np_points, color=(1, 1, 1, 1), size=0.001, pxMode=False) #设置显示数据
            self.graphicsView.addItem(plot) #显示点云


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PyQtGraphicDemo()
    window.show()
    sys.exit(app.exec_())

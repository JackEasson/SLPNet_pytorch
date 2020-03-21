"""
Thanks for the PyQt5 code of Chainsmokers from CSDN:
https://blog.csdn.net/DerrickRose25/article/details/
86744787?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
"""
import sys
from cv2 import resize, imread, imwrite, imdecode
from numpy import ndarray, fromfile, uint8
from img_process import image_det_reg_process, model_initial
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton


# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filepath):
    cv_img = imdecode(fromfile(filepath, dtype=uint8), -1)
    return cv_img


class win(QDialog):
    def __init__(self):
        self.SLPNet_model = model_initial()
        # 初始化一个img的ndarry，用于存储图像
        self.img_initial = ndarray(())
        self.img_for_show = ndarray(())
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 400)
        self.setWindowTitle('SLPNet for ALPR')
        self.btnOpen = QPushButton('Open', self)
        self.btnSave = QPushButton('Save', self)
        self.btnProcess = QPushButton('Process', self)
        self.btnQuit = QPushButton('Quit', self)
        self.label = QLabel()

        # 布局设定
        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 1, 3, 4)
        layout.addWidget(self.btnOpen, 4, 1, 1, 1)
        layout.addWidget(self.btnProcess, 4, 2, 1, 1)
        layout.addWidget(self.btnSave, 4, 3, 1, 1)
        layout.addWidget(self.btnQuit, 4, 4, 1, 1)

        # 信号与槽进行连接，信号可绑定普通成员函数
        self.btnOpen.clicked.connect(self.openSlot)
        self.btnSave.clicked.connect(self.saveSlot)
        self.btnProcess.clicked.connect(self.processSlot)
        self.btnQuit.clicked.connect(self.close)

    def openSlot(self):
        max_len = 800
        # 调用存储文件
        fileName, tmp = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        if fileName is '':
            return
        print('Now process image: %s' % fileName.split('/')[-1])
        # 采用OpenCV函数读取数据
        self.img_initial = imread(fileName, -1)
        if self.img_initial.size == 1:
            return
        img_h, img_w = self.img_initial.shape[:2]
        if img_h > img_w and img_h > max_len:
            tar_h = max_len
            tar_w = int(max_len * (img_w / img_h))
            self.img_for_show = resize(self.img_initial, (tar_w, tar_h))
        elif img_w > img_h and img_w > max_len:
            tar_w = max_len
            tar_h = int(max_len * (img_h / img_w))
            self.img_for_show = resize(self.img_initial, (tar_w, tar_h))
        else:
            self.img_for_show = self.img_initial
        self.refreshShow()

    def saveSlot(self):
        # 调用存储文件dialog
        fileName, tmp = QFileDialog.getSaveFileName(self, 'Save Image', 'Image', '*.png *.jpg *.bmp')
        if fileName is '':
            return
        if self.img_for_show.size == 1:
            return
        # 调用OpenCV写入函数
        imwrite(fileName, self.img_for_show)

    def processSlot(self):
        if self.img_for_show.size == 1:
            return
        # 对图像做模糊处理，窗口设定为5*5
        # self.img = cv2.blur(self.img, (5, 5))
        self.img_for_show = image_det_reg_process(self.SLPNet_model, self.img_initial, self.img_for_show)
        self.refreshShow()

    def refreshShow(self):
        # 提取图像的通道和尺寸，用于将OpenCV下的image转换成Qimage
        height, width, channel = self.img_for_show.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img_for_show.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        # 将QImage显示出来
        self.label.setPixmap(QPixmap.fromImage(self.qImg))


if __name__ == '__main__':
    a = QApplication(sys.argv)
    w = win()
    w.show()
    sys.exit(a.exec_())

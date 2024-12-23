# 作者：弗如
# 日期：2024年12月20日

import math
import os
import sys
import time
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMainWindow, QFileDialog
from PyQt5.QtGui import QPainter, QImage, QPixmap, QPen
from PyQt5.QtCore import Qt
from matplotlib import pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from model import LeNet

# 超参数
WINDOW_WIDTH = 450  # 窗口宽度
WINDOW_HEIGHT = 400  # 窗口高度
CANVAS_HEIGHT = 300  # 画板高
CANVAS_WIDTH = 300  # 画板宽
Pen_width = 25 # 画笔粗细
Fig_size = (15, 5) # 特征图窗口大小

# 定义画板类
class Canvas(QWidget):
    """
    自定义绘图区域，用于绘制手写数字。
    """
    def __init__(self, canvas_width=CANVAS_WIDTH, canvas_height=CANVAS_HEIGHT):
        super().__init__()
        self.image = QImage(canvas_width, canvas_height, QImage.Format_RGB32)  # 创建一个与指定大小相同的图像，格式为RGB32
        self.image.fill(Qt.white)  # 将图像填充为白色背景
        self.draw = False
        self.load = False

    def mousePressEvent(self, event):
        """
        鼠标按下事件，开始绘制。
        """
        if event.button() == Qt.LeftButton:  # 检查是否为左键点击
            self.lastPos = event.pos()  # 记录鼠标按下时的位置
            self.draw = True

    def mouseMoveEvent(self, event): 
        """
        鼠标移动事件，继续绘制。
        """
        if event.buttons() and Qt.LeftButton:  # 检查是否为左键按下且正在绘制
            imagePainter = QPainter(self.image)  # 创建一个QPainter对象，用于在QImage上绘制
            pen = QPen(Qt.black, Pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)  # 设置画笔颜色、宽度、线型和端点样式
            imagePainter.setPen(pen)  # 应用画笔设置
            imagePainter.drawLine(self.lastPos, event.pos())  # 在上次鼠标位置和当前鼠标位置之间绘制一条线。每一步绘制都会保存到QImage中，不会清除
            self.lastPos = event.pos()  # 更新上次鼠标位置
            self.update() # 更新绘图区域，触发重绘（调用paintEvent)。因为绘图操作是作用在QImage上的，需要把QImage上的图像显示到窗口里。

    def paintEvent(self, event):
        """
        重载paintEvent方法，绘制图像。
        """
        canvasPainter = QPainter(self)  # 创建一个QPainter对象，用于在窗口上绘制
        canvasPainter.drawImage(0, 0, self.image)  # 将图像绘制到窗口的指定位置

    def save_image(self, path):
        """
        保存当前绘制的图像到指定路径。
        """
        self.image.save(path)  # 保存图像到指定路径

    def get_image(self):
        """
        获取当前绘制的图像。
        """
        return self.image  # 返回当前图像

    def clear(self):
        """
        清除绘图区域。
        """
        self.image.fill(Qt.white)  # 将图像填充为白色背景
        self.update()  # 更新绘图区域，触发重绘
        self.draw = False
        self.load = False

class Gui(QMainWindow):
    """主窗口"""
    def __init__(self, model, transform):
        super().__init__()
        self.setWindowTitle("手写数字识别")  # 设置窗口标题
        self.setGeometry(500, 300, WINDOW_WIDTH, WINDOW_HEIGHT)  # 设置窗口位置和大小
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)  # 设置窗口固定大小

        self.canvas = Canvas()  # 创建绘图区域并传递大小参数
        self.canvas.setFixedSize(CANVAS_WIDTH, CANVAS_HEIGHT)  # 设置绘图区域固定大小
        self.label = QLabel("识别为：")  # 创建结果显示标签

        btn_recognize = QPushButton("识别", self)  # 创建“识别”按钮
        btn_recognize.clicked.connect(self.recognize)  # 连接按钮点击事件到识别方法
        btn_clear = QPushButton("清除", self)  # 创建“清除”按钮
        btn_clear.clicked.connect(self.clear_canvas)  # 连接按钮点击事件到清除方法
        btn_save = QPushButton("保存手写", self)  # 创建“保存手写”按钮
        btn_save.clicked.connect(self.save_handwriting)  # 连接按钮点击事件到保存手写方法
        btn_load = QPushButton("从文件加载", self)  # 创建“从文件加载”按钮
        btn_load.clicked.connect(self.load_from_file)  # 连接按钮点击事件到从文件加载方法
        btn_fm = QPushButton("查看特征图", self)  # 创建“特征图”按钮
        btn_fm.clicked.connect(self.feature_map)  # 连接按钮点击事件到特征图方法

        layout_btn = QVBoxLayout()
        layout_btn.setContentsMargins(20, 30, 10, 100) # 设置layout_btn的边距，调整按钮位置
        layout_btn.addWidget(btn_recognize)  # 将“识别”按钮添加到按钮布局中
        layout_btn.addWidget(btn_clear)  # 将“清除”按钮添加到按钮布局中
        layout_btn.addWidget(btn_save)  # 将“保存手写”按钮添加到按钮布局中
        layout_btn.addWidget(btn_load)  # 将“从文件加载”按钮添加到按钮布局中
        layout_btn.addWidget(btn_fm)

        layout_canvas = QVBoxLayout()
        layout_canvas.addWidget(self.canvas)  # 将画板添加到画板布局中
        layout_canvas.addWidget(self.label)  # 将结果显示标签添加到画板布局中

        layout_main = QHBoxLayout()  # 创建垂直布局管理器
        layout_main.addLayout(layout_canvas)
        layout_main.addLayout(layout_btn)

        container = QWidget()  # 创建一个容器部件
        container.setLayout(layout_main)  # 将布局应用到容器部件
        self.setCentralWidget(container)  # 将容器部件设置为主窗口的中心部件

        self.transform = transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.model.eval()  # 将模型设置为评估模式
        
        self.recognized = False
        self.img = None

    def recognize(self):
        """
        识别手写数字并更新结果显示标签。
        """
        if self.canvas.draw or self.canvas.load:  # 检查图像是否为空
            image = self.canvas.image  # 获取绘图区域的图像
            temp_path = f"temp_{int(time.time())}.png"  # 使用时间戳确保文件名唯一
            image.save(temp_path)  # 将图像保存到临时文件
            img = Image.open(temp_path)  # 使用PIL库打开临时文件
            img = img.convert('L')  # 将图像转换为灰度图像
            img = Image.eval(img, lambda x: 255 - x)  # 对图像进行反色处理
            img = self.transform(img).to(self.device)
            self.img = torch.reshape(img, (1, 1, 28, 28))  # 重塑图像张量
            
            with torch.no_grad():  # 关闭梯度计算
                output = self.model(self.img)  # 使用模型进行预测
            result = int(output.argmax())  # 获取预测结果的最大值索引
            self.label.setText(f"识别为: {result}")  # 更新结果显示标签
            os.remove(temp_path)  # 删除临时文件
            self.recognized = True

    def clear_canvas(self):
        """
        清空画板
        """
        self.canvas.clear()
        self.label.setText("识别为: ")
        self.recognized = False
    
    def save_handwriting(self):
        """
        保存手写图像到文件。
        """
        options = QFileDialog.Options()  # 创建文件对话框选项
        file_path, _ = QFileDialog.getSaveFileName(self, "保存手写", "", "Images (*.png *.xpm);;All Files (*)", options=options)  # 打开文件保存对话框
        if file_path:  # 检查是否选择了文件路径
            self.canvas.image.save(file_path)  # 保存绘图区域的图像到指定路径
            print(f"手写已保存到 {file_path}")  # 打印保存路径

    def load_from_file(self):
        """
        从文件加载图像到绘图区域。
        """
        options = QFileDialog.Options()  # 创建文件对话框选项
        file_path, _ = QFileDialog.getOpenFileName(self, "从文件加载", "", "Images (*.jpg *.png);;All Files (*)", options=options)  # 打开文件打开对话框
        if file_path:  # 检查是否选择了文件路径
            pixmap = QPixmap(file_path)  # 创建QPixmap对象，加载图像
            scaled_pixmap = pixmap.scaled(self.canvas.width(), self.canvas.height(), Qt.KeepAspectRatio)  # 缩放图像以适应绘图区域
            self.canvas.image = scaled_pixmap.toImage()  # 将缩放后的图像转换为QImage并赋值给绘图区域
            self.canvas.update()
            self.canvas.load = True
    
    def feature_map(self):
        """
        展示卷积层和池化层的特征图
        """
        if self.recognized == True:
            feature_map = []
            layer_name = []
            conv_num = 1
            pool_num = 1
            layers = list(self.model.model)
            output = self.img
            with torch.no_grad():
                for layer in layers:
                    output = layer(output)
                    if isinstance(layer, nn.Conv2d):
                        feature_map.append(output)
                        output_size = output.shape
                        layer_name.append("Conv_layer {}, output_size: {}".format(conv_num, tuple(output_size[-3:])))
                        conv_num += conv_num
                    if isinstance(layer, nn.AvgPool2d):
                        feature_map.append(output)
                        output_size = output.shape
                        layer_name.append("Pool_layer {}, output_size: {}".format(pool_num, tuple(output_size[-3:])))
                        pool_num += pool_num

            num_rows = len(feature_map)
            plt.figure(figsize=Fig_size)
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            # 遍历每一层的特征图
            for i, (fm, name) in enumerate(zip(feature_map, layer_name)):
                num_channels = fm.shape[1] 
                # 遍历当前层的所有通道
                for j in range(num_channels):
                    img = transforms.ToPILImage()(fm[0, j])
                    # 重新定义子图，使整张图中没有空图
                    ax = plt.subplot(num_rows, num_channels, i*num_channels + j + 1)
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    if j == math.floor(num_channels/2 - 1):
                        ax.set_title(name)
            # 显示图像
            plt.show()
# test.py
# 作者：弗如
# 时间：2024年12月18日

import sys
import torchvision.transforms as transforms
import torch
from model import LeNet
from handwriting_board import HandwritingBoard, QLabel
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)

    # 初始化模型
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    model.load_state_dict(torch.load("lenet98.06.pth1"))
    model.eval()

    # 初始化标签
    label = QLabel("结果: ")

    # 启动手写板
    window = HandwritingBoard(model, transform, device, label)
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
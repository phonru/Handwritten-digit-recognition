# 定义模型
from torch import nn
import torch

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 卷积层，输入大小为1x28x28，输出大小为6x28x28
            nn.Conv2d(1, 6, 5, padding = 2), nn.Sigmoid(),
            # 平均池化，输入6x28x28，输出6x14x14
            nn.AvgPool2d(2),
            # 卷积层，输入6x14x14，输出16x10x10
            nn.Conv2d(6, 16, 5), nn.Sigmoid(),
            # 平均池化，输入16x10x10，输出16x5x5
            nn.AvgPool2d(2),
            # 展平
            nn.Flatten(1),
            # 三次全连接
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        return self.model(x)

if __name__=='__main__':
    lenet = LeNet()
    input = torch.ones((64, 1 , 28, 28))
    output = lenet(input)
    print(output.shape)
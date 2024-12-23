# 定义模型
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

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
    
model = LeNet()
layers = list(model.model)
feature_map = []
layer_name = []
output = torch.ones((1, 1, 28, 28))
with torch.no_grad():
    for layer in layers:
        output = layer(output)
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.AvgPool2d):
            feature_map.append(output)
            layer_name.append(str(layer))
# print(feature_map)
# print(layer_name)

# 计算需要的子图数量
# num_rows: 特征图的层数，即 feature_map 列表的长度
# num_cols: 每层特征图的最大通道数，用于确定子图的列数
num_rows = len(feature_map)
num_cols = max([fm.shape[1] for fm in feature_map])

# 创建子图
# 使用 plt.subplots 创建一个包含 num_rows 行和 num_cols 列的子图网格
# figsize 参数设置图像大小为 (20, 15)，以确保有足够的空间显示所有子图
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))

# 遍历每一层的特征图
for i, (fm, name) in enumerate(zip(feature_map, layer_name)):
    # 获取当前层特征图的通道数
    num_channels = fm.shape[1]
    
    # 遍历当前层的所有通道
    for j in range(num_channels):
        # 根据子图的列数选择合适的轴对象
        # 如果有多个列，则使用 axes[i, j]，否则直接使用 axes[i]
        ax = axes[i, j] if num_cols > 1 else axes[i]
        
        # 显示特征图
        # 使用 imshow 方法显示特征图，并将颜色映射设置为灰色
        # fm[0, j].cpu().numpy() 将张量转换为 numpy 数组并移动到 CPU 上
        ax.imshow(fm[0, j].cpu().numpy(), cmap='gray')
        
        # 关闭坐标轴以保持整洁
        ax.axis('off')
        
        # 只在第一列设置标题，标题为当前层的名称
        if j == 0:
            ax.set_title(name)

    # 去掉多余的子图
    # 如果某一层的通道数小于最大通道数，则删除多余的子图，避免空白区域
    for j in range(num_channels, num_cols):
        fig.delaxes(axes[i, j])

# 自动调整子图参数，使之填充整个图像区域
plt.tight_layout()

# 显示图像
plt.show()
# 作者：弗如
# 日期：2024年12月23日

import math
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
from model import *
import torch

# 超参数
Fig_size = (15, 5)
weight_path = "model_weight/lenet98.46.pth"

# 定义数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
# train_dataset =  torchvision.datasets.MNIST(root='./Dataset', train=True, transform = transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./Dataset', train=False, transform = transform, download=True)

img_tensor, _ = test_dataset[1]

# # 查看原图
# img_PIL = transforms.ToPILImage()(img_tensor)
# img_PIL.show()


# 查看特征图
model = LeNet()
model.load_state_dict(torch.load(weight_path, weights_only=True))  # 加载模型参数

feature_map = []
layer_name = []
conv_num = 1
pool_num = 1
layers = list(model.model)
output = torch.reshape(img_tensor, (1, 1, 28, 28))
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
# plt.tight_layout()
plt.show()
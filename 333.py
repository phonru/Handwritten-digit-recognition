# 作者：弗如
# 日期：2024年12月23日

import math
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
import torch
import torch.nn.init as init

# 超参数
Fig_size = (15, 5)
weight_path = "model_weight/lenet98.46.pth"

# 定义数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
train_dataset =  torchvision.datasets.MNIST(root='./Dataset', train=True, transform = transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./Dataset', train=False, transform = transform, download=True)

img_tensor, _ = test_dataset[1]
print(img_tensor)
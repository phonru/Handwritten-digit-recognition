# 作者：弗如
# 时间：2024年12月18日

import torchvision.transforms as transforms
import torch
from PIL import Image
from model import LeNet
from MyDataset import *
from torch.utils.tensorboard import SummaryWriter

# 测试参数
img_path = "imgs/9.jpg"
in_channel = 1
H_size = 28
W_size = 28
model_name = "lenet98.06.pth1" # 模型参数文件
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 定义设备

# 定义数据集
mydata = MyDataset("mydata")

#定义模型
lenet = LeNet()
lenet.load_state_dict(torch.load(model_name, weights_only=True))
lenet.to(device)

# 测试
writer = SummaryWriter("logs")
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print("\tlabel\tprediction")
num = 1
for data in mydata:
    img, label = data
    img = img.convert('L') #转为灰度图片
    img = Image.eval(img, lambda x: 255 - x) #反色
    transform = transforms.Compose([transforms.Resize((H_size, W_size)), transforms.ToTensor()])
    img = transform(img).to(device)
    img = torch.reshape(img, (1, in_channel, H_size, W_size))

    lenet.eval()
    with torch.no_grad():
        output = lenet(img)
    predict = classes[int(output.argmax())]
    writer.add_images("预测结果为：{}".format(predict), img, 0)
    print("图片{}:\t{}\t{}".format(num, label, predict))
writer.close()


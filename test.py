# 作者：弗如
# 时间：2024年12月18日

import torchvision.transforms as transforms
import torch
from PIL import Image
from model import LeNet

# 测试参数
img_path = "imgs/9.jpg"
in_channel = 1
H_size = 28
W_size = 28
model_name = "lenet98.06.pth1" # 模型参数文件
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 定义设备

img = Image.open(img_path)
img = img.convert('L') #转为灰度图片
img = Image.eval(img, lambda x: 255 - x) #反色
transform = transforms.Compose([transforms.Resize((H_size, W_size)), transforms.ToTensor()])
img = transform(img).to(device)
img = torch.reshape(img, (1, in_channel, H_size, W_size))


lenet = LeNet()
lenet.load_state_dict(torch.load(model_name, weights_only=True))
lenet.to(device)

lenet.eval()
with torch.no_grad():
    output = lenet(img)

# print(output)
# classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print("这张图片是:{}".format(int(output.argmax())))

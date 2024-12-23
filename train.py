# 作者：弗如
# 日期：2024年12月18日

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
import torch
import torch.nn.init as init

# 超参数
batch_size = 256
learning_rate = 0.1
epoch = 20
momentum = 0.9
weight_decay=0.0001

# 定义数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
train_dataset =  torchvision.datasets.MNIST(root='./Dataset', train=True, transform = transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./Dataset', train=False, transform = transform, download=True)
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)

# 定义加载器
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

# 定义设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义模型
lenet = LeNet()
lenet.to(device)
# 初始化权重
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
lenet.apply(weights_init)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 定义优化器
optim = torch.optim.SGD(lenet.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# 训练
writer = SummaryWriter('logs')
for n in range(epoch):
    # 训练
    print("----------第{}/{}轮训练----------".format(n + 1, epoch))
    lenet.train()
    loss_train_total = 0
    for data in train_dataloader:
        img, target = data
        if torch.cuda.is_available():
            img = img.cuda()
            target = target.cuda()
        # 前向传播
        outputs = lenet(img)
        loss = loss_fn(outputs, target)
        loss_train_total += loss
        # 反向传播
        optim.zero_grad()
        loss.backward()
        optim.step()
    writer.add_scalar("loss_train_total", loss_train_total/train_dataset_size, n)
    print("训练损失为：{}".format(loss_train_total/train_dataset_size))

    # 测试
    lenet.eval()
    loss_test_total = 0
    accuracy_total = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, target = data
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            # 前向传播
            outputs = lenet(img)
            loss = loss_fn(outputs, target)
            loss_test_total += loss
            accuracy = (outputs.argmax(1) == target).sum()
            accuracy_total += accuracy
        print("测试损失为：{}".format(loss_test_total/test_dataset_size))
        print("正确率为：{:.2f}%".format(float(accuracy_total/test_dataset_size*100)))
        writer.add_scalar("loss_test_total", loss_test_total/test_dataset_size, n)
        writer.add_scalar("accuracy_total", accuracy_total/test_dataset_size*100, n)
writer.close()

# 保存
save_path = "model_weight/lenet{:.2f}.pth".format(float(accuracy_total/test_dataset_size*100))
torch.save(lenet.state_dict(), save_path)
print("模型已保存为：{}".format(save_path))

from PyQt5.QtWidgets import QApplication
from model import *
from GUI import *

weight_path = "model_weight/lenet98.46.pth"
H_size = 28
W_size = 28

model = LeNet()
model.load_state_dict(torch.load(weight_path, weights_only=True))  # 加载模型参数
transform = transforms.Compose([transforms.Resize((H_size, W_size)), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

app = QApplication(sys.argv)
canvas = Gui(model, transform)
canvas.show()
app.exec_()
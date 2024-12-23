import os
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir = None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        if self.label_dir == None:
            self.path = self.root_dir
        else:
            self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_name_list = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        if self.label_dir == None:
            img_path = os.path.join(self.root_dir, img_name)
        else:
            img_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_path)
        label = (os.path.splitext(img_name)[0] if self.label_dir == None else self.label_dir)
        return img, label
    
    def __len__(self): return len(self.img_name_list)

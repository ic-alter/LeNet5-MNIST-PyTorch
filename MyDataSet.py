import os

from torch.utils.data import Dataset
import cv2
import pickle
import numpy as np
import torch
from torchvision.transforms import ToTensor

class MyDataset(Dataset):
    def __init__(self, base_dir, label_dir):
        self.base_dir = base_dir
        self.label_dir = label_dir
        self.img_paths = os.listdir(os.path.join(self.base_dir,self.label_dir))

    def img2numpy(self, filename):
        img = cv2.imread(filename, flags=0)
        #cv2.imshow('imshow',img)
        #cv2.waitKey(0)
        return np.expand_dims(img/255,axis=0)


    def __getitem__(self, idx):
        img_name = self.img_paths[idx]
        img_fullpath = os.path.join(self.base_dir, self.label_dir, img_name)
        image = self.img2numpy(img_fullpath)
        label = int(self.label_dir)-1
        return image, label

    def __len__(self):
        return len(self.img_paths)
    

if __name__ == '__main__':
    dataset = MyDataset('train_data/train','1')
    for i in range(2,13):
        dataset += MyDataset('train_data/train',str(i))
    print(torch.Size(dataset))

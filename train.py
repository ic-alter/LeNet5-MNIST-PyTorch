from Lenet5 import Lenet5
from MyDataSet import MyDataset
import numpy as np
import os
import datetime
import torch
import cv2
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

class PhotoModel(object):
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 256
        self.model = Lenet5().to(self.device)

    def load_data(self):
        self.full_dataset = MyDataset('train_data/train','1')
        for i in range(2,13):
            self.full_dataset += MyDataset('train_data/train',str(i))

    def train(self,generation = 100):
        # 划分训练集测试集
        train_size = int(0.8 * len(self.full_dataset))
        test_size = len(self.full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.full_dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        self.model = Lenet5().to(self.device)
        # 随机梯度下降
        sgd = SGD(self.model.parameters(), lr=1e-1)
        # 交叉熵
        loss_fn = CrossEntropyLoss()
        # 训练100代
        for i in range(generation):
            self.model.train()
            for idx, (train_x, train_label) in enumerate(train_loader):
                train_x = train_x.to(self.device)
                train_label = train_label.to(self.device)
                sgd.zero_grad()
                predict_y = self.model(train_x.float())
                loss = loss_fn(predict_y, train_label.long())
                loss.backward()
                sgd.step()
                # 输出本轮的准确率
            self.test(test_loader)

    def test(self, test_loader):
        all_correct_num = 0
        all_sample_num = 0
        self.model.eval()
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(self.device)
            test_label = test_label.to(self.device)
            predict_y = self.model(test_x.float()).detach()
            predict_y =torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        self.right_rate = all_correct_num / all_sample_num
        print('准确率: {}'.format(self.right_rate), flush=True)

    def predict(self, pred_loader):
        self.model.eval()
        for idx, (test_x, test_label) in enumerate(pred_loader):
            test_x = test_x.to(self.device)
            test_label = test_label.to(self.device)
            predict_y = self.model(test_x.float()).detach()
            predict_y =torch.argmax(predict_y, dim=-1)
            print(predict_y)

    def save_model(self):
        t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename = 'models/{}.{}.pt'.format(t, self.right_rate)
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(self.model, filename)

    def load_model(self, filename):
        self.model = torch.load(filename)

    # 展示所有数据的结果
    def show(self):
        loader = DataLoader(self.full_dataset, batch_size=self.batch_size)
        self.test(loader)


def train_new():
    pm = PhotoModel()
    pm.load_data()
    pm.train(100)
    pm.save_model()

def show():
    pm = PhotoModel()
    pm.load_data()
    pm.load_model('models/2023-10-18-13-43-03.0.9670698924731183.pt')
    pm.show()

def predict():
    pm = PhotoModel()
    pm.load_model('models/2023-10-18-13-43-03.0.9670698924731183.pt')
    dataset = MyDataset('train_data/train','10')
    pred_loader = DataLoader(dataset, batch_size=256)
    print(pm.predict(pred_loader))


if __name__ == '__main__':
    #train_new()
    show()
    predict()

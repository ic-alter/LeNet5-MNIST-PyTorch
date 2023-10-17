from Lenet5 import Lenet5
from MyDataSet import MyDataset
import numpy as np
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

class PhotoModel(object):
    def __init__(self) -> None:
        pass

    def load_data(self):
        self.full_dataset = MyDataset('train_data/train','1')
        for i in range(2,13):
            self.full_dataset += MyDataset('train_data/train',str(i))

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    full_dataset = MyDataset('train_data/train','1')
    for i in range(2,13):
        full_dataset += MyDataset('train_data/train',str(i))
    # 划分训练集测试集
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = Lenet5().to(device)
    # 随机梯度下降
    sgd = SGD(model.parameters(), lr=1e-1)
    # 交叉熵
    loss_fn = CrossEntropyLoss()
    # 训练100代
    generation = 100
    for i in range(generation):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            sgd.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.float()).detach()
            predict_y =torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        right_rate = all_correct_num / all_sample_num
        print('准确率: {}'.format(right_rate), flush=True)
        if not os.path.isdir("models"):
            os.mkdir("models")
    torch.save(model, 'models/1017_{}.torch'.format(right_rate))

if __name__ == '__main__':
    train()

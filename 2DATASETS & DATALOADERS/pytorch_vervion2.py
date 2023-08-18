"""
作者：黄欣
日期：2023年08月18日
"""
import csv
import os.path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class myDataset(Dataset):
    def __init__(self, *, root, train_data=True, transform=None, target_transform=None):
        self.root = root
        self.path = os.path.join(self.root, "train" if train_data is True else "test")
        # transform和target_transform是对数据和标签的预处理方法
        self.transform = transform
        self.target_transform = target_transform
        # 参过函数load_csv获得数据的路径和标签
        self.X, self.y = self.load_csv('save_path.csv')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x, label = self.X[index], self.y[index]

        x = Image.open(x)
        # 必须是tensor类型数据才能使用batch
        # x要先转成np才能转成tensor
        x = np.array(x)
        x = torch.tensor(x)
        label = torch.tensor(label)
        # 并在这里对数据预处理
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return x, label

    def load_csv(self, savepath):
        if not os.path.exists(os.path.join( self.path, savepath)):
            data_X = []
            data_y = []
            # 如果该路径下的文件不存在就创建一个
            # 因为每类数据都存在各自的文件夹下，所以要对每个数据生成唯一的路径方便索引
            # 在数据集中，分为多个类别，每个类别的数据存在以类别命名的文件夹中,把路径集中起来
            labels = os.listdir(self.path)
            # 遍历每个类
            for label in labels:
                files = os.listdir(os.path.join(self.path, label))
                # 遍历类中的每个文件
                for file in files:
                    data_X.append(os.path.join(self.path, label, file))
                    data_y.append(label)

            # 打乱顺序
            key = np.array(range(len(data_X)))
            np.random.shuffle(key)
            data_X = np.array(data_X)
            data_y = np.array(data_y)
            # 变成np型式才不会报错
            data_X = data_X[key]
            data_y = data_y[key]

            # 保存成文件，方便直接从路径索引
            with open(os.path.join(self.path, savepath), mode='w', newline='') as f:
                # mode='w'参数指定以写入模式打开文件，如果文件已经存在，则覆盖原有内容。
                # newline=''参数指定在写入文件时不添加额外的换行符。
                writer = csv.writer(f)
                for X, y in zip(data_X,data_y):
                    writer.writerow([X, y]) # fashion_mnist_images\train\7\3151.png,7

        X = []
        y = []
        # 从保存文件中读数据
        with open(os.path.join(self.path, savepath)) as f:
            reader = csv.reader(f)
            for row in reader:
                x, label = row
                label = int(label)
                X.append(x)
                y.append(label)

        # 返回数据的路径和标签
        return X, y


trainData = myDataset(root='fashion_mnist_images', train_data=False)
train_dataloader = DataLoader(trainData, batch_size=64, shuffle=True)
x, label = next(iter(train_dataloader))
print(x.shape)
print(label)
plt.imshow(x[0], cmap="gray")
plt.show()












# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )
#
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )
#
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()




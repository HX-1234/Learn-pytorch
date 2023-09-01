"""
作者：黄欣
日期：2023年08月31日
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


path = "dog.jpg"
img = Image.open(path)
# plt.imshow(img)
# plt.show()



class Noise_transform():
    def __init__(self, snr, p):
        # 信噪比
        self.snr = snr
        # 中入噪声的概率
        self.p = p

    def __call__(self, img):
        if np.random.uniform(0,1) < self.p:
            noise_p = self.p
            signal_p = 1 - self.p
            img = np.array(img).copy()
            H, W, C = img.shape
            # 按signal_p, noise_p/2, noise_p/2的概率生成0,1,2矩阵
            mask = np.random.choice((0,1,2), size=(H,W,1), p=(signal_p, noise_p/2, noise_p/2))
            mask = np.repeat(mask, C, axis=2)
            img[mask==1] = 0
            img[mask==2] = 255
            return Image.fromarray(img.astype('uint8')).convert('RGB')

        else:
            return img

def transforms_inverse(img, trans):
    # 反向transforms
    if "Normalize" in str(trans): # 如果使用了Normalize
        # trans.transforms返回Compose中的内容
        # trans返回Compose对象
        # filter()它接受一个函数和一个序列作为参数，然后返回一个迭代器，其中包含序列中所有使函数返回值为True的元素。
        # list()函数将返回的迭代器转换为列表
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), trans.transforms))
        # 获得归一化时使用的均值和标准差
        mean = torch.tensor(norm_transform[0].mean)
        std = torch.tensor(norm_transform[0].std)
        # mul_和add_是原位操作
        # std[:, None, None]是一种用于增加张量维度的技巧
        # 形状为(3,)的一维张量转换成形状为(3, 1, 1)的三维张量，其中的3是通道数
        img.mul_(std[:, None, None]).add_(mean[:, None, None])

    img = img.transpose(0,2).transpose(0,1) # CHW => HWC
    # tensor => np
    img = np.array(img) * 255

    # 使用Image.fromarray()函数将数组转换为PIL图像
    if img.shape[2] == 3:# channel==3
        img = Image.fromarray( img.astype('uint8') ).convert('RGB')
    elif img.shape[2] == 1:  # channel==1
        # 当channel==1，必须用squeeze()函数移除多余的维度
        img = Image.fromarray(img.astype('uint8').squeeze())

    return img

trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)

     Noise_transform(0.5, 0.3),

     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)


img_tensor = trans(img)
img_PIL = transforms_inverse(img_tensor,trans)
plt.imshow(img_PIL)
plt.show()

# img_tensor = trans(img)
#
# Ncrops, C, H, M = img_tensor.shape
# for i in range(Ncrops):
#     # transforms.FiveCrop(112),  # 将获得5张图,遍历每张图
#     img_PIL = transforms_inverse(img_tensor[i],trans)
#
#     plt.imshow(img_PIL)
#     plt.show()
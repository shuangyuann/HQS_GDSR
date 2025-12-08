from torchvision import transforms
import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 接受一个图像和模数作为输入
def modcrop(image, modulo):
    h, w = image.shape[0], image.shape[1]  # 获取输入图像的高度和宽度
    h = h - h % modulo   # 确保图像的高度和宽度是模数的整数倍
    w = w - w % modulo

    return image[:h,:w]  # 返回裁剪后的图像

class Middlebury_dataset(Dataset):
    """Middlebury Dataset."""

    def __init__(self, root_dir, scale=8, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.transform = transform
        self.scale = scale

        # 初始化两个空列表，用于存储高分辨率深度图（GTs）和RGB图像（RGBs）的路径
        self.GTs = []
        self.RGBs = []

        # 列出根目录下的所有文件和文件夹
        list_dir = os.listdir(root_dir)
        '''
        for name in list_dir:
            if name.find('output_color') > -1: # output_color-->RGB图像  将路径加至RGBs列表中
                self.RGBs.append('%s/%s' % (root_dir, name))
            elif name.find('output_depth') > -1:  # output_depth-->高分辨率深度图  将路径加至GTs列表中
                self.GTs.append('%s/%s' % (root_dir, name))
        '''
        for name in list_dir:
            if name.find('output_color') > -1: # output_color-->RGB图像  将路径加至RGBs列表中
                self.RGBs.append('%s/%s' % (root_dir, name))
            elif name.find('output_depth') > -1:  # output_depth-->高分辨率深度图  将路径加至GTs列表中
                self.GTs.append('%s/%s' % (root_dir, name))
        self.RGBs.sort()  # 排序，确保匹配
        self.GTs.sort()

    def __len__(self):   # 获取数据集中的样本数量
        return len(self.GTs)

    def __getitem__(self, idx):  # 输入idx，返回对应样本
        
        image = np.array(Image.open(self.RGBs[idx]))
        gt = np.array(Image.open(self.GTs[idx]))
        assert gt.shape[0] == image.shape[0] and gt.shape[1] == image.shape[1]
        s = self.scale  
        image = modcrop(image, s)
        gt = modcrop(gt, s)

        h, w = gt.shape[0], gt.shape[1]
        s = self.scale

        lr = np.array(Image.fromarray(gt).resize((w//s,h//s),Image.BICUBIC)).astype(np.float32)
        gt = gt / 255.0
        image = image / 255.0
        lr = lr / 255.0
        

        if self.transform:
            image = self.transform(image).float()
            gt = self.transform(np.expand_dims(gt,2))
            lr = self.transform(np.expand_dims(lr,2)).float()

        # sample = {'guidance': image, 'lr': lr, 'gt': gt, 'max':maxx, 'min': minn}
        sample = {'guidance': image, 'lr': lr, 'gt': gt}
        return sample
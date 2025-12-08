import argparse
import os

from utils import *
import numpy as np
import torchvision.transforms as transforms
from torchvision import utils
from torch import Tensor
from PIL import Image
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.SGNet import *
from data.nyu_dataloader import *
from data.rgbdd_dataloader import *
from data.middlebury_dataloader import Middlebury_dataset

import cv2
import os
# 设置环境变量，避免潜在冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=8, help='scale factor')
parser.add_argument("--num_feats", type=int, default=40, help="channel number of the middle hidden layer")
# parser.add_argument("--root_dir", type=str, default='./data/nyu_data', help="root dir of dataset")
parser.add_argument("--root_dir", type=str, default='./data/Middlebury', help="root dir of dataset")
# parser.add_argument("--model_dir", type=str, default="./SGNet_test/pth/model/SGNet_X16.pth", help="path of model")
parser.add_argument("--model_dir", type=str, default="experiment/0914/NYUmodelRmse2.885845_8199.pth", help="path of model")
parser.add_argument("--results_dir", type=str, default='./SGNet_test/results', help="root dir of results")

opt = parser.parse_args()

net = SGNet(num_feats=opt.num_feats, kernel_size=3, scale=opt.scale)
net.load_state_dict(torch.load(opt.model_dir, map_location='cuda:0'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

data_transform = transforms.Compose([transforms.ToTensor()])

dataset_name = opt.root_dir.split('/')[-1]  # nyu_data
print(dataset_name)

if dataset_name == 'nyu_data':
    dataset = NYU_v2_dataset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=False)
    test_minmax = np.load('%s/test_minmax.npy' % opt.root_dir)
    rmse = np.zeros(449)
elif dataset_name == 'RGB-D-D':
    dataset = RGBDD_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='sync', train=False, transform=data_transform)
    rmse = np.zeros(405)
elif dataset_name == 'Middlebury':
    dataset = Middlebury_dataset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform)
    rmse = np.zeros(30)
elif dataset_name == 'Lu':
    dataset = Middlebury_dataset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform)
    rmse = np.zeros(6)

# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
data_num = len(dataloader)

def main():
    with torch.no_grad():
        net.eval()
        if dataset_name == 'nyu_data':
            for idx, data in enumerate(dataloader):
                guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)
                out, out_grad = net((guidance, lr))
                minmax = test_minmax[:, idx]
                minmax = torch.from_numpy(minmax).cuda()
                # 计算rmse
                rmse[idx] = calc_rmse(gt[0, 0], out[0, 0], minmax)

                # 输出结果目录 SGNet_test/results/output
                path_output = '{}/output'.format(opt.results_dir)
                
                os.makedirs(path_output, exist_ok=True)
                # 保存预测结果图路径
                path_save_pred = '{}/{:010d}.png'.format(path_output, idx)
                '''
                # 测试自己训练的模型
                # 输出结果目录 SGNet_test/results/output
                path_output_me = '{}/output_me'.format(opt.results_dir)

                os.makedirs(path_output_me, exist_ok=True)
                # 保存预测结果图路径
                path_save_pred = '{}/{:010d}.png'.format(path_output_me, idx)
                '''

                # Save results  (Save the output depth map)
                '''
                pred = out[0, 0] * (minmax[0] - minmax[1]) + minmax[1]
                pred = pred * 1000.0
                pred = pred.cpu().detach().numpy()
                pred = pred.astype(np.uint16)
                pred = Image.fromarray(pred)
                pred.save(path_save_pred)
                '''

                '''
                # visualization  (Visual depth map)
                # 从输出out中提取第一个通道的第一个元素
                pred = out[0, 0]
                # 传至cpu 从当前计算图中分理出pred 这样对其的修改不会影响梯度计算  转到numpy
                pred = pred.cpu().detach().numpy()
                # 将预测的深度图保存为图像文件  深度值缩放至[0,255]范围
                cv2.imwrite(path_save_pred, pred * 255.0)
                '''
                # print(rmse[idx])
                print("rmse_{}: {}".format(idx, rmse[idx]))
            print("rmse_mean: {}".format(rmse.mean()))
        elif dataset_name == 'RGB-D-D':
            for idx, data in enumerate(dataloader):
                guidance, lr, gt, maxx, minn = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device), data[
                    'max'].to(device), data['min'].to(device)
                out, out_grad = net((guidance, lr))
                minmax = [maxx, minn]
                rmse[idx] = rgbdd_calc_rmse(gt[0, 0], out[0, 0], minmax)

                path_output = '{}/output'.format(opt.results_dir)
                os.makedirs(path_output, exist_ok=True)
                path_save_pred = '{}/{:010d}.png'.format(path_output, idx)

                # Save results  (Save the output depth map)
                pred = out[0, 0] * (maxx - minn) + minn
                pred = pred.cpu().detach().numpy()
                pred = pred.astype(np.uint16)
                pred = Image.fromarray(pred)
                pred.save(path_save_pred)

                # visualization  (Visual depth map)
                #pred = out[0, 0]
                #pred = pred.cpu().detach().numpy()
                #cv2.imwrite(path_save_pred, pred * 255.0)
                print(rmse[idx])
            print(rmse.mean())
        elif (dataset_name == 'Middlebury') or (dataset_name == 'Lu'):
            for idx, data in enumerate(dataloader):
                guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)
                out, out_grad = net((guidance, lr))
                rmse[idx] = midd_calc_rmse(gt[0, 0], out[0, 0])

                path_output = '{}/output'.format(opt.results_dir)
                os.makedirs(path_output, exist_ok=True)
                path_save_pred = '{}/{:010d}.png'.format(path_output, idx)

                # Save results  (Save the output depth map)
                pred = out[0,0] * 255.0
                pred = pred.cpu().detach().numpy()
                pred = pred.astype(np.uint16)
                pred = Image.fromarray(pred)
                pred.save(path_save_pred)

                # visualization  (Visual depth map)
                #pred = out[0, 0]
                #pred = pred.cpu().detach().numpy()
                #cv2.imwrite(path_save_pred, pred * 255.0)

                print(rmse[idx])
            print(rmse.mean())

if __name__=="__main__":
    main()
import os
# 设置环境变量，避免潜在冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import logging
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=8, help='scale factor')
parser.add_argument('--lr', default='0.0001', type=float, help='learning rate')
parser.add_argument('--result', default='experiment', help='learning rate')
parser.add_argument('--epoch', default=200, type=int, help='max epoch')
parser.add_argument('--device', default="1", type=str, help='which gpu use')
parser.add_argument("--decay_iterations", type=list, default=[5e4, 1e5, 1.6e5], help="steps to start lr decay")
parser.add_argument("--num_feats", type=int, default=40, help="channel number of the middle hidden layer")
parser.add_argument("--gamma", type=float, default=0.2, help="decay rate of learning rate")
parser.add_argument("--root_dir", type=str, default='./data/nyu_data', help="root dir of dataset")
parser.add_argument("--batchsize", type=int, default=1, help="batchsize of training dataloader")


opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

from numpy.core.fromnumeric import mean

import torch
import numpy as np
# [1] 导入新模块
from models.hqs_wrapper import HQS_SGNet

from models.SGNet import *
from models.common import *

from data.nyu_dataloader import *
from data.rgbdd_dataloader import *
from data.middlebury_dataloader import Middlebury_dataset
from utils import calc_rmse, rgbdd_calc_rmse, midd_calc_rmse

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm



import matplotlib.pyplot as plt
import cv2



s = datetime.now().strftime('%Y%m%d%H%M%S')
dataset_name = opt.root_dir.split('/')[-1]
result_root = '%s/%s-lr_%s-s_%s-%s-b_%s' % (opt.result, s, opt.lr, opt.scale, dataset_name, opt.batchsize)
if not os.path.exists(result_root):
    os.mkdir(result_root)

logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)
logging.info(opt)

# from models.PhyGNet import PhyGNet
# net = PhyGNet(num_feats=opt.num_feats, scale=opt.scale, iterations=3).cuda()
# net = SGNet(num_feats=opt.num_feats, kernel_size=3, scale=opt.scale).cuda()
# [2] 初始化网络改为 HQS_SGNet
net = HQS_SGNet(num_feats=opt.num_feats, kernel_size=3, scale=opt.scale, iterations=3).cuda()
net_getFre = get_Fre()
net_grad = Get_gradient_nopadding_d()

# 损失 优化 学习率调度
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.decay_iterations, gamma=opt.gamma)

net.train()

data_transform = transforms.Compose([transforms.ToTensor()])
up = nn.Upsample(scale_factor=opt.scale, mode='bicubic')    # 上采样

# 数据集加载
if dataset_name == 'nyu_data':
    test_minmax = np.load('%s/test_minmax.npy' % opt.root_dir)
    train_dataset = NYU_v2_dataset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=True)
    test_dataset = NYU_v2_dataset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=False)
if dataset_name == 'RGB-D-D':
    train_dataset = NYU_v2_dataset(root_dir='/data/SRData/NYU_v2', scale=opt.scale, transform=data_transform, train=True)
    test_dataset = RGBDD_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='bicubic', transform=data_transform,
                                 train=False)

# 原版本 num_workers=8
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=0)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# opencv 热力学图保存辅助函数
# --- 新增的可视化辅助函数 ---
def to_heatmap(tensor):
    """
    将单通道的PyTorch Tensor转换为OpenCV格式的彩色热力图 (BGR)。

    Args:
        tensor (torch.Tensor): 输入的单通道张量，形状为 (H, W)。

    Returns:
        np.ndarray: 输出的BGR格式热力图，形状为 (H, W, 3)，数据类型为 uint8。
    """
    # 1. 将Tensor从GPU移动到CPU，并转换为Numpy数组
    #    .squeeze()用于移除可能存在的通道维度，确保是2D数组
    gray_img = tensor.detach().cpu().numpy().squeeze()

    # 2. 归一化到 0-255 范围
    #    确保图像范围在0到1之间，然后乘以255
    if gray_img.max() > 1.0 or gray_img.min() < 0.0:
        # 如果数据不在0-1范围，先进行归一化
        gray_img = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min())

    gray_img_uint8 = np.uint8(gray_img * 255)

    # 3. 应用伪彩色图 (COLORMAP_JET 是一种常用的热力图)
    heatmap = cv2.applyColorMap(gray_img_uint8, cv2.COLORMAP_JET)

    return heatmap


def main():
    max_epoch = opt.epoch
    num_train = len(train_dataloader)
    best_rmse = 10.0
    best_epoch = 0

    # 每轮训练
    for epoch in range(max_epoch):
        # ---------
        # Training
        # ---------
        net.train()
        running_loss = 0.0

        t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))
        # t更新进度条  enumerate(t)会为每个批次生成一个索引idx和数据data
        for idx, data in enumerate(t):
            # 计算到目前为止完成的批次总数 num_train是训练集中的样本数
            batches_done = num_train * epoch + idx
            optimizer.zero_grad()
            # 从数据批次中依次提取guidance(引导图像)、lr(低分辨率图像)、gt(高分辨率目标图像) ->GPU
            guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()

            # 模型向前传播 得到输出和梯度
            out, out_grad = net((guidance, lr))

            # 补充
            if out_grad is None:
                out_grad = net_grad(out)
            # 特征提取  从输出和目标图像中提取幅度和相位特征
            out_amp, out_pha = net_getFre(out)
            gt_amp, gt_pha = net_getFre(gt)

            # 梯度损失计算
            # 计算输出梯度和目标梯度之间的损失
            gt_grad = net_grad(gt)
            loss_grad1 = criterion(out_grad, gt_grad)

            # 特征损失计算
            # 计算输出特征的幅度和相位与目标特征之间的损失，并将其合并
            loss_fre_amp = criterion(out_amp, gt_amp)
            loss_fre_pha = criterion(out_pha, gt_pha)
            loss_fre = 0.5 * loss_fre_amp + 0.5 * loss_fre_pha

            # 空间损失计算
            # 计算输出和目标图像之间的空间损失
            loss_spa = criterion(out, gt)

            # Laplacian Loss
            # 可变形卷积引入边缘感知损失
            '''
            laplacian_kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]],
                                            device='cuda').view(1, 1, 3, 3)

            out_edge = F.conv2d(out, laplacian_kernel, padding=1)
            gt_edge = F.conv2d(gt, laplacian_kernel, padding=1)

            loss_edge = criterion(out_edge, gt_edge)  # criterion is nn.L1Loss()
            '''
            # 将其加入总损失
            # loss = loss_spa + 0.002 * loss_fre + 0.001 * loss_grad1 + 0.05 * loss_edge  # 0.05是超参数
            # 总损失计算 将上面的梯度损失、特征损失和空间损失合并
            loss = loss_spa + 0.002 * loss_fre + 0.001 * loss_grad1   # 一个batch

            loss.backward()   # 梯度累积
            optimizer.step()
            scheduler.step()  # 学习率调整
            # 损失记录  记录当前批次的损失，并更新50个批次的平均损失
            running_loss += loss.data.item()
            running_loss_50 = running_loss

            # 每50个批次更新一次进度条
            if idx % 50 == 0:
                running_loss_50 /= 50
                t.set_description('[train epoch:%d] loss: %.8f' % (epoch + 1, running_loss_50))
                t.refresh()

        logging.info('epoch:%d iteration:%d running_loss:%.10f' % (epoch + 1, batches_done + 1, running_loss / num_train))


        if (epoch % 2 == 0) and (epoch < 30) or epoch >= 30:
            with torch.no_grad():
                net.eval()
                if dataset_name == 'nyu_data':
                    rmse = np.zeros(449)
                if dataset_name == 'RGB-D-D':
                    rmse = np.zeros(405)

                # 创建一个专用于存储当前 epoch 验证结果的文件夹
                # epoch_viz_dir = os.path.join(result_root, f"epoch_{epoch}")
                # os.makedirs(epoch_viz_dir, exist_ok=True)

                t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))

                for idx, data in enumerate(t):
                    if dataset_name == 'nyu_data':
                        guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()

                        out, out_grad = net((guidance, lr))
                        minmax = test_minmax[:, idx] # 提取第idx列的所有行
                        minmax = torch.from_numpy(minmax).cuda()  # 转化为pytorch张量
                        rmse[idx] = calc_rmse(gt[0, 0], out[0, 0], minmax)

                    if dataset_name == 'RGB-D-D':
                        guidance, lr, gt, max, min = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data[
                            'max'].cuda(), data['min'].cuda()
                        out = net((guidance, lr))
                        minmax = [max, min]
                        rmse[idx] = rgbdd_calc_rmse(gt[0, 0], out[0, 0], minmax)

                        t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())
                        t.refresh()
                    lr = up(lr)
                    # heatmap_lr = to_heatmap(lr[0])  # batchsize=1 所以是取[0]
                    # heatmap_out = to_heatmap(out[0])
                    # heatmap_gt = to_heatmap(gt[0])
                    # comparison_img = np.hstack((heatmap_lr, heatmap_out, heatmap_gt))
                    # cv2.imwrite('%s/heatmap_%d.png' % (epoch_viz_dir, idx), comparison_img)

                if epoch % 10 == 0:
                    # 误差图
                    error = torch.abs(out - gt)
                    error_np = error.squeeze().cpu().detach().numpy()
                    error_normalized = (error_np - np.min(error_np)) / (np.max(error_np) - np.min(error_np))
                    # 可视化误差图
                    plt.imshow(error_normalized, cmap='jet')
                    plt.colorbar()
                    plt.title('Error Map')
                    plt.imsave('%s/error%d.png' % (result_root, batches_done), error_normalized, cmap='jet')

                r_mean = rmse.mean()
                if r_mean < best_rmse:
                    best_rmse = r_mean
                    best_epoch = epoch

                    # 保存最佳模型
                    if best_rmse<2.6 :
                        torch.save(net.state_dict(),
                        os.path.join(result_root, "NYUmodelbest%f_8%d.pth" % (best_rmse, best_epoch + 1)))

                logging.info(
                    '---------------------------------------------------------------------------------------------------------------------------')
                logging.info('epoch:%d lr:%f-------mean_rmse:%f (BEST: %f @epoch%d)' % (
                    epoch + 1, scheduler.get_last_lr()[0], r_mean, best_rmse, best_epoch + 1))
                logging.info(
                    '---------------------------------------------------------------------------------------------------------------------------')

if __name__ == '__main__':
    main()
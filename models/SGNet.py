from models.common import *
import torch
import torch.nn as nn

# 改造原SGNet，增加一个可选参数depth_hr_prev
    # 初始状态：输入LR，网络内部进行Bicubic上采样
    # 迭代状态：输入LR+上一轮的HR。网络将使用上一轮的HR来进行梯度引导和残差相加，而不是粗糙的Bicubic结果
'''
class SGNet(nn.Module):
    def __init__(self, num_feats, kernel_size, scale):
        super(SGNet, self).__init__()
        self.conv_rgb1 = nn.Conv2d(in_channels=3, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1)
        self.rgb_rb2 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.rgb_rb3 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.rgb_rb4 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)

        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.conv_dp2 = nn.Conv2d(in_channels=num_feats, out_channels=2*num_feats,
                                  kernel_size=kernel_size, padding=1)
        # 瘦身改进一：n_resblocks从6改到3
        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=3)
        self.dp_rg2 = ResidualGroup(default_conv, 2*num_feats, kernel_size, reduction=16, n_resblocks=3)
        self.dp_rg3 = ResidualGroup(default_conv, 2*num_feats, kernel_size, reduction=16, n_resblocks=3)
        self.dp_rg4 = ResidualGroup(default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=3)

        self.bridge1 = SDM(channels=num_feats, rgb_channels=num_feats,scale=scale)
        self.bridge2 = SDM(channels=2*num_feats, rgb_channels=num_feats,scale=scale)
        self.bridge3 = SDM(channels=3*num_feats, rgb_channels=num_feats,scale=scale)

        self.c_de = default_conv(4*num_feats, 2*num_feats, 1)
        # 瘦身改进二：n_resblocks从8改到4
        my_tail = [
            ResidualGroup(
                default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=4),
            ResidualGroup(
                default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=4),
            ResidualGroup(
                default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=4)
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(3*num_feats, 3*num_feats, scale, up=True, bottleneck=False)
        last_conv = [
            default_conv(3*num_feats, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True)
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

        self.c_rd = default_conv(8*num_feats, 3*num_feats, 1)
        self.c_grad = default_conv(2*num_feats, num_feats, 1)
        self.c_grad2 = default_conv(3*num_feats, 2*num_feats, 1)
        self.c_grad3 = default_conv(3*num_feats, 3*num_feats, 1)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.gradNet = GCM(n_feats=num_feats,scale=scale)

    def forward(self, inputs):
        # 兼容旧代码调用方式，同时支持新参数  增加新参数depth_hr_prev（上一轮得到的高分深度图）
        if len(inputs) == 3:
            image, depth_lr, depth_hr_prev = inputs
        else:
            image, depth_lr = inputs
            depth_hr_prev = None

        # 1. 梯度校准模块 (GCM)
        # 如果有上一轮的 HR 结果，用它来算梯度会更准！(双向优化的体现)
        if depth_hr_prev is not None:
            out_re, grad_d4 = self.gradNet(depth_hr_prev, image)
        else:
            out_re, grad_d4 = self.gradNet(depth_lr, image)

        # 2. 深度图特征提取 (始终使用 LR 输入以节省显存和计算量)
        dp_in = self.act(self.conv_dp1(depth_lr))
        dp1 = self.dp_rg1(dp_in)

        # 不变
        cat10 = torch.cat([dp1, grad_d4], dim=1)
        dp1_ = self.c_grad(cat10)

        rgb1 = self.act(self.conv_rgb1(image))
        rgb2 = self.rgb_rb2(rgb1)


        ca1_in, r1 = self.bridge1(dp1_, rgb2)
        dp2 = self.dp_rg2(torch.cat([dp1, ca1_in + dp_in], 1))

        cat11 = torch.cat([dp2, grad_d4], dim=1)
        dp2_ = self.c_grad2(cat11)

        rgb3 = self.rgb_rb3(r1)
        ca2_in, r2 = self.bridge2(dp2_, rgb3)

        ca2_in_ = ca2_in + self.conv_dp2(dp_in)

        cat1_0 = torch.cat([dp2, ca2_in_], 1)

        dp3 = self.dp_rg3(self.c_de(cat1_0))
        rgb4 = self.rgb_rb4(r2)

        cat12 = torch.cat([dp3, grad_d4], dim=1)
        dp3_ = self.c_grad3(cat12)

        ca3_in, r3 = self.bridge3(dp3_, rgb4)
        cat1 = torch.cat([dp1, dp2, dp3, ca3_in], 1)

        dp4 = self.dp_rg4(self.c_rd(cat1))

        tail_in = self.upsampler(dp4)
        out = self.last_conv(self.tail(tail_in))

        # out = out + self.bicubic(depth)
        # [修改点]：残差连接
        # 如果是迭代中，直接加在 refined depth 上
        # 如果是第一次，加在 bicubic 上
        if depth_hr_prev is not None:
            out = out + depth_hr_prev
        else:
            out = out + self.bicubic(depth_lr)
        return out, out_re
'''

from models.common import *
import torch
import torch.nn as nn


class SGNet(nn.Module):
    def __init__(self, num_feats, kernel_size, scale):
        super(SGNet, self).__init__()

        # --- RGB 分支 (大幅精简) ---
        self.conv_rgb1 = nn.Conv2d(in_channels=3, out_channels=num_feats, kernel_size=kernel_size, padding=1)
        # 只保留第一个 ResBlock，用于给 Bridge1 提供特征
        self.rgb_rb2 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        # [删除] self.rgb_rb3, self.rgb_rb4 (深层 RGB 特征不再需要)

        # --- Depth 分支 ---
        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats, kernel_size=kernel_size, padding=1)

        # [删除] self.conv_dp2 (用于 Bridge2 的旁路不再需要)

        # Residual Groups (建议保持 n_resblocks=3 以节省显存，或者尝试改回 4-6)
        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=3)
        self.dp_rg2 = ResidualGroup(default_conv, 2 * num_feats, kernel_size, reduction=16, n_resblocks=3)
        self.dp_rg3 = ResidualGroup(default_conv, 2 * num_feats, kernel_size, reduction=16, n_resblocks=3)
        self.dp_rg4 = ResidualGroup(default_conv, 3 * num_feats, kernel_size, reduction=16, n_resblocks=3)

        # --- Bridges (只保留 Bridge 1) ---
        self.bridge1 = SDM(channels=num_feats, rgb_channels=num_feats, scale=scale)
        # [删除] self.bridge2, self.bridge3

        # [删除] self.c_de (用于 Bridge2 融合的层不再需要)

        # Tail 部分
        my_tail = [
            ResidualGroup(default_conv, 3 * num_feats, kernel_size, reduction=16, n_resblocks=4),
            ResidualGroup(default_conv, 3 * num_feats, kernel_size, reduction=16, n_resblocks=4),
            ResidualGroup(default_conv, 3 * num_feats, kernel_size, reduction=16, n_resblocks=4)
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(3 * num_feats, 3 * num_feats, scale, up=True, bottleneck=False)
        last_conv = [
            default_conv(3 * num_feats, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True)
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

        # [修改] 融合层 c_rd
        # 原输入: 8*num_feats (1+2+2+3)
        # 新输入: 5*num_feats (1+2+2) -> 因为 bridge3 的 3*num_feats 没了
        self.c_rd = default_conv(5 * num_feats, 3 * num_feats, 1)

        self.c_grad = default_conv(2 * num_feats, num_feats, 1)
        # [删除] self.c_grad2, self.c_grad3

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.gradNet = GCM(n_feats=num_feats, scale=scale)

    def forward(self, inputs):
        # 兼容性处理
        if len(inputs) == 3:
            image, depth_lr, depth_hr_prev = inputs
        else:
            image, depth_lr = inputs
            depth_hr_prev = None

        # 1. GCM 梯度引导 (这是双向的核心，必须保留)
        if depth_hr_prev is not None:
            out_re, grad_d4 = self.gradNet(depth_hr_prev, image)
        else:
            out_re, grad_d4 = self.gradNet(depth_lr, image)

        # 2. 浅层特征提取
        dp_in = self.act(self.conv_dp1(depth_lr))

        # --- Stage 1 ---
        dp1 = self.dp_rg1(dp_in)
        # 融合 GCM 梯度
        cat10 = torch.cat([dp1, grad_d4], dim=1)
        dp1_ = self.c_grad(cat10)

        # --- RGB & Bridge 1 (保留唯一的引导) ---
        rgb1 = self.act(self.conv_rgb1(image))
        rgb2 = self.rgb_rb2(rgb1)
        ca1_in, r1 = self.bridge1(dp1_, rgb2)  # 只有这里发生交互

        # --- Stage 2 ---
        # dp2 接收 Bridge1 的引导
        dp2 = self.dp_rg2(torch.cat([dp1, ca1_in + dp_in], 1))

        # --- Stage 3 (无 Bridge 2) ---
        # 直接把 dp2 传给 dp3 (特征直通)
        dp3 = self.dp_rg3(dp2)

        # --- Stage 4 (无 Bridge 3) ---
        # 融合前三层特征
        # dp1(1) + dp2(2) + dp3(2) = 5 * num_feats
        cat1 = torch.cat([dp1, dp2, dp3], 1)

        dp4 = self.dp_rg4(self.c_rd(cat1))

        # --- Tail & Upsample ---
        tail_in = self.upsampler(dp4)
        out = self.last_conv(self.tail(tail_in))

        # 残差连接
        if depth_hr_prev is not None:
            out = out + depth_hr_prev
        else:
            out = out + self.bicubic(depth_lr)

        return out, out_re

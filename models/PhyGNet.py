import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -------------------------------------------------------------------------
# 基础组件 (可以直接复用 models.common 中的，为了独立性这里写出)
# -------------------------------------------------------------------------
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


# -------------------------------------------------------------------------
# 创新模块 1: 几何一致性门控 (Geometry-Consistency Gating, GCG)
# 作用: 解决纹理拷贝问题，只允许一致的边缘通过
# -------------------------------------------------------------------------
class GeometryGating(nn.Module):
    def __init__(self, channels):
        super(GeometryGating, self).__init__()
        # 梯度提取器: 使用分组卷积计算特征梯度
        self.grad_extract = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.InstanceNorm2d(channels),
            nn.Sigmoid()
        )
        # 融合层
        self.fusion = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, feat_depth, feat_rgb):
        # 1. 提取特征梯度 (边缘强度)
        grad_d = self.grad_extract(feat_depth)
        grad_r = self.grad_extract(feat_rgb)

        # 2. 计算一致性图 (Consistency Map)
        # 只有当 Depth 和 RGB 都有强梯度时，值才接近 1
        consistency = grad_d * grad_r

        # 3. 门控机制 (Gating)
        # 利用一致性图对 RGB 特征进行加权：一致性低的地方 RGB 特征会被抑制
        gated_rgb = feat_rgb * (1 + consistency)

        # 4. 融合
        out = self.fusion(torch.cat([feat_depth, gated_rgb], dim=1))
        return self.act(out + feat_depth)  # 残差连接


# -------------------------------------------------------------------------
# 创新模块 2: 轻量级谱变换 Transformer (Spectral Transformer Block, STB)
# 作用: 解决感受野受限问题，在频域实现全局交互
# -------------------------------------------------------------------------
class SpectralTransformer(nn.Module):
    def __init__(self, channels):
        super(SpectralTransformer, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 2, 1)
        # 频域处理层: 处理实部和虚部
        self.conv_freq = nn.Conv2d(channels // 2, channels // 2, 1)
        self.conv2 = nn.Conv2d(channels // 2, channels, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # [B, C, H, W]
        identity = x

        # 1. 降维
        x_short = self.conv1(x)

        # 2. 快速傅里叶变换 (Real FFT)
        # 输出形状 [B, C/2, H, W/2+1] (复数)
        x_fft = torch.fft.rfft2(x_short, norm='backward')

        # 3. 频域全局交互
        # 将实部和虚部分开处理 (模拟复数卷积)
        real = x_fft.real
        imag = x_fft.imag

        real = self.conv_freq(real)
        imag = self.conv_freq(imag)

        # 重新组合复数
        x_fft_processed = torch.complex(real, imag)

        # 4. 傅里叶逆变换
        # 输出形状 [B, C/2, H, W]
        x_back = torch.fft.irfft2(x_fft_processed, s=x_short.shape[-2:], norm='backward')

        # 5. 升维与残差融合
        out = self.conv2(self.relu(x_short + x_back))

        return identity + out


# -------------------------------------------------------------------------
# 创新模块 3: 物理约束层 (Data Consistency Layer)
# 作用: 强制解符合物理观测模型 (D_down == D_LR)
# -------------------------------------------------------------------------
class DataConsistencyLayer(nn.Module):
    def __init__(self, scale):
        super(DataConsistencyLayer, self).__init__()
        self.scale = scale
        # 可学习的步长参数，控制物理约束的力度
        self.eta = nn.Parameter(torch.tensor(0.5))

    def forward(self, d_hr, d_lr):
        # 1. 模拟退化: HR -> LR
        d_down = F.interpolate(d_hr, scale_factor=1 / self.scale, mode='bicubic', align_corners=False)

        # 2. 计算物理残差
        residual = d_down - d_lr

        # 3. 误差反向投影: LR -> HR
        residual_up = F.interpolate(residual, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # 4. 修正估计值
        return d_hr - self.eta * residual_up


# -------------------------------------------------------------------------
# 主模型: PhyG-Net (物理引导的几何感知网络)
# 架构: 深度展开迭代架构 (Deep Unrolling Architecture)
# -------------------------------------------------------------------------
class PhyGNet(nn.Module):
    def __init__(self, num_feats=40, scale=8, iterations=3):
        super(PhyGNet, self).__init__()
        self.iterations = iterations
        self.scale = scale
        self.num_feats = num_feats

        # --- 1. 浅层特征提取器 (Shared Encoder) ---
        # RGB 分支
        self.head_rgb = nn.Sequential(
            nn.Conv2d(3, num_feats, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(default_conv, num_feats, 3)
        )
        # Depth 分支
        self.head_depth = nn.Sequential(
            nn.Conv2d(1, num_feats, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # --- 2. 核心处理单元 (Shared Body) ---
        # 包含: STB (全局特征) + GCG (纹理过滤)
        # 这里设计一个轻量级的 Body，反复使用
        self.body_stb1 = SpectralTransformer(num_feats)
        self.body_stb2 = SpectralTransformer(num_feats)

        self.gating = GeometryGating(num_feats)

        self.body_stb3 = SpectralTransformer(num_feats)
        self.body_conv = default_conv(num_feats, num_feats, 3)  # 普通卷积细化

        # --- 3. 重建头 (Reconstruction) ---
        self.tail = nn.Conv2d(num_feats, 1, 3, 1, 1)

        # --- 4. 物理约束层 ---
        self.dc_layer = DataConsistencyLayer(scale)

    def forward(self, inputs):
        # 适配不同的输入格式
        if len(inputs) == 2:
            image, depth_lr = inputs
        elif len(inputs) == 3:  # 兼容某些训练代码
            image, depth_lr, _ = inputs

        # -----------------------------------------------
        # Step 0: 初始化 (Initialization)
        # -----------------------------------------------
        # 使用 Bicubic 插值作为初始猜测
        d_est = F.interpolate(depth_lr, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # 预先提取 RGB 特征 (迭代中复用，节省计算)
        f_rgb = self.head_rgb(image)

        outputs = []

        # -----------------------------------------------
        # Step 1: 深度展开迭代 (Unrolled Iterations)
        # -----------------------------------------------
        for i in range(self.iterations):
            # === A. 物理约束步 (D-step) ===
            # 强制当前估计符合 LR 观测
            d_est = self.dc_layer(d_est, depth_lr)

            # === B. 先验优化步 (Z-step / Network step) ===

            # 1. 提取当前深度图特征
            f_d = self.head_depth(d_est)

            # 2. 全局特征建模 (STB)
            x = self.body_stb1(f_d)
            x = self.body_stb2(x)

            # 3. 几何门控融合 (GCG) - 核心双向交互
            # 这里融合 RGB 特征，但会自动过滤纹理
            x = self.gating(x, f_rgb)

            # 4. 后处理与重建
            x = self.body_stb3(x)
            x = self.body_conv(x)

            # 残差重建: d_new = d_old + residual
            res = self.tail(x)
            d_est = d_est + res

            outputs.append(d_est)

        # 返回最后一次迭代的结果 (也可返回 list 用于深监督)
        # 这里返回 tuple (d_est, None) 是为了兼容原 SGNet 的接口 (d, grad)
        return outputs[-1], None
# models/hqs_wrapper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SGNet import SGNet


class DataConsistency(nn.Module):
    """
    HQS 数据一致性层 (Data Consistency Layer)
    公式: D_k+1 = D_k - eta * BackProject(DownSample(D_k) - D_LR)
    """

    def __init__(self, scale):
        super(DataConsistency, self).__init__()
        self.scale = scale
        # 这是一个可学习的参数，控制数据项约束的力度
        self.eta = nn.Parameter(torch.tensor(0.2))  # 初始步长

    def forward(self, d_current_hr, d_input_lr):
        # 1. 模拟退化过程：HR -> LR (双三次下采样)
        d_down = F.interpolate(d_current_hr, scale_factor=1 / self.scale, mode='bicubic', align_corners=False)

        # 2. 计算残差
        residual = d_down - d_input_lr

        # 3. 误差反向投影：LR -> HR
        residual_up = F.interpolate(residual, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # 4. 更新深度图
        d_refined = d_current_hr - self.eta * residual_up

        return d_refined


class HQS_SGNet(nn.Module):
    def __init__(self, num_feats, kernel_size, scale, iterations=3):
        super(HQS_SGNet, self).__init__()
        self.iterations = iterations

        # 核心：只实例化一个 SGNet，所有迭代共享权重！
        self.body = SGNet(num_feats, kernel_size, scale)

        # HQS 物理约束层
        self.dc_layer = DataConsistency(scale)

    def forward(self, inputs):
        image, depth_lr = inputs

        # 1. 初始化 (Iteration 0)
        # 第一次前向传播，没有 depth_hr_prev
        d_est, out_grad = self.body((image, depth_lr))

        outputs = [d_est]  # 记录每次迭代的结果用于深监督

        # 2. 开始迭代 (Iteration 1 to K)
        for i in range(self.iterations - 1):
            # --- D-step: 数据一致性校正 ---
            d_dc = self.dc_layer(d_est, depth_lr)

            # --- Z-step: SGNet 先验去噪 ---
            # 这里的关键是把 d_dc 作为 depth_hr_prev 传进去
            # SGNet 内部会利用这个更准的 depth 计算更准的 GCM 梯度
            d_est, out_grad = self.body((image, depth_lr, d_dc))

            outputs.append(d_est)

        # 返回最后一次的结果，以及最后一次的梯度图(用于原始Loss)
        return outputs[-1], out_grad
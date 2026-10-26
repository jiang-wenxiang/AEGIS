import torch
import torch.nn as nn
from .decoder import FlexibleDecoder
import torch_dct as dct  # 必须安装: pip install torch-dct
from .HaarDWT import HaarDWT
import torch.nn.functional as F


class MultiDomainDecoder(nn.Module):
    """
    多域解码器：替代原版 Decoder 类。
    依赖外部定义的 HaarDWT。
    """
    def __init__(self, config):
        super(MultiDomainDecoder, self).__init__()

        # --- 1. Spatial Transformer Network (STN) 矫正模块 ---
        # StegaStamp 的核心亮点，用于抵抗透视变换
        self.stn_params = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        # 计算 Flatten 后的尺寸: H/8 * W/8 * 128
        self.flatten_dim = (config.H // 8) * (config.W // 8) * 128
        self.fc_loc = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)  # 输出 2x3 的仿射变换矩阵参数
        )

        # 初始化 STN 为单位变换 (Identity Transform)
        # 这是一个关键的 Trick，否则 STN 很难训练
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


        # 复用 Encoder 中已经定义好的 HaarDWT 类
        self.dwt = HaarDWT()

        # 初始化 4 个独立的子解码器
        self.spatial_decoder = FlexibleDecoder(config)

        self.dwt_decoder = FlexibleDecoder(config)

        self.dct_decoder = FlexibleDecoder(config)

        self.svd_decoder = FlexibleDecoder(config)

        # --- 新增: 融合层 (Fusion Layer) ---
        # 输入通道: 4 (Spatial, DWT, DCT, SVD)
        # 输出通道: 1 (最终融合结果)
        # Kernel: 1 (相当于对这4个值做全连接/加权求和)
        self.fusion_layer = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)

    def stn(self, x):
        xs = self.stn_params(x)
        xs = xs.view(-1, self.flatten_dim)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # 生成网格并采样
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, image_with_wm):
        # 1. 矫正图像
        x = self.stn(image_with_wm)

        # 1. Spatial Domain
        res_spatial = self.spatial_decoder(x)

        # --- 变换: DWT ---
        # 使用共用的 HaarDWT 模块
        LL1, _ = self.dwt(x)

        # 2. DWT Domain
        res_dwt = self.dwt_decoder(LL1)

        # --- 变换: DCT ---
        LL1_dct = dct.dct_2d(LL1)

        # 3. DCT Domain
        res_dct = self.dct_decoder(LL1_dct)

        # --- 变换: DWT Level 2 & SVD ---
        LL2, _ = self.dwt(LL1)

        # SVD (只取 S)
        # 注意：这里需要处理奇异值分解
        _, S, _ = torch.linalg.svd(LL2)

        # 将 S 向量映射回对角矩阵图像
        S_matrix = torch.zeros_like(LL2)
        for i in range(S.shape[-1]):
            S_matrix[:, :, i, i] = S[:, :, i]

        # 4. SVD Domain
        res_svd = self.svd_decoder(S_matrix)

        # --- 融合步骤 ---

        # 1. 堆叠 (Stack): [B, L] x 4 -> [B, 4, L]
        # 这里 4 被视为“通道(Channels)”，L 被视为“长度(Length)”
        stacked_results = torch.stack([res_spatial, res_dwt, res_dct, res_svd], dim=1)

        # 2. 融合 (Fusion): [B, 4, L] -> Conv1d -> [B, 1, L]
        # 这一步网络会学习: Final = w1*Spatial + w2*DWT + w3*DCT + w4*SVD + bias
        fused_result = self.fusion_layer(stacked_results)

        # 3. 降维: [B, 1, L] -> [B, L]
        fused_result = fused_result.squeeze(1)

        # 返回: 融合后的最终结果 (用于验证/主要Loss)，以及 4个中间结果 (用于辅助Loss)
        return [fused_result, (res_spatial, res_dwt, res_dct, res_svd)]

        # --- 输出 ---
        # return [res_spatial, res_dwt, res_dct, res_svd]

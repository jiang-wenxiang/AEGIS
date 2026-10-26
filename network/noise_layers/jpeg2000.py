import torch
import torch.nn as nn
import torch.nn.functional as F


class Jpeg2000Sim(nn.Module):
    """
    高级可微 JPEG2000 模拟层 (专为 256x256 或 2的幂次方分辨率设计)
    采用 3 级 Haar DWT 分解，并对不同频率子带实施阶梯化平滑量化。
    """

    def __init__(self, min_Q=10, levels=3):
        super(Jpeg2000Sim, self).__init__()
        self.min_Q = min_Q
        self.levels = levels  # 默认 3 级小波分解 (256 -> 128 -> 64 -> 32)
        self.intensity = 1.0

        # 定义 Haar 小波的 4 个基础卷积核 (LL, HL, LH, HH)
        haar_weights = torch.tensor([
            [[1, 1], [1, 1]],  # LL (低频)
            [[-1, -1], [1, 1]],  # HL (水平高频)
            [[-1, 1], [-1, 1]],  # LH (垂直高频)
            [[1, -1], [-1, 1]]  # HH (对角线高频)
        ], dtype=torch.float32) / 2.0

        self.register_buffer('haar_weights', haar_weights.unsqueeze(1))
        self.register_buffer('idwt_weights', haar_weights.unsqueeze(1))

    def set_intensity(self, intensity):
        self.intensity = intensity

    def round_ss(self, x):
        """可微的阶跃函数近似，保持梯度回传"""
        cond = (torch.abs(x) < 0.5).float().to(x.device)
        return cond * (x ** 3) + (1 - cond) * x

    def _dwt2(self, x):
        """单级 2D DWT"""
        B, C, H, W = x.shape
        x_reshaped = x.reshape(B * C, 1, H, W)
        dwt_coeffs = F.conv2d(x_reshaped, self.haar_weights, stride=2)
        LL = dwt_coeffs[:, 0:1, :, :].reshape(B, C, H // 2, W // 2)
        Highs = dwt_coeffs[:, 1:4, :, :].reshape(B, C * 3, H // 2, W // 2)
        return LL, Highs

    def _idwt2(self, LL, Highs):
        """单级 2D IDWT"""
        B, C, H_half, W_half = LL.shape
        LL_reshaped = LL.reshape(B * C, 1, H_half, W_half)
        Highs_reshaped = Highs.reshape(B * C, 3, H_half, W_half)
        recon_coeffs = torch.cat([LL_reshaped, Highs_reshaped], dim=1)
        reconstructed = F.conv_transpose2d(recon_coeffs, self.idwt_weights, stride=2)
        return reconstructed.reshape(B, C, H_half * 2, W_half * 2)

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        if self.intensity == 0:
            return image

        # 动态计算基准量化步长 (Base Step Size)
        cur_Q = 100 - (100 - self.min_Q) * self.intensity
        base_step = 2 - cur_Q * 0.02 if cur_Q >= 50 else 50 / cur_Q

        # 存储各级的高频特征
        highs_list = []
        current_LL = image

        # === 前向分解阶段 (DWT) ===
        for i in range(self.levels):
            current_LL, current_Highs = self._dwt2(current_LL)
            highs_list.append(current_Highs)

        # === 频率级联量化阶段 (模拟 JPEG2000 核心特性) ===
        # JPEG2000 特性：外层(极高频)量化最严重，内层(中高频)量化较轻
        quantized_highs = []
        for i in range(self.levels):
            # 权重随层级指数衰减：第一层(最外层高频) 乘 2.0，第二层 1.0，第三层 0.5
            # 这使得伪影表现与真实 OpenJPEG 的码率分配高度一致
            level_weight = 2.0 / (2 ** i)
            step_size = base_step * level_weight

            q_high = highs_list[i] / step_size
            q_high = self.round_ss(q_high)
            q_high = q_high * step_size
            quantized_highs.append(q_high)

        # === 后向重构阶段 (IDWT) ===
        reconstructed_LL = current_LL
        for i in reversed(range(self.levels)):
            reconstructed_LL = self._idwt2(reconstructed_LL, quantized_highs[i])

        return reconstructed_LL.clamp(-1.0, 1.0)
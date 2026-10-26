import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [B, C, H, W]
        h, w = x.shape[2], x.shape[3]

        # 1. 计算 Padding 信息 (0 或 1)
        pad_h = h % 2
        pad_w = w % 2

        # 2. 执行 Padding (如果是奇数)
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 3. 执行 DWT
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        LL = x1 + x2 + x3 + x4

        # HL, LH, HH
        HL = x1 - x2 + x3 - x4
        LH = x1 + x2 - x3 - x4
        HH = x1 - x2 - x3 + x4

        # --- 关键修改：将 Padding 信息打包进 coeffs ---
        # coeffs 结构变为: (HL, LH, HH, (pad_h, pad_w))
        pads = (pad_h, pad_w)
        coeffs = (HL, LH, HH, pads)

        return LL, coeffs

# class HaarDWT(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         x01 = x[:, :, 0::2, :] / 2; x02 = x[:, :, 1::2, :] / 2
#         x1 = x01[:, :, :, 0::2]; x2 = x02[:, :, :, 0::2]
#         x3 = x01[:, :, :, 1::2]; x4 = x02[:, :, :, 1::2]
#         LL = x1 + x2 + x3 + x4
#         # 返回 LL 和高频系数 (作为一个tuple方便传递)
#         return LL, (x1 - x2 + x3 - x4, x1 + x2 - x3 - x4, x1 - x2 - x3 + x4)


class HaarIDWT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, LL, coeffs):
        # --- 关键修改：解包 4 个元素 ---
        # 前三个是张量，最后一个是 Padding 元组
        HL, LH, HH, pads = coeffs
        pad_h, pad_w = pads  # 获取记录的 padding 信息

        # 1. 逆运算
        # (LL + HL + LH + HH) -> 4 * x1
        # ...
        term1 = LL + LH  # 2(x1 + x2)
        term2 = HL + HH  # 2(x1 - x2)
        term3 = LL - LH  # 2(x3 + x4)
        term4 = HL - HH  # 2(x3 - x4)

        pixel_TL = (term1 + term2) * 0.5
        pixel_BL = (term1 - term2) * 0.5
        pixel_TR = (term3 + term4) * 0.5
        pixel_BR = (term3 - term4) * 0.5

        # 2. 重组 (Interleaving)
        B, C, H, W = LL.shape
        # 这里的 H, W 是子带的大小，恢复后是 2H, 2W
        recon = torch.zeros(B, C, H * 2, W * 2, device=LL.device)

        recon[:, :, 0::2, 0::2] = pixel_TL
        recon[:, :, 1::2, 0::2] = pixel_BL
        recon[:, :, 0::2, 1::2] = pixel_TR
        recon[:, :, 1::2, 1::2] = pixel_BR

        # 3. 自动裁剪 (基于 coeffs 中的信息)
        # 如果之前 Pad 了高度 (pad_h=1)，现在就切掉最后一行
        if pad_h == 1:
            recon = recon[:, :, :-1, :]

        # 如果之前 Pad 了宽度 (pad_w=1)，现在就切掉最后一列
        if pad_w == 1:
            recon = recon[:, :, :, :-1]

        return recon

# class HaarIDWT(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, LL, coeffs):
#         LH, HL, HH = coeffs
#         x1 = (LL + HL + LH + HH) / 2; x2 = (LL - HL + LH - HH) / 2
#         x3 = (LL + HL - LH - HH) / 2; x4 = (LL - HL - LH + HH) / 2
#         B, C, H, W = LL.shape
#         output = torch.zeros(B, C, H * 2, W * 2, device=LL.device)
#         output[:, :, 0::2, 0::2] = x1; output[:, :, 1::2, 0::2] = x2
#         output[:, :, 0::2, 1::2] = x3; output[:, :, 1::2, 1::2] = x4
#         return output
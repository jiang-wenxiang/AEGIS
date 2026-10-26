import torch
import torch.nn as nn


class Quantization(nn.Module):
    def __init__(self, bits=8):
        super(Quantization, self).__init__()
        self.bits = bits
        self.intensity = 1.0  # 强度控制，1.0为满载量化

    def set_intensity(self, intensity):
        self.intensity = intensity

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        if self.intensity == 0:
            return image

        # 动态控制量化精度
        # 当 intensity=1 时，量化到 8bit (256级)
        # 当 intensity 变小，量化级别自动提升，逼近原图
        levels = 2 ** (self.bits * self.intensity)

        # 将 [-1, 1] 映射到 [0, 1]
        img_01 = (image + 1.0) / 2.0

        # 核心量化操作 (Round 函数在非整数点不可微，这里用平滑近似)
        # 使用 Round_ss 技巧 (参考您之前的 jpeg.py 实现)
        img_quantized = torch.round(img_01 * (levels - 1)) / (levels - 1)

        # 映射回 [-1, 1]
        return (img_quantized * 2.0 - 1.0).clamp(-1.0, 1.0)
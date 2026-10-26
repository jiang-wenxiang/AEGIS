import torchvision.transforms.functional as TF
import torch.nn as nn
import torch
import kornia as K


class Sharpening(nn.Module):
    """
    可微锐化噪声层 (基于 Unsharp Mask)
    """

    def __init__(self, max_alpha=2.0, kernel_size=5, sigma=1.5):
        super(Sharpening, self).__init__()
        # max_alpha 控制锐化的最大强度
        self.max_alpha = max_alpha
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.intensity = 1.0

        # 预生成高斯核 (可微)
        self.gaussian_blur = K.filters.GaussianBlur2d(
            (kernel_size, kernel_size),
            (sigma, sigma)
        )

    def set_intensity(self, intensity):
        self.intensity = intensity

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        if self.intensity == 0:
            return image

        cur_alpha = self.max_alpha * self.intensity

        # 1. 获取模糊后的图像 (低频分量)
        blurred = self.gaussian_blur(image)

        # 2. 提取高频细节
        high_freq = image - blurred

        # 3. 将高频细节加权叠加回原图
        sharpened = image + cur_alpha * high_freq

        # 确保数值范围不溢出 [-1, 1]
        return sharpened.clamp(-1.0, 1.0)
import torch.nn as nn
from kornia.filters import median_blur


class MF(nn.Module):
    def __init__(self, kernel):
        super(MF, self).__init__()
        self.max_kernel = kernel  # 改名
        self.intensity = 1.0

    def set_intensity(self, intensity):
        self.intensity = intensity

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        if self.intensity == 0:
            return image

        # 动态计算 kernel 大小，确保结果为奇数
        cur_kernel = int(self.max_kernel * self.intensity)
        if cur_kernel % 2 == 0:
            cur_kernel -= 1

        # Kernel 必须 >= 3 才有滤波效果
        if cur_kernel < 3:
            return image

        return median_blur(image, (cur_kernel, cur_kernel))
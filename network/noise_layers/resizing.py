import torch.nn.functional as F
import torch.nn as nn


class Resizing(nn.Module):
    """
    分辨率缩放噪声 (Downsample -> Upsample)
    模拟图像被压缩分辨率后再恢复原尺寸时的信息丢失
    """

    def __init__(self, min_scale=0.5):
        super(Resizing, self).__init__()
        # min_scale 代表最极端情况下，图像分辨率缩小到的比例 (如 0.5 代表长宽各一半)
        self.min_scale = min_scale
        self.intensity = 1.0

    def set_intensity(self, intensity):
        self.intensity = intensity

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        if self.intensity == 0:
            return image

        # intensity = 0 时，scale_factor = 1.0
        # intensity = 1 时，scale_factor = min_scale
        cur_scale = 1.0 - (1.0 - self.min_scale) * self.intensity

        if cur_scale >= 1.0:
            return image

        _, _, H, W = image.size()

        # 降采样 (丢失信息)
        downsampled = F.interpolate(image, scale_factor=cur_scale, mode='bilinear', align_corners=False)
        # 恢复原分辨率 (双线性插值)
        restored = F.interpolate(downsampled, size=(H, W), mode='bilinear', align_corners=False)

        return restored
import torch.nn as nn
import kornia.augmentation as K


class ColorJitter(nn.Module):
    """
    Color & Photometric Distortion Layer (StegaStamp style).
    Simulates physical lighting and printer/camera color gamut shifts.
    """

    def __init__(self, brightness=0.3, contrast=0.5, saturation=1.0, hue=0.1):
        super(ColorJitter, self).__init__()
        # 保存最大扰动幅度
        self.max_brightness = brightness
        self.max_contrast = contrast
        self.max_saturation = saturation
        self.max_hue = hue
        self.intensity = 1.0

    def set_intensity(self, intensity):
        self.intensity = intensity

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        if self.intensity == 0:
            return image

        # 动态计算当前的颜色扰动幅度
        cur_b = self.max_brightness * self.intensity
        cur_c = self.max_contrast * self.intensity
        cur_s = self.max_saturation * self.intensity
        cur_h = self.max_hue * self.intensity

        # 1. 转换到 [0, 1] 空间供 Kornia 处理
        image_01 = (image + 1.0) / 2.0

        # p=1.0 保证每次只要进到这里就必然执行颜色增强
        aug = K.ColorJitter(
            brightness=cur_b,
            contrast=cur_c,
            saturation=cur_s,
            hue=cur_h,
            p=1.0
        ).to(device=image.device, dtype=image.dtype)

        image_jitter = aug(image_01)

        # 2. 转换回 [-1, 1] 并截断防越界
        return (image_jitter * 2.0 - 1.0).clamp(-1.0, 1.0)
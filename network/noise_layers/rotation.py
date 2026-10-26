import torch.nn as nn
import kornia.augmentation as K


class Rotation(nn.Module):
    def __init__(self, degrees):
        super(Rotation, self).__init__()
        self.max_degrees = degrees
        self.intensity = 1.0

    def set_intensity(self, intensity):
        self.intensity = intensity

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        cur_degrees = self.max_degrees * self.intensity

        if cur_degrees == 0:
            return image

        # 动态生成并应用变换 (转移至与 image 相同的 device)
        aug = K.RandomRotation(degrees=cur_degrees, p=1.0, keepdim=True).to(device=image.device, dtype=image.dtype)
        return aug(image)
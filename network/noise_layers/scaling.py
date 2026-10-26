import torch.nn as nn
import kornia.augmentation as K


class Scaling(nn.Module):
    def __init__(self, scale_range=0.8):
        super(Scaling, self).__init__()
        self.max_scale = scale_range
        self.intensity = 1.0

    def set_intensity(self, intensity):
        self.intensity = intensity

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        if self.intensity == 0:
            return image

        # intensity = 0 时，返回 1.0 (不缩放)
        # intensity = 1 时，返回 max_scale (如 0.8)
        cur_scale = 1.0 - (1.0 - self.max_scale) * self.intensity

        aug = K.RandomAffine(degrees=0.0, scale=(cur_scale, cur_scale), p=1.0, keepdim=True).to(device=image.device, dtype=image.dtype)
        return aug(image)
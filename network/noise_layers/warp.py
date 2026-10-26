import torch.nn as nn
import kornia.augmentation as K


class Warp(nn.Module):
    def __init__(self, distortion_scale=0.5):
        super(Warp, self).__init__()
        self.max_distortion_scale = distortion_scale
        self.intensity = 1.0

    def set_intensity(self, intensity):
        self.intensity = intensity

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        cur_distortion_scale = self.max_distortion_scale * self.intensity

        if cur_distortion_scale == 0:
            return image

        aug = K.RandomPerspective(distortion_scale=cur_distortion_scale, p=1.0).to(device=image.device, dtype=image.dtype)
        return aug(image)
import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def set_intensity(self, intensity):
        pass # Identity 不需要强度控制

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        return image
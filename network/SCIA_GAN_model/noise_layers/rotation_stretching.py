import torch
import torch.nn as nn
import numpy as np
import torchvision.utils
from torchvision.transforms.functional import rotate
import torchvision.transforms.functional as F


def center_crop(tensor, output_size):
    """
    对输入Tensor进行中心裁剪

    参数:
        tensor: 输入Tensor (C, H, W) 或 (B, C, H, W)
        output_size: 目标尺寸 (height, width) 或 int(正方形裁剪)

    返回:
        中心裁剪后的Tensor
    """
    # 确保output_size是(height, width)格式
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    crop_h, crop_w = output_size

    # 获取输入尺寸
    if tensor.dim() == 3:  # (C, H, W)
        _, h, w = tensor.shape
        batch_mode = False
    elif tensor.dim() == 4:  # (B, C, H, W)
        _, _, h, w = tensor.shape
        batch_mode = True
    else:
        raise ValueError("输入Tensor必须是3D(C,H,W)或4D(B,C,H,W)")

    # 计算裁剪起始位置
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    # 执行裁剪
    if batch_mode:
        return tensor[:, :, top:top + crop_h, left:left + crop_w]
    else:
        return tensor[:, top:top + crop_h, left:left + crop_w]
class Rotation(nn.Module):
    def __init__(self, angle):
        super(Rotation, self).__init__()
        self.angle = angle

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        rotated_image1 = F.rotate(image,
                                 self.angle,
                                 interpolation=F.InterpolationMode.BILINEAR,
                                 expand=True,
                                 fill=0)  # 用0填充空白区域
        B,C,H,W = image.shape
        resized_image = F.resize(rotated_image1, [H, W])
        B1, C1, H1, W1 = rotated_image1.shape
        resized_image1 = F.resize(rotated_image1, [H1, W1])
        rotated_image2 = F.rotate(resized_image1,
                                 -self.angle,
                                 interpolation=F.InterpolationMode.BILINEAR,
                                 expand=True,
                                 fill=0)  # 用0填充空白区域
        cropped_image = center_crop(rotated_image2, (H, W))  # 裁剪到4x3
        return resized_image, cropped_image

class Stretching(nn.Module):
    def __init__(self, x, y):
        super(Stretching, self).__init__()
        self.x = x
        self.y = y

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        B,C,H,W = image.shape

        resized_image = F.resize(image, [H*self.x, W*self.y])

        resized_image1 = F.resize(image, [H, W])

        return resized_image, resized_image1


class Rotation_Stretching(nn.Module):
    def __init__(self, angle,x,y):
        super(Rotation_Stretching, self).__init__()
        self.angle = angle
        self.x = x
        self.y = y
    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        B,C,H,W = image.shape
        rotated_image = F.rotate(image,
                                 self.angle,
                                 interpolation=F.InterpolationMode.BILINEAR,
                                 expand=True,
                                 fill=0)  # 用0填充空白区域
        B1, C1, H1, W1 = rotated_image.shape
        torchvision.utils.save_image(rotated_image,"/media/WRL/D/WRL/MBRS-main/1.png")
        resized_image = F.resize(rotated_image, [H1*self.x, W1*self.y])
        torchvision.utils.save_image(resized_image,"/media/WRL/D/WRL/MBRS-main/2.png")


        B2, C2, H2, W2 = resized_image.shape
        a = min(H2,W2)
        resized_image1 = F.resize(resized_image, [a, a])
        torchvision.utils.save_image(resized_image1, "/media/WRL/D/WRL/MBRS-main/3.png")

        rotated_image1 = F.rotate(resized_image1,
                                 -self.angle,
                                 interpolation=F.InterpolationMode.BILINEAR,
                                 expand=True,
                                 fill=0)  # 用0填充空白区域

        torchvision.utils.save_image(rotated_image1, "/media/WRL/D/WRL/MBRS-main/4.png")
        cropped_image = center_crop(rotated_image1, (H, W))  # 裁剪到4x3
        torchvision.utils.save_image(cropped_image, "/media/WRL/D/WRL/MBRS-main/5.png")
        return cropped_image

class Stretching_Rotation(nn.Module):
    def __init__(self, x,y,angle):
        super(Stretching_Rotation, self).__init__()
        self.angle = angle
        self.x = x
        self.y = y
    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        B,C,H,W = image.shape
        torchvision.utils.save_image(image, "/media/WRL/D/WRL/MBRS-main/11.png")
        resized_image = F.resize(image, [H * self.x, W * self.y])
        torchvision.utils.save_image(resized_image, "/media/WRL/D/WRL/MBRS-main/6.png")

        rotated_image = F.rotate(resized_image,
                                 self.angle,
                                 interpolation=F.InterpolationMode.BILINEAR,
                                 expand=True,
                                 fill=0)  # 用0填充空白区域
        torchvision.utils.save_image(rotated_image,"/media/WRL/D/WRL/MBRS-main/7.png")





        rotated_image1 = F.rotate(rotated_image,
                                 -self.angle,
                                 interpolation=F.InterpolationMode.BILINEAR,
                                 expand=True,
                                 fill=0)  # 用0填充空白区域
        torchvision.utils.save_image(rotated_image1, "/media/WRL/D/WRL/MBRS-main/8.png")
        cropped_image = center_crop(rotated_image1, (H*self.x, W*self.y))  # 裁剪到4x3
        torchvision.utils.save_image(cropped_image, "/media/WRL/D/WRL/MBRS-main/9.png")

        resized_image1 = F.resize(cropped_image, [H, W])
        torchvision.utils.save_image(resized_image1, "/media/WRL/D/WRL/MBRS-main/10.png")

        return resized_image1

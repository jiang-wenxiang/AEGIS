from numpy.core.fromnumeric import cumprod, shape
from model.dct_ycbcr import rgb2ycbcr_batch, ycbcr2rgb_batch, dwt, iwt, dct_dxc, idct_dxc
import torch
import torch.nn as nn
# from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu

from noise_layers.crop import random_float
import numpy as np
from model.Ltransformer import CVTEncoder


class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 4, (3,3), padding=(1,1)),
            nn.Conv2d(4, 1, (3,3), padding=(1,1))
        )
        # freeze(self.conv)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        # x = Ms * x

        return Ms

class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self,FLAGS):
        super(Encoder, self).__init__()
        self.H = FLAGS.image_size
        self.W = FLAGS.image_size

        # self.chan = 1024

        # self.conv_layers = nn.Sequential(
        #     ConvBNRelu(64 + config.message_length, self.chan),
        #     ConvBNRelu(self.chan, self.chan),
        #     ConvBNRelu(self.chan, self.chan),
        #     ConvBNRelu(self.chan, self.chan),
        #     ConvBNRelu(self.chan, self.chan),
        # )
        self.cvt = nn.Sequential(
            CVTEncoder(              image_size=32,
                                     in_channels=64+FLAGS.redundant_length,
                                     depth = 4,
                                     heads = 6,
                                     share_kv=True,
                                    length=FLAGS.redundant_length
                                     )
        )
        # self.cam = ChannelAttention(in_planes=64)
        self.sa = SpatialAttentionModul()
        # self.scam = SCAM(in_planes=3)
        self.final_layer = nn.Conv2d(64+FLAGS.redundant_length, 64, kernel_size=1)

    def forward(self, image, message):
        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        message = message * 2.0 - 1.0

        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)          # [12, 30, 1, 1]


        # zzw modify
        image_ycbcr = rgb2ycbcr_batch(images=image, cuda=True)/255.

        y_dct = dct_dxc(image_ycbcr[:,0,:,:])

        expanded_message = expanded_message.expand(-1,-1, y_dct.shape[2], y_dct.shape[3])    # [12, 30, 128, 128]
        concat1 = torch.cat([expanded_message, y_dct], dim=1)

        # image [12, 3, 128, 128]
        #ywy modify
        encoded_image = self.cvt(concat1)    # [12, 64, 128, 128]
        im_w1 = self.final_layer(encoded_image)

        y_idct = idct_dxc(im_w1)  # [b,1,128,128]
        image_dct_ycbcr = torch.cat([y_idct.unsqueeze(1), (image_ycbcr[:,1,:,:]).unsqueeze(1), (image_ycbcr[:,2,:,:]).unsqueeze(1)], dim=1) * 255.
        im_w = ycbcr2rgb_batch(image_dct_ycbcr, cuda=True)

        weights = self.sa(image)
        # im = im_w
        # weights1 = (torch.ones_like(image) * 0.2).cuda(1)
        im = im_w*weights + image*(1-weights)



        return im



def get_random_rectangle_inside_for_encoder(image, height_ratio_range, width_ratio_range):

    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[1]) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[0]) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = (image_height - remaining_height) // 2

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = (image_width - remaining_width) // 2

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False
if __name__ == '__main__':
    image = torch.randn((1,3,128,128))
    c = image.shape[1]
    # model = SCAM(in_planes=c)
    #  model2 = ChannelAttention(channel=c)
    # output = model(image)
    model = SpatialAttentionModul()
    output= model(image)


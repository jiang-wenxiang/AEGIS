import torch.nn as nn
import torch

import os
import sys
sys.path.append(os.path.join(os.path.abspath(__file__).split('HiDDeN')[0], 'HiDDeN'))
"""
加一层conv层，将transformer的输出映射为config.message_length 
修改transformer的dim 不让transformer的dim为in_channels
"""

from model.conv_bn_relu import ConvBNRelu
from model.dct_ycbcr import rgb2ycbcr_batch, ycbcr2rgb_batch, dwt, iwt, dct_dxc

# import stn
from model.Ltransformer import CVTDecoder
class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self,FLAGS):

        super(Decoder, self).__init__()
        # self.chan = 1024

        # self.layers = nn.Sequential(
        #     ConvBNRelu(64, self.chan),
        #     # ChannelAttention(in_planes=self.chan),  # add
        #     ConvBNRelu(self.chan, self.chan),
        #     # ChannelAttention(in_planes=self.chan),  # add
        #     ConvBNRelu(self.chan, self.chan),
        #     # ChannelAttention(in_planes=self.chan),  # add
        #     ConvBNRelu(self.chan, self.chan),
        #     # ChannelAttention(in_planes=self.chan),  # add
        #     ConvBNRelu(self.chan, self.chan),
        #     # ChannelAttention(in_planes=self.chan),
        #     ConvBNRelu(self.chan, config.message_length),
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # )
        self.cvt = CVTDecoder(image_size=32,
                                 in_channels=64,
                                 depth = 4,
                                 heads = 6,
                                 share_kv=True
                                     )
        self.linear = nn.Linear(64, FLAGS.redundant_length)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.chanatten = ChannelAttention(in_planes=64)
    def forward(self, image_with_wm):
        # zzw add stn
        # image_with_wm_stn = self.stn(image_with_wm)
        # image_with_wm_stn = image_with_wm
        # zzw modify
        image_ycbcr = rgb2ycbcr_batch(images=image_with_wm, cuda=True)/255.

        y_dct = dct_dxc(image_ycbcr[:,0,:,:])  # [b,1,128,128]
        y_dct_atten_w = self.chanatten(y_dct)
        y_dct = y_dct*y_dct_atten_w.expand_as(y_dct)

        x = self.cvt(y_dct)
        x = self.pool(x)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return torch.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("x", x.get_device())
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # print("avg_out", avg_out.get_device())
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        w = self.sigmoid(out)
        # print("w[0:10]:", w.squeeze(-1).squeeze(-1).squeeze(-1)[0,0:10])
        return w
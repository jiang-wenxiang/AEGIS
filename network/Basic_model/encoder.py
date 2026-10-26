import torch
import torch.nn as nn
from .options import HiDDenConfiguration
from .conv_bn_relu import ConvBNRelu


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1, -1, self.H, self.W)
        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w


class FlexibleEncoder(nn.Module):
    """
    修改后的 Encoder：
    1. 支持动态输入尺寸 (适应 DWT 后的 32x32, 64x64)
    2. 支持自定义输入输出通道 (in_channels, out_channels)
    """

    def __init__(self, config: HiDDenConfiguration, in_channels=3, out_channels=3):
        super(FlexibleEncoder, self).__init__()
        # self.H = config.H  <-- 移除硬编码尺寸
        # self.W = config.W  <-- 移除硬编码尺寸
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        # 修改 1: 输入通道参数化
        layers = [ConvBNRelu(in_channels, self.conv_channels)]

        for _ in range(config.encoder_blocks - 1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)

        # 修改 2: 拼接后的通道数计算 (conv_channels + in_channels + message_length)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + in_channels + config.message_length,
                                             self.conv_channels)

        # 修改 3: 输出通道参数化
        self.final_layer = nn.Conv2d(self.conv_channels, out_channels, kernel_size=1)

    def forward(self, image, message):
        # 获取当前输入的动态尺寸 (例如 32x32 或 64x64)
        curr_h, curr_w = image.shape[2], image.shape[3]

        # Expanded message to match the current image size
        expanded_message = message.unsqueeze(-1).unsqueeze(-1)
        expanded_message = expanded_message.expand(-1, -1, curr_h, curr_w)

        encoded_image = self.conv_layers(image)

        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        # 残差连接 (HiDDeN 的核心思想是输出残差或直接输出修改图，这里保持原逻辑直接输出)
        # 如果原模型有残差连接逻辑，请在这里加上： return image + im_w
        return im_w


class ResFlexibleEncoder(nn.Module):
    """
    修改后的 Encoder：
    1. 支持动态输入尺寸 (适应 DWT 后的 32x32, 64x64)
    2. 支持自定义输入输出通道 (in_channels, out_channels)
    """

    def __init__(self, config: HiDDenConfiguration, in_channels=3, out_channels=3, res_output=False):
        super(ResFlexibleEncoder, self).__init__()
        # self.H = config.H  <-- 移除硬编码尺寸
        # self.W = config.W  <-- 移除硬编码尺寸
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        # 修改 1: 输入通道参数化
        layers = [ConvBNRelu(in_channels, self.conv_channels)]

        for _ in range(config.encoder_blocks - 1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)

        # 修改 2: 拼接后的通道数计算 (conv_channels + in_channels + message_length)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + in_channels + config.message_length,
                                             self.conv_channels)

        # 修改 3: 输出通道参数化
        self.final_layer = nn.Conv2d(self.conv_channels, out_channels, kernel_size=1)

        self.res_output = res_output

    def forward(self, cat_image, ori_image, message):

        # 获取当前输入的动态尺寸 (例如 32x32 或 64x64)
        curr_h, curr_w = cat_image.shape[2], cat_image.shape[3]

        # Expanded message to match the current image size
        expanded_message = message.unsqueeze(-1).unsqueeze(-1)
        expanded_message = expanded_message.expand(-1, -1, curr_h, curr_w)

        encoded_image = self.conv_layers(cat_image)

        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, cat_image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        # 残差连接 (HiDDeN 的核心思想是输出残差或直接输出修改图，这里保持原逻辑直接输出)
        # 如果原模型有残差连接逻辑，请在这里加上： return image + im_w
        if self.res_output:
            return ori_image + im_w
        return im_w

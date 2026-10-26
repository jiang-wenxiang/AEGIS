import torch
import torch.nn as nn
from .attention import AttentiveEncoder, AttentiveDecoder
from ..Noise import Noise


class RivaGAN_Encoder(nn.Module):
    """
    封装 RivaGAN 的 Encoder，使其对外表现得像一个普通的 2D 图像 Encoder。
    输入: (B, 3, H, W) in [0, 1]
    输出: (B, 3, H, W) in [0, 1]
    """

    def __init__(self, message_length):
        super(RivaGAN_Encoder, self).__init__()
        # 实例化原始的 AttentiveEncoder
        self.core_encoder = AttentiveEncoder(data_dim=message_length)

    def forward(self, image, message):
        # 1. 维度适配: (B, C, H, W) -> (B, C, 1, H, W)
        video_input = image.unsqueeze(2)

        # 2. 归一化适配: [0, 1] -> [-1, 1]
        # RivaGAN 内部使用 Tanh，期望输入范围是 -1 到 1
        # video_input = video_input * 2.0 - 1.0

        # 3. 核心编码
        encoded_video = self.core_encoder(video_input, message)

        # 4. 还原维度与范围: (B, C, 1, H, W) -> (B, C, H, W) in [0, 1]
        encoded_image = encoded_video.squeeze(2)

        # 加上 clamp 防止数值溢出
        return torch.clamp(encoded_image, -1.0, 1.0)


class RivaGAN_Decoder(nn.Module):
    """
    封装 RivaGAN 的 Decoder。
    输入: (B, 3, H, W) in [0, 1]
    输出: (B, L)
    """

    def __init__(self, encoder_instance):
        super(RivaGAN_Decoder, self).__init__()
        # 关键：RivaGAN Decoder 必须共享 Encoder 内部的 attention 模块
        # 这里传入的是我们封装后的 RivaGAN_Encoder，所以要取 .core_encoder
        self.core_decoder = AttentiveDecoder(encoder_instance.core_encoder)

    def forward(self, image):
        # 1. 维度适配: (B, C, H, W) -> (B, C, 1, H, W)
        video_input = image.unsqueeze(2)

        # 2. 归一化适配: [0, 1] -> [-1, 1]
        # video_input = video_input * 2.0 - 1.0

        # 3. 核心解码
        decoded_message = self.core_decoder(video_input)

        return decoded_message


class RivaGAN_EncoderDecoder(nn.Module):
    """
    符合 ModelTrainer 接口的顶层封装
    """

    def __init__(self, H, W, message_length, noise_layers):
        super(RivaGAN_EncoderDecoder, self).__init__()

        # 1. 实例化封装后的 Encoder
        self.encoder = RivaGAN_Encoder(message_length)

        # 2. 实例化 Noise
        self.noise = Noise(noise_layers)

        # 3. 实例化封装后的 Decoder (传入 encoder 实例以共享权重)
        self.decoder = RivaGAN_Decoder(self.encoder)

    def forward(self, image, message):
        # 这里的流程和普通 MBRS 完全一致，不需要特殊处理
        encoded_image = self.encoder(image, message)
        noised_image = self.noise([encoded_image, image])
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message
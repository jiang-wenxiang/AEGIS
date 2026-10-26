import torch
import torch.nn as nn
from .Encoder_U import DW_Encoder
from .Decoder_U import DW_Decoder
from ..Noise import Noise


class EditGuard_Model(nn.Module):
    def __init__(self, H, W, message_length, noise_layers):
        super(EditGuard_Model, self).__init__()

        # 1. 实例化 Encoder (BEM)
        # 关键点：根据论文，attention 必须设置为 'se' (Squeeze-and-Excitation)
        self.encoder = DW_Encoder(message_length, attention='se')

        # 2. 实例化 Noise (使用您自己的噪声层)
        self.noise = Noise(noise_layers)

        # 3. 实例化 Decoder (BRM)
        # 同样使用 'se' attention
        self.decoder = DW_Decoder(message_length, attention='se')

    def forward(self, image, message):
        # 1. 编码：生成带水印图像
        # Encoder_U 的 forward 返回的是叠加后的图像 (image + residual)
        encoded_image = self.encoder(image, message)

        # 2. 攻击：经过噪声层
        # 传入 [encoded_image, image] 以兼容 Cropout 等依赖原图的噪声
        noised_image = self.noise([encoded_image, image])

        # 3. 解码：提取水印
        decoded_message = self.decoder(noised_image)

        # 返回标准三元组，符合 ModelTrainer 的接口
        return encoded_image, noised_image, decoded_message
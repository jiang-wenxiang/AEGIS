import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..Noise import Noise


class StegaStampEncoder(nn.Module):
    def __init__(self, H=400, W=400, message_length=100):
        super(StegaStampEncoder, self).__init__()
        self.H = H
        self.W = W
        self.secret_channels = 3  # StegaStamp 原始设计是 reshape 成 (50,50,3)

        # 1. 秘密信息预处理
        # 原始代码: Dense(7500) -> Reshape((50, 50, 3))
        # 这里假设输入图 H,W 能够被 8 整除 (因为 UpSampling size=8)
        # 如果 H=256, W=256，建议调整 secret_map_size 为 32x32
        self.secret_map_size = H // 8
        flat_size = self.secret_map_size * self.secret_map_size * self.secret_channels

        self.secret_dense = nn.Linear(message_length, flat_size)

        # 2. 卷积层 (U-Net Downsampling)
        # 输入通道 = 3 (图像) + 3 (秘密信息图) = 6
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # 3. 上采样与融合 (U-Net Upsampling)
        self.up6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1)

        self.up7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)

        self.up8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1)

        # 最后一层：原始输入(6通道) + conv8上采样(32通道)
        self.up9 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(32 + 6 + 32, 32, kernel_size=3,
                               padding=1)  # Note: +32 from up9 output, +6 from input, +32 from conv1?
        # 修正：StegaStamp原文的merge9是 concatenate([conv1, up9, inputs])
        # conv1 是 32通道, up9 是 32通道, inputs 是 6通道 => 总共 70 通道
        self.conv9_real = nn.Conv2d(32 + 32 + 6, 32, kernel_size=3, padding=1)

        self.conv10 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.residual = nn.Conv2d(32, 3, kernel_size=1, padding=0)

    def forward(self, image, secret):
        # image: (B, 3, H, W)
        # secret: (B, L)

        # 处理 Secret
        secret = self.secret_dense(secret)
        secret = secret.view(-1, self.secret_channels, self.secret_map_size, self.secret_map_size)
        # 上采样到与图像同尺寸
        secret_enlarged = F.interpolate(secret, size=(self.H, self.W), mode='nearest')

        # 拼接
        inputs = torch.cat([secret_enlarged, image], dim=1)

        # U-Net Down
        conv1 = F.relu(self.conv1(inputs))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        conv5 = F.relu(self.conv5(conv4))

        # U-Net Up
        up6 = self.up6(conv5)
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = F.relu(self.conv6(merge6))

        up7 = self.up7(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = F.relu(self.conv7(merge7))

        up8 = self.up8(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = F.relu(self.conv8(merge8))

        up9 = self.up9(conv8)
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = F.relu(self.conv9_real(merge9))

        conv10 = F.relu(self.conv10(conv9))

        # 生成残差并叠加
        residual = self.residual(conv10)
        encoded_image = image + residual
        return encoded_image


class StegaStampDecoder(nn.Module):
    def __init__(self, H=400, W=400, message_length=100):
        super(StegaStampDecoder, self).__init__()
        self.H = H
        self.W = W

        # --- 1. Spatial Transformer Network (STN) 矫正模块 ---
        # StegaStamp 的核心亮点，用于抵抗透视变换
        self.stn_params = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        # 计算 Flatten 后的尺寸: H/8 * W/8 * 128
        self.flatten_dim = (H // 8) * (W // 8) * 128
        self.fc_loc = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)  # 输出 2x3 的仿射变换矩阵参数
        )

        # 初始化 STN 为单位变换 (Identity Transform)
        # 这是一个关键的 Trick，否则 STN 很难训练
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # --- 2. 核心解码器 ---
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        # 计算解码 Flatten 后的尺寸: H/32 * W/32 * 128
        self.decoder_flatten_dim = (H // 32) * (W // 32) * 128

        self.fc_decode = nn.Sequential(
            nn.Linear(self.decoder_flatten_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, message_length)
        )

    def stn(self, x):
        xs = self.stn_params(x)
        xs = xs.view(-1, self.flatten_dim)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # 生成网格并采样
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, image):
        # 1. 矫正图像
        x = self.stn(image)

        # 2. 提取特征
        features = self.decoder(x)
        features = features.view(-1, self.decoder_flatten_dim)

        # 3. 预测消息
        predicted_message = self.fc_decode(features)
        return predicted_message


# 兼容 ModelTrainer 的封装类
class StegaStamp_EncoderDecoder(nn.Module):
    def __init__(self, H, W, message_length, noise_layers):
        super(StegaStamp_EncoderDecoder, self).__init__()
        self.encoder = StegaStampEncoder(H, W, message_length)
        self.noise = Noise(noise_layers)

        self.decoder = StegaStampDecoder(H, W, message_length)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_image = self.noise([encoded_image, image])
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message
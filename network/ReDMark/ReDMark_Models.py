import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio  # 需要安装 scipy: pip install scipy
import numpy as np
import os
from ..Noise import Noise


# ==========================================
# 辅助层：循环卷积 (Circular Conv2D)
# ==========================================
class CircularConv2d(nn.Module):
    """
    模拟 ReDMark 中的 Circular Conv2D (kernel_size=2)
    TF 的 'same' padding 配合 2x2 核在频域处理时的特殊行为
    我们在右侧和下侧进行循环填充，保持输出尺寸不变
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, activation='elu'):
        super(CircularConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.act = nn.ELU() if activation == 'elu' else nn.Identity()

    def forward(self, x):
        # 1. 手动进行循环填充 (Circular Padding)
        # padding format: (left, right, top, bottom)
        # 对于 2x2 核，我们在右边补1列，下边补1行，取自左边和上边
        x_pad = F.pad(x, (0, 1, 0, 1), mode='circular')

        # 2. 卷积
        out = self.conv(x_pad)

        # 3. 激活
        return self.act(out)


# ==========================================
# 辅助层：DCT/IDCT 变换层 (固定权重)
# ==========================================
class DCT_Layer(nn.Module):
    def __init__(self, mat_path, type='DCT'):
        super(DCT_Layer, self).__init__()

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Missing weight file: {mat_path}")

        # 加载 .mat 文件
        # content 形状通常是 (1, 1, 64, 64) -> (H, W, In, Out) for TF
        data = sio.loadmat(mat_path)
        if type == 'DCT':
            weights = data['DCT_coef']
        else:
            weights = data['IDCT_coef']

        # ================= [新增修复代码] =================
        # 1. 检查并修正维度 (Fix: Shape mismatch)
        # 原始 .mat 可能是 (64, 64)，需要先 reshape 成 TF 格式 (1, 1, 64, 64)
        if weights.ndim == 2:
            h, w = weights.shape
            # 强制转换为 4 维: (H=1, W=1, In=64, Out=64)
            weights = weights.reshape(1, 1, h, w)
        # =================================================

        # 确保形状是 (64, 64, 1, 1) -> (Out, In, H, W) for PyTorch Conv2d 1x1
        # TF: (1, 1, 64, 64) -> PyTorch: (64, 64, 1, 1)
        # 我们需要 transpose: (3, 2, 0, 1)
        weights = np.transpose(weights, (3, 2, 0, 1))

        # 创建 1x1 卷积层
        self.conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)

        # 赋值并冻结
        self.conv.weight.data = torch.from_numpy(weights).float()
        self.conv.weight.requires_grad = False  # 冻结 DCT 参数

    def forward(self, x):
        return self.conv(x)


# ==========================================
# 编码器 (ReDMark Encoder)
# ==========================================
class ReDMark_Encoder(nn.Module):
    def __init__(self, message_length, H, W, dct_path, idct_path):
        super(ReDMark_Encoder, self).__init__()
        self.H = H
        self.W = W
        self.message_length = message_length
        self.block_size = 8

        # 1. Space-to-Depth (H, W, 1) -> (H/8, W/8, 64)
        # 注意：PyTorch 的 PixelUnshuffle 是 (C, H, W) -> (C*r*r, H/r, W/r)
        self.space2depth = nn.PixelUnshuffle(downscale_factor=8)

        # 2. DCT Layer
        self.dct = DCT_Layer(dct_path, type='DCT')

        # 3. 水印处理
        # ReDMark 直接拼接水印，我们需要将 message 扩展到空间尺寸
        # 输入 DCT 是 64 通道，水印假设扩展为 1 通道，共 65 通道
        self.concat_channels = 64 + 1

        # 4. 核心卷积层
        # layer 1: 1x1 Conv
        self.conv1 = nn.Conv2d(self.concat_channels, 64, kernel_size=1)
        self.elu1 = nn.ELU()

        # layer 2-5: Circular Conv 2x2
        self.c_conv2 = CircularConv2d(64, 64)
        self.c_conv3 = CircularConv2d(64, 64)
        self.c_conv4 = CircularConv2d(64, 64)
        self.c_conv5 = CircularConv2d(64, 64)

        # 5. IDCT Layer
        self.idct = DCT_Layer(idct_path, type='IDCT')

        # 6. Depth-to-Space
        self.depth2space = nn.PixelShuffle(upscale_factor=8)

    def forward(self, image, message):
        # image: (B, 3, H, W) -> 我们只取亮度通道或者对每个通道分别做，
        # ReDMark 原文是针对单通道 (Luma) 的。
        # 为了适配您的 RGB 流程，我们在 RGB 三通道上共享参数运行，或者只加在 Y 通道
        # 这里简化为：只加在 G 通道或平均通道，或者把 RGB 视为 Batch 处理
        # **为了兼容性，我们这里将 (B, 3, H, W) 视为 (B*3, 1, H, W) 处理**

        b, c, h, w = image.shape
        x = image.reshape(b * c, 1, h, w)

        # 1. 预处理
        # 转换范围：输入是 [-1, 1]，ReDMark 偏好 [-0.5, 0.5]
        x_centered = x * 0.5

        # Space to Depth: (B*3, 1, H, W) -> (B*3, 64, H/8, W/8)
        x_s2d = self.space2depth(x_centered)

        # DCT
        x_dct = self.dct(x_s2d)

        # 2. 水印拼接
        # message: (B, L) -> 需要扩展到 (B*3, 1, H/8, W/8)
        # 简单的重复平铺策略
        msg_expanded = message.view(b, 1, -1)  # (B, 1, L)
        # 重复 3 次以匹配 RGB
        msg_expanded = msg_expanded.repeat(3, 1, 1)  # (B*3, 1, L)

        target_size = (h // 8) * (w // 8)
        if msg_expanded.shape[2] < target_size:
            # 如果消息短，就循环填充
            repeat_times = target_size // msg_expanded.shape[2] + 1
            msg_expanded = msg_expanded.repeat(1, 1, repeat_times)

        msg_expanded = msg_expanded[:, :, :target_size]  # 截断
        msg_map = msg_expanded.view(b * c, 1, h // 8, w // 8)

        # 拼接: (B*3, 65, H/8, W/8)
        encoder_input = torch.cat([x_dct, msg_map], dim=1)

        # 3. 卷积网络
        out = self.elu1(self.conv1(encoder_input))
        out = self.c_conv2(out)
        out = self.c_conv3(out)
        out = self.c_conv4(out)
        out = self.c_conv5(out)

        # 4. 后处理
        out_idct = self.idct(out)

        # 残差连接 (在 Space2Depth 域)
        # ReDMark 原代码: residual_add = idct_out + input_space2depth
        out_res = out_idct + x_s2d

        # Depth to Space
        out_image = self.depth2space(out_res)

        # 反归一化
        out_image = out_image * 2.0

        # 恢复 RGB 形状
        encoded_image = out_image.reshape(b, c, h, w)

        return encoded_image


# ==========================================
# 解码器 (ReDMark Decoder)
# ==========================================
class ReDMark_Decoder(nn.Module):
    def __init__(self, message_length, H, W, dct_path):
        super(ReDMark_Decoder, self).__init__()
        self.message_length = message_length
        self.H = H
        self.W = W

        # 1. Space-to-Depth
        self.space2depth = nn.PixelUnshuffle(downscale_factor=8)

        # 2. DCT
        self.dct = DCT_Layer(dct_path, type='DCT')

        # 3. 卷积层
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1)
        self.elu1 = nn.ELU()

        self.c_conv2 = CircularConv2d(64, 64)
        self.c_conv3 = CircularConv2d(64, 64)
        self.c_conv4 = CircularConv2d(64, 64)

        # 4. 输出层
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)  # 输出单通道概率图
        self.sig = nn.Sigmoid()

        # 5. 消息提取头
        # 因为我们在 RGB 上都做了嵌入，解码时需要融合 3 通道结果
        # 然后映射回 message_length
        flatten_dim = (H // 8) * (W // 8) * 3
        self.final_fc = nn.Linear(flatten_dim, message_length)

    def forward(self, image):
        b, c, h, w = image.shape
        x = image.reshape(b * c, 1, h, w)

        # 预处理
        x_centered = x * 0.5
        x_s2d = self.space2depth(x_centered)
        x_dct = self.dct(x_s2d)

        # 卷积
        out = self.elu1(self.conv1(x_dct))
        out = self.c_conv2(out)
        out = self.c_conv3(out)
        out = self.c_conv4(out)

        # 输出特征图
        out = self.out_conv(out)  # (B*3, 1, H/8, W/8)
        out = self.sig(out)

        # 提取消息
        # 展平并重组为 (B, -1)
        out_flat = out.reshape(b, -1)  # (B, 3 * H/8 * W/8)

        # 全连接预测
        decoded_message = self.final_fc(out_flat)

        return decoded_message


# ==========================================
# 封装类 (接口适配)
# ==========================================
class ReDMark_EncoderDecoder(nn.Module):
    def __init__(self, H, W, message_length, noise_layers, dct_path='./network/ReDMark/DCT_coef.mat',
                 idct_path='./network/ReDMark/IDCT_coef.mat'):
        super(ReDMark_EncoderDecoder, self).__init__()

        self.encoder = ReDMark_Encoder(message_length, H, W, dct_path, idct_path)

        self.noise = Noise(noise_layers)

        self.decoder = ReDMark_Decoder(message_length, H, W, dct_path)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_image = self.noise([encoded_image, image])
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message
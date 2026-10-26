import torch
import torch.nn as nn
import ptwt  # PyTorch Wavelet Transforms # pip install ptwt torch-dct
import torch_dct  # PyTorch 2D DCT # pip install ptwt torch-dct
from ..Noise import Noise  # 继承你框架原有的 Noise 模块

class DWT_DCT_SVD_Encoder(nn.Module):
    """
    纯 GPU 版 DWT-DCT-SVD Encoder
    输入: image (B, 3, H, W) in [0, 1], message (B, L) in {0, 1}
    输出: encoded_image (B, 3, H, W) in [0, 1]
    """

    def __init__(self, alpha=15.0):
        super(DWT_DCT_SVD_Encoder, self).__init__()
        self.alpha = alpha
        self.saved_S = None

    def forward(self, image, message):
        # 强制切断整条计算图的梯度跟踪，等同于不被训练
        with torch.no_grad():
            B, C, H, W = image.shape

            # 我们仅在 Y 通道或 Channel 0 嵌入
            # 提取形状: (B, 1, 256, 256)
            img_channel = image[:, 0:1, :, :]
            msg_tensor = message.float()  # (B, 64)

            # 1. 批量一级 DWT (使用 haar)
            # 返回低频 LL (B, 1, 128, 128) 和高频 coeffs
            LL, coeffs = ptwt.wavedec2(img_channel, 'haar', level=1)

            # 2. 批量 2D DCT
            dct_LL = torch_dct.dct_2d(LL)

            # 3. 批量 SVD 奇异值分解 (原生支持 GPU Batched 计算)
            # U: (B, 1, 128, 128), S: (B, 1, 128), Vh: (B, 1, 128, 128)
            U, S, Vh = torch.linalg.svd(dct_LL, full_matrices=False)

            # 保存原始奇异值 (clone 保证不受后续 inplace 修改的影响)
            self.saved_S = S.clone()

            # 4. 批量嵌入 64 bits 水印
            # 映射 {0, 1} 到 {-1, 1}
            msg_mapped = msg_tensor * 2.0 - 1.0  # (B, 64)
            S_mark = S.clone()

            # 将水印加到最大的前 64 个奇异值上
            # S_mark[:, 0, :64] 的形状是 (B, 64)
            S_mark[:, 0, :len(msg_tensor[0])] += self.alpha * msg_mapped

            # 5. 批量 SVD 逆重构
            # 使用 diag_embed 将 S (B, 1, 128) 变成对角矩阵 (B, 1, 128, 128)
            S_diag = torch.diag_embed(S_mark)
            dct_LL_mark = U @ S_diag @ Vh

            # 6. 批量 IDCT 逆余弦变换
            LL_mark = torch_dct.idct_2d(dct_LL_mark)

            # 7. 批量 IDWT 逆小波变换
            img_mark_channel = ptwt.waverec2([LL_mark, coeffs], 'haar')

            # 替换回原图
            encoded_images = image.clone()
            encoded_images[:, 0:1, :, :] = img_mark_channel

            # 加上 clamp 防止像素值溢出
            return torch.clamp(encoded_images, -1.0, 1.0)


class DWT_DCT_SVD_Decoder(nn.Module):
    """
    纯 GPU 版 DWT-DCT-SVD Decoder
    输入: noised_image (B, 3, H, W) in [0, 1]
    输出: decoded_message (B, L)
    """

    def __init__(self, encoder_instance, message_length=64):
        super(DWT_DCT_SVD_Decoder, self).__init__()
        self.encoder = encoder_instance
        self.message_length = message_length

    def forward(self, image):
        with torch.no_grad():
            # 提取嵌入了水印的 Channel 0 -> (B, 1, 256, 256)
            img_channel = image[:, 0:1, :, :]

            # 获取 Encoder 缓存的原始宿主奇异值 -> (B, 1, 128)
            original_S = self.encoder.saved_S

            # 1. 批量一级 DWT
            LL, _ = ptwt.wavedec2(img_channel, 'haar', level=1)

            # 2. 批量 2D DCT
            dct_LL = torch_dct.dct_2d(LL)

            # 3. 批量 SVD 提取受攻击后的奇异值矩阵
            _, S_attacked, _ = torch.linalg.svd(dct_LL, full_matrices=False)

            # 4. 批量提取 64 bits 水印
            # 计算受损奇异值与原始奇异值的差值 (B, 64)
            diff = S_attacked[:, 0, :self.message_length] - original_S[:, 0, :self.message_length]

            # 大于 0 判为 1，否则判为 0
            extracted_bits = (diff > 0.0).float()

            return extracted_bits


class DWT_DCT_SVD_EncoderDecoder(nn.Module):
    """
    符合 ModelTrainer 接口的顶层封装
    """

    def __init__(self, H, W, message_length, noise_layers, alpha=15.0):
        super(DWT_DCT_SVD_EncoderDecoder, self).__init__()

        if noise_layers is None:
            noise_layers = []

        self.encoder = DWT_DCT_SVD_Encoder(alpha=alpha)
        self.noise = Noise(noise_layers)
        self.decoder = DWT_DCT_SVD_Decoder(self.encoder, message_length=message_length)

    def forward(self, image, message):
        # 此处不加 torch.no_grad() 是为了包容 self.noise 层。
        # 如果你的 Noise 层包含需要梯度的攻击（例如针对 RivaGAN 的白盒攻击），
        # Noise 层将正常追踪梯度，但进入 Encoder 和 Decoder 内部时由于
        # 显式声明了 with torch.no_grad():，传统水印算法本身绝对不会被追踪梯度。

        encoded_image = self.encoder(image, message)
        noised_image = self.noise([encoded_image, image])
        decoded_message = self.decoder(noised_image)

        return encoded_image, noised_image, decoded_message
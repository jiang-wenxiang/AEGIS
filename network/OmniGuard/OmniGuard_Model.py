import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入 OmniGuard 提供的核心依赖 (确保这些在你的运行目录下)
from .model_invert import Model as OmniGuard_core
from network.Noise import Noise
from .omniguard_cop.omniguard_cop.models import Decoder
from .omniguard_cop.omniguard_cop.transforms import VAE
from .modules import Unet_common as common

# ==========================================
# 1. 编码器层 (Encoder) - 完全闭环，无外部依赖假设
# ==========================================
class OmniGuard_Encoder(nn.Module):
    def __init__(self, secret_len=64):
        super(OmniGuard_Encoder, self).__init__()
        # 1. 实例化 OmniGuard 原版的 INN 主网络
        self.inn_net = OmniGuard_core()
        
        # 2. 实例化 DWT (小波变换)，用于将空间域图像转入频域
        self.dwt = common.DWT()
        
        # 3. 核心补充：将 1D 比特映射到 2D 频域的转换器
        # 为了避免显存爆炸，我们先将 secret_len 映射到 12 * 32 * 32 大小，后续再动态上采样
        self.secret_dense = nn.Sequential(
            nn.Linear(secret_len, 12 * 32 * 32),
            nn.LeakyReLU(0.2, inplace=True)  # 拯救那些 0.000 的死区
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 把扩音器开到 10 倍！
        
    def expand_message_to_tensor(self, message, target_shape):
        """
        内部辅助函数：将 (B, L) 的 message 转化为 (B, 12, H/2, W/2) 的张量，
        以完美匹配小波变换后的 cover_dwt。
        """
        B, C, H, W = target_shape
        
        # 将一维向量推过 Linear 层，输出 (B, 12288)
        x = self.secret_dense(message)
        
        # 重塑为 2D 空间特征图: (B, 12, 32, 32)
        x = x.view(B, 12, 32, 32)
        
        # 动态上采样到目标频域尺寸 (H/2, W/2)，保证能适应任何尺寸的输入图片
        secret_tensor = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return secret_tensor

    def forward(self, image, message):
        """
        image: (B, 3, H, W) 期望值在 [-1, 1]
        message: (B, secret_len) 
        """
        # 1. 图像转频域: (B, 3, H, W) -> (B, 12, H/2, W/2)
        cover_dwt = self.dwt(image)
        
        # 2. 比特转频域图: (B, secret_len) -> (B, 12, H/2, W/2)
        # secret_input = self.expand_message_to_tensor(message, cover_dwt.shape)

        # 在 forward 中：
        secret_input = self.expand_message_to_tensor(message, cover_dwt.shape)
        secret_input = secret_input * self.alpha  # 强制注入超强信号
        
        # 3. 频域拼接与正向隐藏
        # OmniGuard 的 Hinet 会在内部自己完成 concat，我们直接传入即可
        encoded_image, output_z, out_temp, secret_temp = self.inn_net(
            cover_input=cover_dwt, 
            secret_input=secret_input, 
            rev=False
        )
        
        # 4. 截断防溢出，直接输出带有水印的 RGB 图像
        # encoded_image = torch.clamp(encoded_image, -1.0, 1.0)

        # 🚀 替换为 STE 神奇截断写法：
        # 1. 先计算严格截断后的图像
        encoded_image_clamped = torch.clamp(encoded_image, -1.0, 1.0)

        # 2. 直通估计 (Straight-Through Estimator)
        # 前向传播时：结果等于 encoded_image_clamped
        # 反向传播时：(clamped - 原图).detach() 没有梯度，梯度直接顺着 encoded_image 流回网络！
        encoded_final = encoded_image + (encoded_image_clamped - encoded_image).detach()

        return encoded_final


# ==========================================
# 3. 解码器层 (Decoder)
# ==========================================
class OmniGuard_Decoder(nn.Module):
    def __init__(self, secret_len=64, pretrained=True):
        super(OmniGuard_Decoder, self).__init__()
        # 直接使用原版基于 ResNet50 的强大提取器
        self.decoder = Decoder(secret_len=secret_len, pretrained=pretrained, resize=224)

    def forward(self, image):
        # 输入受损图像，直接输出提取出的比特 logit
        predicted_message = self.decoder(image)
        return predicted_message


class OmniGuard_Model(nn.Module):
    def __init__(self, H, W, message_length, noise_layers):
        super(OmniGuard_Model, self).__init__()

        # 1. 实例化 Encoder (BEM)
        # 关键点：根据论文，attention 必须设置为 'se' (Squeeze-and-Excitation)
        self.encoder = OmniGuard_Encoder(message_length)

        # 2. 实例化 Noise (使用您自己的噪声层)
        self.noise = Noise(noise_layers)

        # 3. 实例化 Decoder (BRM)
        # 同样使用 'se' attention
        self.decoder = OmniGuard_Decoder(message_length)

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
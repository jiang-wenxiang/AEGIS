import torch
from torch import nn

from .trustmark import TrustMark
from ..Noise import Noise


class TrustMarkEncoder(nn.Module):
    """输入 [B,3,256,256]、[B,64]，输出 [B,3,256,256]。"""

    def __init__(self, encoder_core):
        super().__init__()
        self.encoder_core = encoder_core

        self.encoder_core.requires_grad_(False)
        self.encoder_core.eval()

    def train(self, mode=True):
        # 官方权重只测试，外部调用train()时仍保持eval
        super().train(False)
        self.encoder_core.eval()
        return self

    @staticmethod
    def expand_message(message):
        if message.ndim != 2 or message.shape[1] != 64:
            raise ValueError(
                f"TrustMark expects message [B,64], got {tuple(message.shape)}"
            )

        # 64个独立消息位扩展到官方模型的100个输入位
        return torch.cat(
            [message.float(), message[:, :36].float()],
            dim=1,
        )

    def forward(self, image, message):
        if image.ndim != 4 or image.shape[1:] != (3, 256, 256):
            raise ValueError(
                "TrustMark expects image [B,3,256,256], "
                f"got {tuple(image.shape)}"
            )

        payload_100 = self.expand_message(message)

        # 官方底层编码器返回：(stego, residual)
        raw_encoded, _ = self.encoder_core(image, payload_100)

        # 保留官方残差处理：限制范围并消除通道颜色偏移
        residual = raw_encoded.clamp(-1.0, 1.0) - image
        residual = residual - residual.mean(
            dim=(2, 3),
            keepdim=True,
        )

        encoded_image = (image + residual).clamp(-1.0, 1.0)

        return encoded_image


class TrustMarkDecoder(nn.Module):
    """输入 [B,3,H,W]，输出 [B,64]概率。"""

    def __init__(self, decoder_core, return_logits=False):
        super().__init__()
        self.decoder_core = decoder_core
        self.return_logits = return_logits

        self.decoder_core.requires_grad_(False)
        self.decoder_core.eval()

    def train(self, mode=True):
        super().train(False)
        self.decoder_core.eval()
        return self

    def forward(self, image):
        # 官方核心解码器输出100个logits
        decoded_100_logits = self.decoder_core.decoder(image)

        # 只返回前64个有效消息位
        decoded_64_logits = decoded_100_logits[:, :64]

        if self.return_logits:
            # 适用于BCEWithLogitsLoss和阈值0
            return decoded_64_logits

        # 适用于当前常见的阈值0.5误码率计算
        return torch.sigmoid(decoded_64_logits)


class EncoderDecoder_TrustMark(nn.Module):
    """完全匹配 Encoder_MP_Decoder 的调用接口。"""

    def __init__(
        self,
        H,
        W,
        message_length,
        noise_layers,
        return_logits=False,
    ):
        super().__init__()

        if (H, W) != (256, 256):
            raise ValueError(
                f"TrustMark evaluation requires 256x256, got {H}x{W}"
            )

        if message_length != 64:
            raise ValueError(
                f"TrustMark adapter requires message_length=64, "
                f"got {message_length}"
            )

        # 保持官方Q模型内部100-bit网络结构
        tm = TrustMark(
            model_type="Q",
            secret_len=100,
            use_ECC=False,
            device="cpu",
            loadRemover=False,
            loadBBoxDetector=False,
            verbose=True,
        )

        self.encoder = TrustMarkEncoder(tm.encoder)
        self.noise = Noise(noise_layers)
        self.decoder = TrustMarkDecoder(
            tm.decoder,
            return_logits=return_logits,
        )

    def train(self, mode=True):
        super().train(mode)

        # Noise可跟随外部状态，但官方编码器和解码器始终eval
        self.encoder.eval()
        self.decoder.eval()

        return self

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_image = self.noise([encoded_image, image])
        decoded_message = self.decoder(noised_image)

        return encoded_image, noised_image, decoded_message


# 如果模型工厂统一导入EncoderDecoder
# EncoderDecoder = EncoderDecoder_TrustMark
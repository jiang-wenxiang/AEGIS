"""Core reproduction of Cao et al., ESWA 2024 (article 122062).

The paper learns a *universal RGB watermark mask* from a grayscale
watermark image.  The same mask can then be added to arbitrary cover images.
This module follows the paper's encoder -> noise -> decoder organization and
its dense 32-channel convolutional blocks with DCT channel attention.

The learned real screen-shooting simulator from the paper is not exactly
reproducible from the publication (its complete layer configuration and
weights are not reported).  ``ScreenShootingSurrogate`` is therefore an
explicit, differentiable approximation and can be replaced by a calibrated
simulator when paired screen/camera data are available.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _dct_matrix(n: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Return the orthonormal DCT-II transform matrix."""
    x = torch.arange(n, dtype=dtype).view(1, n)
    k = torch.arange(n, dtype=dtype).view(n, 1)
    matrix = torch.cos(math.pi / n * (x + 0.5) * k)
    matrix[0] *= math.sqrt(1.0 / n)
    matrix[1:] *= math.sqrt(2.0 / n)
    return matrix


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DCTChannelAttention(nn.Module):
    """DCT-domain channel attention reconstructed from Fig. 3 and Eqs. 3-8.

    Each channel is reduced to a 7x7 DCT map (49 frequency components).  A
    shared linear projection compresses those components to one descriptor
    per channel.  A second fully connected layer produces sigmoid channel
    weights, which are multiplied with the original feature map.
    """

    def __init__(self, channels: int, dct_size: int = 7):
        super().__init__()
        self.channels = channels
        self.dct_size = dct_size
        self.pool = nn.AdaptiveAvgPool2d((dct_size, dct_size))
        self.frequency_projection = nn.Linear(dct_size * dct_size, 1)
        self.channel_projection = nn.Linear(channels, channels)
        self.register_buffer("dct", _dct_matrix(dct_size), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected BCHW with C={self.channels}, got {tuple(x.shape)}"
            )
        pooled = self.pool(x)
        dct = self.dct.to(dtype=pooled.dtype, device=pooled.device)
        coefficients = torch.matmul(torch.matmul(dct, pooled), dct.t())
        descriptors = F.relu(
            self.frequency_projection(coefficients.flatten(2)).squeeze(-1),
            inplace=False,
        )
        weights = torch.sigmoid(self.channel_projection(descriptors))
        return x * weights[:, :, None, None]


class UniversalMaskEncoder(nn.Module):
    """Paper Table 1 encoder: 1->32, three dense 32-channel blocks, 128->3."""

    def __init__(self, mask_scale: float = 1.0):
        super().__init__()
        self.mask_scale = float(mask_scale)
        self.stem = ConvBNReLU(1, 32)
        self.stem_attention = DCTChannelAttention(32)
        self.dense_blocks = nn.ModuleList(
            [ConvBNReLU(32 * (i + 1), 32) for i in range(3)]
        )
        self.attentions = nn.ModuleList([DCTChannelAttention(32) for _ in range(3)])
        # The paper lists ReLU after this 128->3 convolution.
        self.to_mask = nn.Sequential(
            nn.Conv2d(128, 3, 3, padding=1, bias=True),
        )

    def forward(self, watermark: Tensor) -> Tensor:
        if watermark.ndim != 4 or watermark.shape[1] != 1:
            raise ValueError("watermark must have shape [B, 1, H, W]")
        features = [self.stem_attention(self.stem(watermark))]
        for block, attention in zip(self.dense_blocks, self.attentions):
            new_feature = attention(block(torch.cat(features, dim=1)))
            features.append(new_feature)
        return self.mask_scale * self.to_mask(torch.cat(features, dim=1))


class SpatialTransformer(nn.Module):
    """Compact STN used before the decoder, as specified but not detailed.

    The last affine layer is initialized to identity, so this component is a
    no-op before training.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )
        nn.init.zeros_(self.regressor[-1].weight)
        with torch.no_grad():
            self.regressor[-1].bias.copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )

    def forward(self, x: Tensor) -> Tensor:
        theta = self.regressor(self.localization(x)).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=False
        )


class WatermarkDecoder(nn.Module):
    """Paper Table 1 decoder: STN, 3->32, four dense blocks, 160->1."""

    def __init__(self, use_stn: bool = True):
        super().__init__()
        self.stn = SpatialTransformer(3) if use_stn else nn.Identity()
        self.stem = ConvBNReLU(3, 32)
        self.stem_attention = DCTChannelAttention(32)
        self.dense_blocks = nn.ModuleList(
            [ConvBNReLU(32 * (i + 1), 32) for i in range(4)]
        )
        self.attentions = nn.ModuleList([DCTChannelAttention(32) for _ in range(4)])
        self.to_watermark = nn.Sequential(
            nn.Conv2d(160, 1, 3, padding=1, bias=True),
        )

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        x = self.stn(image)
        features = [self.stem_attention(self.stem(x))]
        for block, attention in zip(self.dense_blocks, self.attentions):
            new_feature = attention(block(torch.cat(features, dim=1)))
            features.append(new_feature)
        return self.to_watermark(torch.cat(features, dim=1))


def _random_affine(x: Tensor) -> Tensor:
    batch = x.shape[0]
    angle = (torch.rand(batch, device=x.device, dtype=x.dtype) - 0.5) * 20.0
    scale = 0.9 + torch.rand(batch, device=x.device, dtype=x.dtype) * 0.2
    tx = (torch.rand(batch, device=x.device, dtype=x.dtype) - 0.5) * 0.12
    ty = (torch.rand(batch, device=x.device, dtype=x.dtype) - 0.5) * 0.12
    radians = angle * math.pi / 180.0
    cos, sin = torch.cos(radians) * scale, torch.sin(radians) * scale
    theta = torch.zeros(batch, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


def _gaussian_blur(x: Tensor, sigma: float = 1.2, kernel_size: int = 5) -> Tensor:
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype)
    coords = coords - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)
    weight = kernel.expand(x.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(x, weight, padding=kernel_size // 2, groups=x.shape[1])


class DifferentiableJPEG(nn.Module):
    """Block-DCT JPEG approximation with straight-through quantization."""

    def __init__(self, quality: int = 70):
        super().__init__()
        self.quality = int(quality)
        self.register_buffer("dct8", _dct_matrix(8), persistent=False)
        q = torch.tensor(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("base_qtable", q, persistent=False)

    def _qtable(self, quality: int, x: Tensor) -> Tensor:
        quality = max(1, min(100, int(quality)))
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        table = torch.floor((self.base_qtable * scale + 50) / 100).clamp(1, 255)
        return table.to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor, quality: Optional[int] = None) -> Tensor:
        quality = self.quality if quality is None else quality
        b, c, h, w = x.shape
        pad_h, pad_w = (-h) % 8, (-w) % 8
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        hp, wp = padded.shape[-2:]
        blocks = F.unfold(padded * 255.0 - 128.0, 8, stride=8)
        blocks = blocks.transpose(1, 2).reshape(b, -1, c, 8, 8)
        dct = self.dct8.to(device=x.device, dtype=x.dtype)
        coeff = torch.matmul(torch.matmul(dct, blocks), dct.t())
        qtable = self._qtable(quality, x)
        scaled = coeff / qtable
        quantized = scaled + (torch.round(scaled) - scaled).detach()
        restored = torch.matmul(torch.matmul(dct.t(), quantized * qtable), dct)
        restored = restored.reshape(b, -1, c * 64).transpose(1, 2)
        image = F.fold(restored, (hp, wp), 8, stride=8)
        return ((image + 128.0) / 255.0)[:, :, :h, :w].clamp(0.0, 1.0)


class ScreenShootingSurrogate(nn.Module):
    """Differentiable physical-channel approximation, not the paper's CNN."""

    def forward(self, x: Tensor) -> Tensor:
        y = _random_affine(x)
        gamma = float(torch.empty(1).uniform_(0.8, 1.25))
        gain = float(torch.empty(1).uniform_(0.85, 1.15))
        y = (y.clamp_min(1e-6).pow(gamma) * gain).clamp(0.0, 1.0)
        y = _gaussian_blur(y, sigma=float(torch.empty(1).uniform_(0.6, 1.4)))
        # A weak sinusoidal pattern approximates display/camera interference.
        h, w = y.shape[-2:]
        xx = torch.linspace(0, 1, w, device=y.device, dtype=y.dtype)[None, None, None]
        phase = float(torch.empty(1).uniform_(0, 2 * math.pi))
        frequency = float(torch.empty(1).uniform_(25, 55))
        moire = 0.012 * torch.sin(2 * math.pi * frequency * xx + phase)
        y = y + moire
        return (y + torch.randn_like(y) * 0.015).clamp(0.0, 1.0)


class PaperNoiseLayer(nn.Module):
    """Random single-distortion layer matching the paper's training policy."""

    MODES: Tuple[str, ...] = (
        "identity",
        "affine",
        "blur",
        "gaussian_noise",
        "cropout",
        "jpeg",
        "sharpen",
        "screen",
    )

    def __init__(self, modes: Optional[Sequence[str]] = None):
        super().__init__()
        self.modes = tuple(modes or self.MODES)
        unknown = set(self.modes) - set(self.MODES)
        if unknown:
            raise ValueError(f"Unknown noise modes: {sorted(unknown)}")
        self.jpeg = DifferentiableJPEG(quality=70)
        self.screen = ScreenShootingSurrogate()

    def forward(self, image: Tensor, mode: Optional[str] = None) -> Tensor:
        mode = random.choice(self.modes) if mode is None else mode
        if mode == "identity":
            return image
        if mode == "affine":
            return _random_affine(image)
        if mode == "blur":
            return _gaussian_blur(image, sigma=float(torch.empty(1).uniform_(0.6, 1.8)))
        if mode == "gaussian_noise":
            sigma = float(torch.empty(1).uniform_(0.0, 0.2))
            return (image + torch.randn_like(image) * sigma).clamp(0.0, 1.0)
        if mode == "cropout":
            _, _, h, w = image.shape
            rh, rw = max(1, int(h * 0.2)), max(1, int(w * 0.2))
            top = random.randint(0, h - rh)
            left = random.randint(0, w - rw)
            mask = torch.ones_like(image)
            mask[:, :, top : top + rh, left : left + rw] = 0
            return image * mask
        if mode == "jpeg":
            return self.jpeg(image, quality=random.randint(50, 90))
        if mode == "sharpen":
            blurred = _gaussian_blur(image, sigma=1.0)
            return (image + 0.8 * (image - blurred)).clamp(0.0, 1.0)
        if mode == "screen":
            return self.screen(image)
        raise ValueError(f"Unknown noise mode: {mode}")


class WatermarkCritic(nn.Module):
    """Small WGAN critic corresponding to the paper's discriminator role."""

    def __init__(self):
        super().__init__()
        layers = []
        channels = [3, 32, 64, 128]
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(in_c, out_c, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.AvgPool2d(2),
                ]
            )
        self.features = nn.Sequential(*layers)
        self.score = nn.Linear(128, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x).mean(dim=(-2, -1))
        return self.score(x).squeeze(1)


class BitImageAdapter(nn.Module):
    """Deterministic bridge between an L-bit message and a watermark image.

    For the comparison setting ``L=64`` and ``H=W=256``, the message is first
    reshaped to an 8x8 grid.  Every bit is then repeated to a 32x32 constant
    block.  Decoding applies the exact inverse grouping operation by averaging
    each 32x32 block and flattening the resulting 8x8 grid in row-major order.

    No learnable parameters are introduced, so this adapter changes only the
    representation expected by the paper's image-watermark architecture.
    """

    def __init__(
        self,
        message_length: int = 64,
        height: int = 256,
        width: int = 256,
    ):
        super().__init__()
        grid_size = math.isqrt(message_length)
        if grid_size * grid_size != message_length:
            raise ValueError("message_length must be a perfect square")
        if height % grid_size or width % grid_size:
            raise ValueError(
                "height and width must be divisible by sqrt(message_length)"
            )
        self.message_length = int(message_length)
        self.height = int(height)
        self.width = int(width)
        self.grid_size = grid_size
        self.block_height = height // grid_size
        self.block_width = width // grid_size

    def bits_to_image(self, bits: Tensor) -> Tensor:
        """Expand ``[B, L]`` binary/soft bits to ``[B, 1, H, W]``."""
        if bits.ndim == 3 and bits.shape[1:] == (1, self.message_length):
            bits = bits[:, 0]
        if bits.ndim != 2 or bits.shape[1] != self.message_length:
            raise ValueError(
                f"bits must have shape [B, {self.message_length}] or "
                f"[B, 1, {self.message_length}], got {tuple(bits.shape)}"
            )
        grid = bits.reshape(-1, 1, self.grid_size, self.grid_size)
        return grid.repeat_interleave(self.block_height, dim=2).repeat_interleave(
            self.block_width, dim=3
        )

    def image_to_bits(self, watermark_image: Tensor) -> Tensor:
        """Average non-overlapping blocks and return soft ``[B, L]`` bits."""
        expected = (1, self.height, self.width)
        if watermark_image.ndim != 4 or tuple(watermark_image.shape[1:]) != expected:
            raise ValueError(
                f"watermark_image must have shape [B, {expected[0]}, "
                f"{expected[1]}, {expected[2]}], got {tuple(watermark_image.shape)}"
            )
        grid = F.avg_pool2d(
            watermark_image,
            kernel_size=(self.block_height, self.block_width),
            stride=(self.block_height, self.block_width),
        )
        return grid.flatten(1)

    def hard_decode(self, watermark_image: Tensor, threshold: float = 0.5) -> Tensor:
        """Return binary values after block averaging and thresholding."""
        return (self.image_to_bits(watermark_image) >= threshold).to(
            watermark_image.dtype
        )


class UniversalScreenShootingModel(nn.Module):
    """Encoder-noise-decoder wrapper analogous to Encoder_MP_Decoder.py."""

    def __init__(
        self,
        height: int = 128,
        width: int = 128,
        noise_modes: Optional[Sequence[str]] = None,
        mask_scale: float = 1.0,
        use_stn: bool = True,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.encoder = UniversalMaskEncoder(mask_scale=mask_scale)
        self.noise = PaperNoiseLayer(noise_modes)
        self.decoder = WatermarkDecoder(use_stn=use_stn)

    def forward(
        self,
        cover_image: Tensor,
        watermark_image: Tensor,
        noise_mode: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if cover_image.ndim != 4 or cover_image.shape[1] != 3:
            raise ValueError("cover_image must have shape [B, 3, H, W]")
        if cover_image.shape[-2:] != watermark_image.shape[-2:]:
            raise ValueError("cover and watermark spatial sizes must match")
        mask = self.encoder(watermark_image)
        watermarked = (cover_image + mask).clamp(0.0, 1.0)
        distorted = self.noise(watermarked, mode=noise_mode)
        recovered = self.decoder(distorted)
        return watermarked, distorted, recovered

    def generate_mask(self, watermark_image: Tensor) -> Tensor:
        """Generate the universal mask once for reuse on arbitrary covers."""
        return self.encoder(watermark_image)


class UniversalBitScreenShootingModel(UniversalScreenShootingModel):
    """64-bit/256px comparison wrapper preserving the paper's core model.

    Its public forward signature mirrors common bit-watermark baselines:

    ``watermarked, distorted, decoded_bits = model(cover, message)``

    Use :meth:`forward_with_images` when the expanded target and recovered
    watermark images are also needed for the paper's image-level NCC loss.
    """

    def __init__(
        self,
        height: int = 256,
        width: int = 256,
        message_length: int = 64,
        noise_modes: Optional[Sequence[str]] = None,
        mask_scale: float = 1.0,
        use_stn: bool = True,
    ):
        super().__init__(
            height=height,
            width=width,
            noise_modes=noise_modes,
            mask_scale=mask_scale,
            use_stn=use_stn,
        )
        self.message_length = int(message_length)
        self.bit_adapter = BitImageAdapter(message_length, height, width)

    def forward(
        self,
        cover_image: Tensor,
        message: Tensor,
        noise_mode: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        watermark_image = self.bit_adapter.bits_to_image(message).to(
            dtype=cover_image.dtype, device=cover_image.device
        )
        watermarked, distorted, recovered_image = super().forward(
            cover_image, watermark_image, noise_mode
        )
        decoded_bits = self.bit_adapter.image_to_bits(recovered_image)
        return watermarked, distorted, decoded_bits

    def forward_with_images(
        self,
        cover_image: Tensor,
        message: Tensor,
        noise_mode: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        watermark_image = self.bit_adapter.bits_to_image(message).to(
            dtype=cover_image.dtype, device=cover_image.device
        )
        watermarked, distorted, recovered_image = UniversalScreenShootingModel.forward(
            self, cover_image, watermark_image, noise_mode
        )
        decoded_bits = self.bit_adapter.image_to_bits(recovered_image)
        return (
            watermarked,
            distorted,
            decoded_bits,
            watermark_image,
            recovered_image,
        )

    def decode_hard(self, distorted_image: Tensor, threshold: float = 0.5) -> Tensor:
        recovered_image = self.decoder(distorted_image)
        return self.bit_adapter.hard_decode(recovered_image, threshold)


def normalized_cross_correlation(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    a = a.flatten(1)
    b = b.flatten(1)
    numerator = (a * b).sum(dim=1)
    denominator = torch.sqrt((a.square().sum(1) * b.square().sum(1)).clamp_min(eps))
    return (numerator / denominator).mean()


def generator_losses(
    cover: Tensor,
    watermarked: Tensor,
    watermark: Tensor,
    recovered: Tensor,
    critic: Optional[nn.Module] = None,
) -> Dict[str, Tensor]:
    """Paper-inspired generator objectives (MSE, 1-NCC, optional WGAN term)."""
    losses: Dict[str, Tensor] = {
        "fidelity_mse": F.mse_loss(watermarked, cover),
        "recovery_ncc": 1.0 - normalized_cross_correlation(recovered, watermark),
    }
    if critic is not None:
        losses["generator_adversarial"] = -critic(watermarked).mean()
    losses["total"] = sum(losses.values())
    return losses


def critic_loss(critic: nn.Module, cover: Tensor, watermarked: Tensor) -> Tensor:
    """Wasserstein critic loss; paper clips critic weights to [-0.01, 0.01]."""
    return critic(watermarked.detach()).mean() - critic(cover).mean()


def bit_comparison_metrics(decoded_bits: Tensor, target_bits: Tensor) -> Dict[str, Tensor]:
    """Differentiable bit loss plus BER/accuracy for the comparison protocol."""
    if target_bits.ndim == 3 and target_bits.shape[1] == 1:
        target_bits = target_bits[:, 0]
    target_bits = target_bits.to(dtype=decoded_bits.dtype, device=decoded_bits.device)
    hard_bits = (decoded_bits >= 0.5).to(decoded_bits.dtype)
    ber = (hard_bits != target_bits).to(decoded_bits.dtype).mean()
    return {
        "bit_mse": F.mse_loss(decoded_bits, target_bits),
        "bit_error_rate": ber,
        "bit_accuracy": 1.0 - ber,
    }


def clip_critic_weights(critic: nn.Module, limit: float = 0.01) -> None:
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.clamp_(-limit, limit)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = UniversalScreenShootingModel(noise_modes=["identity"])
    cover = torch.rand(2, 3, 128, 128)
    watermark = torch.rand(2, 1, 128, 128)
    watermarked, distorted, recovered = model(cover, watermark, "identity")
    print("watermarked", tuple(watermarked.shape))
    print("distorted", tuple(distorted.shape))
    print("recovered", tuple(recovered.shape))
    print({k: float(v) for k, v in generator_losses(
        cover, watermarked, watermark, recovered
    ).items()})

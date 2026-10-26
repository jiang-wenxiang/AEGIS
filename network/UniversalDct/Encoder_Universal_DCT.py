"""Framework-compatible encoder for the Cao et al. DCT-attention baseline."""

from __future__ import annotations

from torch import Tensor, nn

try:  # Package import used by the existing training project.
    from .universal_dct_watermark import BitImageAdapter, UniversalMaskEncoder
except ImportError:  # Allows standalone architecture tests in this folder.
    from universal_dct_watermark import BitImageAdapter, UniversalMaskEncoder


class Encoder_Universal_DCT(nn.Module):
    """Match ``Encoder_MP(H, W, message_length)`` and ``forward(image, message)``.

    The bit-to-image conversion is internal.  For the default comparison,
    64 bits become an 8x8 grid and each grid value is repeated over a 32x32
    block at 256x256 resolution.  The paper encoder converts that image into
    a universal RGB mask, which is added to the input cover image.
    """

    def __init__(self, H: int, W: int, message_length: int):
        super().__init__()
        self.H = int(H)
        self.W = int(W)
        self.message_length = int(message_length)
        self.message_adapter = BitImageAdapter(message_length, H, W)
        self.mask_encoder = UniversalMaskEncoder(mask_scale=1.0)

    def forward(self, image: Tensor, message: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        if tuple(image.shape[-2:]) != (self.H, self.W):
            raise ValueError(
                f"Expected image size {(self.H, self.W)}, got {tuple(image.shape[-2:])}"
            )
        watermark_image = self.message_adapter.bits_to_image(message).to(
            device=image.device, dtype=image.dtype
        )
        watermark_mask = self.mask_encoder(watermark_image)
        # The source paper defines I_w = I_c + W_e.  Deliberately do not clamp
        # here: the existing framework/noise/loss pipeline owns normalization.
        return image + watermark_mask


# Optional alias if the project's model factory expects a generic name.
Encoder = Encoder_Universal_DCT

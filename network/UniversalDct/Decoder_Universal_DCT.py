"""Framework-compatible decoder for the Cao et al. DCT-attention baseline."""

from __future__ import annotations

from torch import Tensor, nn

try:  # Package import used by the existing training project.
    from .universal_dct_watermark import BitImageAdapter, WatermarkDecoder
except ImportError:  # Allows standalone architecture tests in this folder.
    from universal_dct_watermark import BitImageAdapter, WatermarkDecoder


class Decoder_Universal_DCT(nn.Module):
    """Match ``Decoder(H, W, message_length)`` and return ``[B, L]``.

    The paper decoder first recovers a grayscale watermark image.  This class
    hides that image-level output from the training framework and averages the
    predefined repeated blocks back into soft bit values.
    """

    def __init__(self, H: int, W: int, message_length: int):
        super().__init__()
        self.H = int(H)
        self.W = int(W)
        self.message_length = int(message_length)
        self.image_decoder = WatermarkDecoder(use_stn=True)
        self.message_adapter = BitImageAdapter(message_length, H, W)

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        if tuple(image.shape[-2:]) != (self.H, self.W):
            raise ValueError(
                f"Expected image size {(self.H, self.W)}, got {tuple(image.shape[-2:])}"
            )
        recovered_watermark_image = self.image_decoder(image)
        return self.message_adapter.image_to_bits(recovered_watermark_image)


# Optional alias if the project's model factory expects a generic name.
Decoder = Decoder_Universal_DCT

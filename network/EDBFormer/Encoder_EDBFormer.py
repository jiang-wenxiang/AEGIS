from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

try:
    from .edbformer_imports import load_official_classes
except ImportError:  # Supports copying these files directly into network/.
    from edbformer_imports import load_official_classes


class Encoder_EDBFormer(nn.Module):
    """Framework adapter around the released EDBFormer encoder.

    Inputs:
        image:   [B, 3, 256, 256], normalized to [-1, 1]
        message: [B, message_length], binary values represented as floats
    Output:
        encoded image with the same shape as ``image``
    """

    def __init__(self, H: int, W: int, message_length: int):
        super().__init__()
        if (H, W) != (256, 256):
            raise ValueError(
                "The recoverable source version released with EDBFormer fixes "
                "the post-DCT grid at 32x32, so this adapter requires 256x256 "
                "inputs (256 / 8 = 32)."
            )
        if message_length <= 0:
            raise ValueError("message_length must be positive")

        OfficialEncoder, _ = load_official_classes()
        flags = SimpleNamespace(image_size=H, redundant_length=message_length)
        self.core = OfficialEncoder(flags)
        self.H = H
        self.W = W
        self.message_length = message_length

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, self.H, self.W):
            raise ValueError(
                f"image must have shape [B, 3, {self.H}, {self.W}], "
                f"but got {tuple(image.shape)}"
            )
        if message.ndim != 2 or message.shape != (
            image.shape[0],
            self.message_length,
        ):
            raise ValueError(
                f"message must have shape [B, {self.message_length}], "
                f"but got {tuple(message.shape)}"
            )
        return self.core(image, message.float())


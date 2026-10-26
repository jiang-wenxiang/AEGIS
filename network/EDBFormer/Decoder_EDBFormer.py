from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

try:
    from .edbformer_imports import load_official_classes
except ImportError:  # Supports copying these files directly into network/.
    from edbformer_imports import load_official_classes


class Decoder_EDBFormer(nn.Module):
    """Framework adapter around the released EDBFormer decoder."""

    def __init__(self, H: int, W: int, message_length: int):
        super().__init__()
        if (H, W) != (256, 256):
            raise ValueError(
                "This released EDBFormer implementation requires 256x256 input."
            )
        if message_length <= 0:
            raise ValueError("message_length must be positive")

        _, OfficialDecoder = load_official_classes()
        flags = SimpleNamespace(image_size=H, redundant_length=message_length)
        self.core = OfficialDecoder(flags)
        self.H = H
        self.W = W
        self.message_length = message_length

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, self.H, self.W):
            raise ValueError(
                f"image must have shape [B, 3, {self.H}, {self.W}], "
                f"but got {tuple(image.shape)}"
            )
        return self.core(image)


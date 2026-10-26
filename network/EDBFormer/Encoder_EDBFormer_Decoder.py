from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .Decoder_EDBFormer import Decoder_EDBFormer
    from .Encoder_EDBFormer import Encoder_EDBFormer
    from ..Noise import Noise
except ImportError:  # Supports the project's existing flat network package.
    from Decoder_EDBFormer import Decoder_EDBFormer
    from Encoder_EDBFormer import Encoder_EDBFormer
    from Noise import Noise


class EncoderDecoder_EDBFormer(nn.Module):
    """EDBFormer adapted to the project's encoder-noise-decoder contract."""

    def __init__(self, H: int, W: int, message_length: int, noise_layers):
        super().__init__()
        self.encoder = Encoder_EDBFormer(H, W, message_length)
        self.noise = Noise(noise_layers)
        self.decoder = Decoder_EDBFormer(H, W, message_length)

    def forward(self, image: torch.Tensor, message: torch.Tensor):
        encoded_image = self.encoder(image, message)
        noised_image = self.noise([encoded_image, image])
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message


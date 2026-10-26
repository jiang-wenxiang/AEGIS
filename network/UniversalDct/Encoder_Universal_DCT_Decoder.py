"""Drop-in encoder-noise-decoder wrapper for the existing framework.

This file intentionally mirrors ``Encoder_MP_Decoder.py``.  In particular,
the existing ``Noise`` implementation is imported and called unchanged.
"""

from torch import nn

# try:  # Normal use from inside the existing project package.
from .Encoder_Universal_DCT import Encoder_Universal_DCT
from .Decoder_Universal_DCT import Decoder_Universal_DCT
from ..Noise import Noise
# except ImportError:  # Standalone import fallback; still uses the project's Noise.
#     from Encoder_Universal_DCT import Encoder_Universal_DCT
#     from Decoder_Universal_DCT import Decoder_Universal_DCT
#     from ..Noise import Noise


class EncoderDecoder_Universal_DCT(nn.Module):
    """A Sequential of Encoder_Universal_DCT - Noise - Decoder_Universal_DCT."""

    def __init__(self, H, W, message_length, noise_layers):
        super().__init__()
        self.encoder = Encoder_Universal_DCT(H, W, message_length)
        self.noise = Noise(noise_layers)
        self.decoder = Decoder_Universal_DCT(H, W, message_length)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_image = self.noise([encoded_image, image])
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message


# Use this alias if the existing training code imports `EncoderDecoder` from
# each model-specific wrapper module.
# EncoderDecoder = EncoderDecoder_Universal_DCT

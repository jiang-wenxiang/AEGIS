import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .options import HiDDenConfiguration
from ..Noise import Noise
from .MultiDomainDecoder import MultiDomainDecoder
from .MultiDomainEncoder import MultiDomainEncoder
from .ResMultiDomainEncoder import ResMultiDomainEncoder
from .AttentionResMultiDomainEncoder import AttentionResMultiDomainEncoder


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noise_layers):

        super(EncoderDecoder, self).__init__()
        if config.using_new_model:
            if config.using_res_model:
                if config.using_attention:
                    if config.using_sp_attention:
                        if config.dl_res_attention:
                            self.encoder = AttentionResMultiDomainEncoder(config, sp_attention=True, dl_res_attention=True)
                        else:
                            self.encoder = AttentionResMultiDomainEncoder(config, sp_attention=True)
                    else:
                        self.encoder = AttentionResMultiDomainEncoder(config, sp_attention=False)
                else:
                    self.encoder = ResMultiDomainEncoder(config)
            else:
                self.encoder = MultiDomainEncoder(config)
            self.noiser = Noise(noise_layers)
            self.decoder = MultiDomainDecoder(config)
        else:
            self.encoder = Encoder(config)
            self.noiser = Noise(noise_layers)
            self.decoder = Decoder(config)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_image = self.noiser([encoded_image, image])
        # noised_image = noised_and_cover[0]
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message

    # encoded_images, noised_images, decoded_messages

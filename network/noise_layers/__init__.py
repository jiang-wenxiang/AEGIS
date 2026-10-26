import random


def get_random_float(float_range: [float]):
	return random.random() * (float_range[1] - float_range[0]) + float_range[0]


def get_random_int(int_range: [int]):
	return random.randint(int_range[0], int_range[1])


from .identity import Identity
from .crop import Crop, Cropout, Dropout
from .gaussian_noise import GN
from .middle_filter import MF
from .gaussian_filter import GF
from .salt_pepper_noise import SP
from .jpeg import Jpeg, JpegSS, JpegMask, JpegTest
from .combined import Combined
from .rotation import Rotation
from .warp import Warp
from .color_jitter import ColorJitter
from .scaling import Scaling
from .quantization import Quantization
from .sd_vae import SDVAE
from .resizing import Resizing
from .sharpening import Sharpening
from .jpeg2000 import Jpeg2000Sim


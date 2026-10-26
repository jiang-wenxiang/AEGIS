import random


def get_random_float(float_range: [float]):
	return random.random() * (float_range[1] - float_range[0]) + float_range[0]


def get_random_int(int_range: [int]):
	return random.randint(int_range[0], int_range[1])


from .identity import Identity
# from .crop import Crop, Cropout, Dropout
from .crop import Crop as C
from .crop import Cropout as CO
from .crop import Dropout as DO
from .gaussian_noise import GN
from .middle_filter import MF as MB
from .gaussian_filter import GF as GB
from .gaussian_blur import Gaussian_blur2 as GB1
from .salt_pepper_noise import SP
from .jpeg import Jpeg, JpegSS, JpegSS_diffround, JpegMask, JpegMask_diffround, JpegTest, Jpeg_compression, RandomJpegSS, RandomJpegMask, RandomJpeg
from .combined import Combined
from .osn import Osn
from .DiffJPEG_PRIS import DiffJPEG as Jpeg_PRIS

from .rotation_stretching import Rotation as RN
from .rotation_stretching import Stretching as SG
from .rotation_stretching import Rotation_Stretching as RNSG
from .rotation_stretching import Stretching_Rotation as SGRN



# from .gaussian_blur import Gaussian_blur as GB

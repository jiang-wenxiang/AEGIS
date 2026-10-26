from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import sys

# 将当前 network.Trustmark 包注册为官方配置需要的 trustmark
sys.modules["trustmark"] = sys.modules[__name__]

from .trustmark import TrustMark

__all__ = ["TrustMark"]

from .trustmark import TrustMark

import torch.nn as nn
from kornia.filters import gaussian_blur2d


class GF(nn.Module):

	def __init__(self, sigma, kernel=7):
		super(GF, self).__init__()
		self.max_sigma = sigma  # 改名
		self.kernel = kernel
		self.intensity = 1.0

	def set_intensity(self, intensity):
		self.intensity = intensity

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		if self.intensity == 0:
			return image

		# 防止 sigma 完全等于 0 导致 kornia 报错
		cur_sigma = max(1e-5, self.max_sigma * self.intensity)
		return gaussian_blur2d(image, (self.kernel, self.kernel), (cur_sigma, cur_sigma))

import numpy as np
import torch
import torch.nn as nn


class GN(nn.Module):

	def __init__(self, var, mean=0):
		super(GN, self).__init__()
		self.max_var = var
		self.mean = mean
		self.intensity = 1.0  # 默认满载

	def set_intensity(self, intensity):
		self.intensity = intensity

	# def gaussian_noise(self, image, mean, var):
	# 	noise = torch.Tensor(np.random.normal(mean, var ** 0.5, image.shape)).to(image.device)
	# 	out = image + noise
	# 	return out

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		# 强度为0时直接返回，节省算力
		if self.intensity == 0:
			return image

		cur_var = self.max_var * self.intensity
		# 优化：直接使用 torch.randn_like 避免 NumPy 到 Tensor 的转换开销
		noise = torch.randn_like(image) * (cur_var ** 0.5) + self.mean
		return (image + noise).clamp(-1.0, 1.0)

		# return self.gaussian_noise(image, self.mean, self.var)

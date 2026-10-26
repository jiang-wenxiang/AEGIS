import torch
import torch.nn as nn


class SP(nn.Module):

	def __init__(self, prob):
		super(SP, self).__init__()
		self.prob = prob
		self.intensity = 1.0

	def set_intensity(self, intensity):
		self.intensity = intensity

	def sp_noise(self, image, prob):
		# 强度为0时直接返回，节省算力
		if prob == 0:
			return image

		prob_zero = prob / 2
		prob_one = 1 - prob_zero
		rdn = torch.rand(image.shape).to(image.device)

		output = torch.where(rdn > prob_one, torch.zeros_like(image).to(image.device), image)
		output = torch.where(rdn < prob_zero, torch.ones_like(output).to(output.device), output)

		return output

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		return self.sp_noise(image, self.prob * self.intensity)

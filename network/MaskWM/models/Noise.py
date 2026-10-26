from torch import nn
from .noise_layers import *


class Noise(nn.Module):

	def __init__(self, layers):
		super(Noise, self).__init__()
		for i in range(len(layers)):
			layers[i] = eval(layers[i])

	def forward(self, image, mask):
		results = self.noise(image, mask)
		return results

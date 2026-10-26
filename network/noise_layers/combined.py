from . import Identity
import torch.nn as nn
from . import get_random_int


class Combined(nn.Module):

	def __init__(self, list=None):
		super(Combined, self).__init__()
		if list is None:
			list = [Identity()]
		self.list = list

	# 新增：递归下发强度指令给列表中的每一个实体噪声层
	def set_intensity(self, intensity):
		for layer in self.list:
			if hasattr(layer, 'set_intensity'):
				layer.set_intensity(intensity)

	def forward(self, image_and_cover):
		id = get_random_int([0, len(self.list) - 1])
		return self.list[id](image_and_cover)

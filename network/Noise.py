from . import *
from .noise_layers import *


class Noise(nn.Module):

	def __init__(self, layers):
		super(Noise, self).__init__()
		for i in range(len(layers)):
			layers[i] = eval(layers[i])

		# 【关键修改 1】: 放弃 nn.Sequential，改用 nn.ModuleList
		# 这样网络依然能识别这些子层并保存权重，但不会自动串联执行它们
		self.noise_module_list = nn.ModuleList(layers)

	# 新增：递归下发强度指令
	def set_intensity(self, intensity):
		for layer in self.noise_module_list:
			if hasattr(layer, 'set_intensity'):
				layer.set_intensity(intensity)

	def forward(self, image_and_cover):
		# 拆包获取初始的水印图和干净的原图
		image, cover_image = image_and_cover

		# 【关键修改 2】: 手动控制接力赛流水线
		for layer in self.noise_module_list:
			# 每次给下一层传参时，都把"最新受损的 image"和"干净的 cover_image"重新打包
			image = layer([image, cover_image])

		# 所有层遍历完后，返回最终伤痕累累的图像
		return image

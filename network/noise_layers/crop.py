import torch
import torch.nn as nn
import numpy as np

def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
	image_height = image_shape[2]
	image_width = image_shape[3]

	remaining_height = int(height_ratio * image_height)
	remaining_width = int(width_ratio * image_width)

	if remaining_height == image_height:
		height_start = 0
	else:
		height_start = np.random.randint(0, image_height - remaining_height)

	if remaining_width == image_width:
		width_start = 0
	else:
		width_start = np.random.randint(0, image_width - remaining_width)

	return height_start, height_start + remaining_height, width_start, width_start + remaining_width


class Crop(nn.Module):

	def __init__(self, height_ratio, width_ratio):
		### ratio 越小，噪声强度越大，当 ratio == 1 时，输出原图
		super(Crop, self).__init__()
		self.max_height_ratio = height_ratio  # 改名：保存最大裁剪程度
		self.max_width_ratio = width_ratio
		self.intensity = 1.0  # 新增：默认强度为 1.0 (满载，兼容旧代码)

	# 新增：供外部调用的强度设置接口
	def set_intensity(self, intensity):
		self.intensity = intensity

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		if self.intensity == 0.0:
			return image

		# 新增：动态计算当前的裁剪比例
		# intensity=0时，ratio=1.0；intensity=1时，ratio=max_ratio
		cur_h_ratio = 1.0 - (1.0 - self.max_height_ratio) * self.intensity
		cur_w_ratio = 1.0 - (1.0 - self.max_width_ratio) * self.intensity

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, cur_h_ratio, cur_w_ratio)
		output = cover_image.clone()
		output[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
		return output

class Cropout(nn.Module):

	def __init__(self, height_ratio, width_ratio):
		### ratio 越大，噪声强度越大，当 ratio == 0 时，输出原图
		super(Cropout, self).__init__()
		self.max_height_ratio = height_ratio
		self.max_width_ratio = width_ratio
		self.intensity = 1.0  # 新增：默认强度为 1.0 (满载，兼容旧代码)

	# 新增：供外部调用的强度设置接口
	def set_intensity(self, intensity):
		self.intensity = intensity

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		# 动态计算当前要“挖掉”的比例
		cur_h_ratio = self.max_height_ratio * self.intensity
		cur_w_ratio = self.max_width_ratio * self.intensity

		# 如果 intensity 为 0，不挖洞，直接返回水印图
		if cur_h_ratio == 0 and cur_w_ratio == 0:
			return image

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, cur_h_ratio, cur_w_ratio)
		output = image.clone()
		output[:, :, h_start: h_end, w_start: w_end] = cover_image[:, :, h_start: h_end, w_start: w_end]
		return output

class Dropout(nn.Module):

	def __init__(self, prob):
		### ratio 越大，噪声强度越大，当 ratio == 0 时，输出原图
		super(Dropout, self).__init__()
		self.max_prob = prob
		self.intensity = 1.0  # 新增：默认强度为 1.0 (满载，兼容旧代码)

	# 新增：供外部调用的强度设置接口
	def set_intensity(self, intensity):
		self.intensity = intensity

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		cur_prob = self.max_prob * self.intensity

		if cur_prob == 0:
			return image

		rdn = torch.rand(image.shape).to(image.device)
		output = torch.where(rdn < cur_prob, cover_image, image)
		return output

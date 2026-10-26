# # import torch
# # import torch.nn as nn
# # import numpy as np
# # import math
# # def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
# # 	image_height = image_shape[2]
# # 	image_width = image_shape[3]
# #
# # 	remaining_height = int(height_ratio * image_height)
# # 	remaining_width = int(width_ratio * image_width)
# #
# # 	if remaining_height == image_height:
# # 		height_start = 0
# # 	else:
# # 		height_start = np.random.randint(0, image_height - remaining_height)
# #
# # 	if remaining_width == image_width:
# # 		width_start = 0
# # 	else:
# # 		width_start = np.random.randint(0, image_width - remaining_width)
# #
# # 	return height_start, height_start + remaining_height, width_start, width_start + remaining_width
# #
# # class Crop(nn.Module):
# #
# # 	def __init__(self, ratio):
# # 		super(Crop, self).__init__()
# # 		self.height_ratio = ratio#math.sqrt(ratio)
# # 		self.width_ratio = ratio#math.sqrt(ratio)
# #
# # 	def forward(self, image_and_cover):
# # 		#print(self.height_ratio)
# # 		image, cover_image = image_and_cover
# #
# # 		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
# # 																	 self.width_ratio)
# # 		mask = torch.zeros_like(image)
# # 		mask[:, :, h_start: h_end, w_start: w_end] = 1
# #
# # 		return image * mask
# #
# #
# # # class Crop(nn.Module):
# # #
# # # 	def __init__(self, height_ratio, width_ratio):
# # # 		super(Crop, self).__init__()
# # # 		self.height_ratio = height_ratio
# # # 		self.width_ratio = width_ratio
# # #
# # # 	def forward(self, image_and_cover):
# # #
# # # 		image, cover_image = image_and_cover
# # #
# # # 		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
# # # 																	 self.width_ratio)
# # # 		mask = torch.zeros_like(image)
# # # 		mask[:, :, h_start: h_end, w_start: w_end] = 1
# # #
# # # 		return image * mask
# # # 		# h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio, self.width_ratio)
# # # 		#
# # # 		#
# # # 		# output = image[:, :, h_start: h_end, w_start: w_end]
# # # 		#
# # # 		# return output
# #
# # class RandomCrop(nn.Module):
# # 	def __init__(self, min_ratio=0.7, max_ratio=1, target_size=0, proportional=False):
# # 		super(RandomCrop, self).__init__()
# # 		self.min_ratio = min_ratio
# # 		self.max_ratio = max_ratio
# # 		self.proportional = proportional
# # 		self.target_size = target_size
# #
# # 	def forward(self, image_and_cover):
# # 		self.height_ratio = np.random.randint(self.min_ratio, self.max_ratio) #(np.random.rand() * (self.max_ratio - self.min_ratio) + self.min_ratio)
# # 		self.width_ratio = self.height_ratio #(np.random.rand() * (self.max_ratio - self.min_ratio) + self.min_ratio)
# #
# # 		# image, cover_image = image_and_cover
# # 		#
# # 		# h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, height_ratio, width_ratio)
# # 		# output = image[:, :, h_start: h_end, w_start: w_end]
# # 		#
# # 		# return output
# # 		image, cover_image = image_and_cover
# #
# # 		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
# # 																	 self.width_ratio)
# # 		mask = torch.zeros_like(image)
# # 		mask[:, :, h_start: h_end, w_start: w_end] = 1
# #
# # 		return image * mask
# #
# #
# # class Cropout(nn.Module):
# #
# # 	def __init__(self, height_ratio, width_ratio):
# # 		super(Cropout, self).__init__()
# # 		self.height_ratio = height_ratio
# # 		self.width_ratio = width_ratio
# #
# # 	def forward(self, image_and_cover):
# # 		image, cover_image = image_and_cover
# #
# # 		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
# # 																	 self.width_ratio)
# # 		output = cover_image.clone()
# # 		output[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
# # 		return output
# #
# # class Dropout(nn.Module):
# #
# # 	def __init__(self, prob):
# # 		super(Dropout, self).__init__()
# # 		self.prob = prob
# #
# # 	def forward(self, image_and_cover):
# # 		image, cover_image = image_and_cover
# #
# # 		rdn = torch.rand(image.shape).to(image.device)
# # 		output = torch.where(rdn > self.prob * 1., cover_image, image)
# # 		return output
# #
#
#
# import torch
# import torch.nn as nn
# import numpy as np
# import math
# def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
# 	image_height = image_shape[2]
# 	image_width = image_shape[3]
#
# 	remaining_height = int(height_ratio * image_height)
# 	remaining_width = int(width_ratio * image_width)
#
# 	if remaining_height == image_height:
# 		height_start = 0
# 	else:
# 		height_start = np.random.randint(0, image_height - remaining_height)
#
# 	if remaining_width == image_width:
# 		width_start = 0
# 	else:
# 		width_start = np.random.randint(0, image_width - remaining_width)
#
# 	return height_start, height_start + remaining_height, width_start, width_start + remaining_width
#
# class Crop(nn.Module):
#
# 	def __init__(self, ratio):
# 		super(Crop, self).__init__()
# 		self.height_ratio = 1-ratio#math.sqrt(ratio)
# 		self.width_ratio = 1-ratio#math.sqrt(ratio)
#
# 	def forward(self, image_and_cover):
# 		#print(self.height_ratio)
# 		image, cover_image = image_and_cover
#
# 		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
# 																	 self.width_ratio)
# 		mask = torch.zeros_like(image)
# 		mask[:, :, h_start: h_end, w_start: w_end] = 1
#
# 		return image * mask
#
#
# # class Crop(nn.Module):
# #
# # 	def __init__(self, height_ratio, width_ratio):
# # 		super(Crop, self).__init__()
# # 		self.height_ratio = height_ratio
# # 		self.width_ratio = width_ratio
# #
# # 	def forward(self, image_and_cover):
# #
# # 		image, cover_image = image_and_cover
# #
# # 		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
# # 																	 self.width_ratio)
# # 		mask = torch.zeros_like(image)
# # 		mask[:, :, h_start: h_end, w_start: w_end] = 1
# #
# # 		return image * mask
# # 		# h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio, self.width_ratio)
# # 		#
# # 		#
# # 		# output = image[:, :, h_start: h_end, w_start: w_end]
# # 		#
# # 		# return output
#
# class RandomCrop(nn.Module):
# 	def __init__(self, min_ratio=0.7, max_ratio=1, target_size=0, proportional=False):
# 		super(RandomCrop, self).__init__()
# 		self.min_ratio = min_ratio
# 		self.max_ratio = max_ratio
# 		self.proportional = proportional
# 		self.target_size = target_size
#
# 	def forward(self, image_and_cover):
# 		self.height_ratio = np.random.randint(self.min_ratio, self.max_ratio) #(np.random.rand() * (self.max_ratio - self.min_ratio) + self.min_ratio)
# 		self.width_ratio = self.height_ratio #(np.random.rand() * (self.max_ratio - self.min_ratio) + self.min_ratio)
#
# 		# image, cover_image = image_and_cover
# 		#
# 		# h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, height_ratio, width_ratio)
# 		# output = image[:, :, h_start: h_end, w_start: w_end]
# 		#
# 		# return output
# 		image, cover_image = image_and_cover
#
# 		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
# 																	 self.width_ratio)
# 		mask = torch.zeros_like(image)
# 		mask[:, :, h_start: h_end, w_start: w_end] = 1
#
# 		return image * mask
#
#
# class Cropout(nn.Module):
#
# 	def __init__(self, ratio):
# 		super(Cropout, self).__init__()
# 		self.height_ratio = 1-ratio#height_ratio
# 		self.width_ratio = 1-ratio#width_ratio
#
# 	def forward(self, image_and_cover):
# 		image, cover_image = image_and_cover
#
# 		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
# 																	 self.width_ratio)
# 		output = cover_image.clone()
# 		output[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
# 		return output
#
# class Dropout(nn.Module):
#
# 	def __init__(self, prob):
# 		super(Dropout, self).__init__()
# 		self.prob = 1 - prob
#
# 	def forward(self, image_and_cover):
# 		image, cover_image = image_and_cover
#
# 		rdn = torch.rand(image.shape).to(image.device)
# 		output = torch.where(rdn > self.prob * 1., cover_image, image)
# 		return output
#
#
import torch
import torch.nn as nn
import numpy as np
import math
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

	def __init__(self, ratio):
		super(Crop, self).__init__()
		self.height_ratio = 1-ratio#math.sqrt(ratio)
		self.width_ratio = 1-ratio#math.sqrt(ratio)

	def forward(self, image_and_cover):
		#print(self.height_ratio)
		image, cover_image = image_and_cover

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
		mask = torch.zeros_like(image)
		mask[:, :, h_start: h_end, w_start: w_end] = 1

		return image * mask


# class Crop(nn.Module):
#
# 	def __init__(self, height_ratio, width_ratio):
# 		super(Crop, self).__init__()
# 		self.height_ratio = height_ratio
# 		self.width_ratio = width_ratio
#
# 	def forward(self, image_and_cover):
#
# 		image, cover_image = image_and_cover
#
# 		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
# 																	 self.width_ratio)
# 		mask = torch.zeros_like(image)
# 		mask[:, :, h_start: h_end, w_start: w_end] = 1
#
# 		return image * mask
# 		# h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio, self.width_ratio)
# 		#
# 		#
# 		# output = image[:, :, h_start: h_end, w_start: w_end]
# 		#
# 		# return output

class RandomCrop(nn.Module):
	def __init__(self, min_ratio=0.7, max_ratio=1, target_size=0, proportional=False):
		super(RandomCrop, self).__init__()
		self.min_ratio = min_ratio
		self.max_ratio = max_ratio
		self.proportional = proportional
		self.target_size = target_size

	def forward(self, image_and_cover):
		self.height_ratio = np.random.randint(self.min_ratio, self.max_ratio) #(np.random.rand() * (self.max_ratio - self.min_ratio) + self.min_ratio)
		self.width_ratio = self.height_ratio #(np.random.rand() * (self.max_ratio - self.min_ratio) + self.min_ratio)

		# image, cover_image = image_and_cover
		#
		# h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, height_ratio, width_ratio)
		# output = image[:, :, h_start: h_end, w_start: w_end]
		#
		# return output
		image, cover_image = image_and_cover

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
		mask = torch.zeros_like(image)
		mask[:, :, h_start: h_end, w_start: w_end] = 1

		return image * mask


class Cropout(nn.Module):

	def __init__(self, ratio):
		super(Cropout, self).__init__()
		self.height_ratio = 1-ratio#height_ratio
		self.width_ratio = 1-ratio#width_ratio

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
		output = cover_image.clone()
		output[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
		return output

class Dropout(nn.Module):

	def __init__(self, prob):
		super(Dropout, self).__init__()
		self.prob = 1 - prob

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		rdn = torch.rand(image.shape).to(image.device)
		output = torch.where(rdn > self.prob * 1., cover_image, image)
		return output


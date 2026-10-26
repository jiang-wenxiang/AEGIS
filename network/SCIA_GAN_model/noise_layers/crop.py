import torch
import torch.nn as nn
import numpy as np
# def get_random_rectangle_inside(image, height_ratio, width_ratio):
# 	image_height = image.shape[2]
# 	image_width = image.shape[3]
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


def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):

    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(np.rint(random_float(height_ratio_range, height_ratio_range) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range, width_ratio_range) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:#np.random.randint()返回一个随机整数，包括低范围，不包括高范围
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width

class Crop(nn.Module):
	def __init__(self, ratio):
		super(Crop, self).__init__()
		self.height_ratio = np.sqrt(ratio)#height_ratio
		self.width_ratio = np.sqrt(ratio)#width_ratio

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image, self.height_ratio, self.width_ratio)

		###############################################################显示用  # 创建一个全白（1.0）的 mask
		mask = torch.ones_like(image)  # 初始化为全白
		mask[:, :, h_start:h_end, w_start:w_end] = image[:, :, h_start:h_end, w_start:w_end]		# 裁剪区域设为 1（保留原图），其余部分保持白色
		return mask

		###############################################################正式测试用
		# mask = torch.zeros_like(image)
		# mask[:, :, h_start: h_end, w_start: w_end] = 1
		# return image * mask

class Cropout(nn.Module):

	def __init__(self, ratio):
		super(Cropout, self).__init__()
		self.height_ratio = np.sqrt(ratio)
		self.width_ratio = np.sqrt(ratio)

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image, self.height_ratio,
																	 self.width_ratio)
		# print(f'image={image.shape} | S={image.shape[2]*image.shape[3]} | S = {(h_end-h_start) * (w_end-w_start)} | R = {((h_end-h_start) * (w_end-w_start)) / (image.shape[2]*image.shape[3])} | H={h_end-h_start} | W={w_end-w_start}')

		output = cover_image.clone()
		output[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
		return output

class Dropout(nn.Module):

	def __init__(self, prob):
		super(Dropout, self).__init__()
		# self.prob = prob  				# Combined
		self.prob = 1 - prob				# Single
	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		rdn = torch.rand(image.shape).to(image.device)
		output = torch.where(rdn > self.prob * 1., cover_image, image)
		return output

# class Dropout(nn.Module):
#     def __init__(self, keep_ratio):
#         super(Dropout, self).__init__()
#         self.keep_min = np.sqrt(keep_ratio)
#         self.keep_max = np.sqrt(keep_ratio)
#
#
#     def forward(self, image_and_cover):
#         image, cover_image = image_and_cover
#
#         mask_percent = np.random.uniform(self.keep_min, self.keep_max)# random number in [min,max]
#
#         mask = np.random.choice([0.0, 1.0], image.shape[2:], p=[1 - mask_percent, mask_percent])
#
#         mask_tensor = torch.tensor(mask, device=image.device, dtype=torch.float)
#
#         mask_tensor = mask_tensor.expand_as(image)
#
#         image = image * mask_tensor + cover_image * (1-mask_tensor)
#
#         return image

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import random, string
import numpy
from torch.autograd import Variable
import cv2
Image.MAX_IMAGE_PIXELS = None


class JpegTest(nn.Module):
	def __init__(self, Q, subsample=2, path="temp/"):
		super(JpegTest, self).__init__()
		self.Q = Q
		self.subsample = subsample
		self.path = path
		if not os.path.exists(path): os.mkdir(path) #如果指定的保存路径不存在，则创建该路径。

	def get_path(self):
		return self.path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".jpg" #在临时路径后面拼接16位随机字母和数字组成的文件名，并加上".jpg"后缀，返回生成的临时文件路径。

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		shape = image.shape
		noised_image = torch.zeros_like(image)#创建一个与原始图像形状相同的全零张量，用于存储压缩后的图像。

		for i in range(shape[0]): #遍历原始图像的所有通道。
			single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).to('cpu', torch.uint8).numpy() #将当前通道的原始图像张量转换为NumPy数组，并将像素值归一化到0-255范围内。
			im = Image.fromarray(single_image) #使用PIL库将NumPy数组转换为PIL图像对象。

			file = self.get_path()#调用get_path()方法获取临时文件路径。
			while os.path.exists(file): #如果生成的临时文件路径已经存在，则重新生成，直到得到一个不存在的临时文件路径。
				file = self.get_path()
			im.save(file, format="JPEG", quality=self.Q, subsampling=self.subsample) #将PIL图像对象以JPEG格式保存到临时文件中，指定压缩质量因子为self.Q，子采样级别为self.subsample。
			jpeg = np.array(Image.open(file), dtype=np.uint8)#使用PIL库打开保存的JPEG压缩图像文件，并将其转换为NumPy数组。
			os.remove(file)#删除临时文件。

			transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			])#定义了一个图像转换管道，将NumPy数组转换为PyTorch张量，并进行归一化处理。

			noised_image[i] = transform(jpeg).unsqueeze(0).to(image.device)#将处理后的图像张量赋值给noised_image张量的当前通道。

		return noised_image

class JpegBasic(nn.Module):
	def __init__(self):
		super(JpegBasic, self).__init__()

	def std_quantization(self, image_yuv_dct, scale_factor, round_func=torch.round):

		luminance_quant_tbl = (torch.tensor([
			[16, 11, 10, 16, 24, 40, 51, 61],
			[12, 12, 14, 19, 26, 58, 60, 55],
			[14, 13, 16, 24, 40, 57, 69, 56],
			[14, 17, 22, 29, 51, 87, 80, 62],
			[18, 22, 37, 56, 68, 109, 103, 77],
			[24, 35, 55, 64, 81, 104, 113, 92],
			[49, 64, 78, 87, 103, 121, 120, 101],
			[72, 92, 95, 98, 112, 100, 103, 99]
		], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
			image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

		chrominance_quant_tbl = (torch.tensor([
			[17, 18, 24, 47, 99, 99, 99, 99],
			[18, 21, 26, 66, 99, 99, 99, 99],
			[24, 26, 56, 99, 99, 99, 99, 99],
			[47, 66, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99]
		], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
			image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

		q_image_yuv_dct = image_yuv_dct.clone()
		q_image_yuv_dct[:, :1, :, :] = image_yuv_dct[:, :1, :, :] / luminance_quant_tbl
		q_image_yuv_dct[:, 1:, :, :] = image_yuv_dct[:, 1:, :, :] / chrominance_quant_tbl
		q_image_yuv_dct_round = round_func(q_image_yuv_dct)
		return q_image_yuv_dct_round

	def std_reverse_quantization(self, q_image_yuv_dct, scale_factor):

		luminance_quant_tbl = (torch.tensor([
			[16, 11, 10, 16, 24, 40, 51, 61],
			[12, 12, 14, 19, 26, 58, 60, 55],
			[14, 13, 16, 24, 40, 57, 69, 56],
			[14, 17, 22, 29, 51, 87, 80, 62],
			[18, 22, 37, 56, 68, 109, 103, 77],
			[24, 35, 55, 64, 81, 104, 113, 92],
			[49, 64, 78, 87, 103, 121, 120, 101],
			[72, 92, 95, 98, 112, 100, 103, 99]
		], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
			q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

		chrominance_quant_tbl = (torch.tensor([
			[17, 18, 24, 47, 99, 99, 99, 99],
			[18, 21, 26, 66, 99, 99, 99, 99],
			[24, 26, 56, 99, 99, 99, 99, 99],
			[47, 66, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99]
		], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
			q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

		image_yuv_dct = q_image_yuv_dct.clone()
		image_yuv_dct[:, :1, :, :] = q_image_yuv_dct[:, :1, :, :] * luminance_quant_tbl
		image_yuv_dct[:, 1:, :, :] = q_image_yuv_dct[:, 1:, :, :] * chrominance_quant_tbl
		return image_yuv_dct

	def dct(self, image):
		# coff for dct and idct
		coff = torch.zeros((8, 8), dtype=torch.float).to(image.device)
		coff[0, :] = 1 * np.sqrt(1 / 8)
		for i in range(1, 8):
			for j in range(8):
				coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

		split_num = image.shape[2] // 8
		image_dct = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
		image_dct = torch.matmul(coff, image_dct)
		image_dct = torch.matmul(image_dct, coff.permute(1, 0))
		image_dct = torch.cat(torch.cat(image_dct.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

		return image_dct

	def idct(self, image_dct):
		# coff for dct and idct
		coff = torch.zeros((8, 8), dtype=torch.float).to(image_dct.device)
		coff[0, :] = 1 * np.sqrt(1 / 8)
		for i in range(1, 8):
			for j in range(8):
				coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

		split_num = image_dct.shape[2] // 8
		image = torch.cat(torch.cat(image_dct.split(8, 2), 0).split(8, 3), 0)
		image = torch.matmul(coff.permute(1, 0), image)
		image = torch.matmul(image, coff)
		image = torch.cat(torch.cat(image.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

		return image

	def rgb2yuv(self, image_rgb):
		image_yuv = torch.empty_like(image_rgb)
		image_yuv[:, 0:1, :, :] = 0.299 * image_rgb[:, 0:1, :, :] \
								  + 0.587 * image_rgb[:, 1:2, :, :] + 0.114 * image_rgb[:, 2:3, :, :]
		image_yuv[:, 1:2, :, :] = -0.1687 * image_rgb[:, 0:1, :, :] \
								  - 0.3313 * image_rgb[:, 1:2, :, :] + 0.5 * image_rgb[:, 2:3, :, :]
		image_yuv[:, 2:3, :, :] = 0.5 * image_rgb[:, 0:1, :, :] \
								  - 0.4187 * image_rgb[:, 1:2, :, :] - 0.0813 * image_rgb[:, 2:3, :, :]
		return image_yuv

	def yuv2rgb(self, image_yuv):
		image_rgb = torch.empty_like(image_yuv)
		image_rgb[:, 0:1, :, :] = image_yuv[:, 0:1, :, :] + 1.40198758 * image_yuv[:, 2:3, :, :]
		image_rgb[:, 1:2, :, :] = image_yuv[:, 0:1, :, :] - 0.344113281 * image_yuv[:, 1:2, :, :] \
								  - 0.714103821 * image_yuv[:, 2:3, :, :]
		image_rgb[:, 2:3, :, :] = image_yuv[:, 0:1, :, :] + 1.77197812 * image_yuv[:, 1:2, :, :]
		return image_rgb

	def yuv_dct(self, image, subsample):
		# clamp and convert from [-1,1] to [0,255]
		image = (image.clamp(-1, 1) + 1) * 255 / 2

		# pad the image so that we can do dct on 8x8 blocks
		pad_height = (8 - image.shape[2] % 8) % 8
		pad_width = (8 - image.shape[3] % 8) % 8
		image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(image)

		# convert to yuv
		image_yuv = self.rgb2yuv(image)

		assert image_yuv.shape[2] % 8 == 0
		assert image_yuv.shape[3] % 8 == 0

		# subsample
		image_subsample = self.subsampling(image_yuv, subsample)

		# apply dct
		image_dct = self.dct(image_subsample)

		return image_dct, pad_width, pad_height

	def idct_rgb(self, image_quantization, pad_width, pad_height):
		# apply inverse dct (idct)
		image_idct = self.idct(image_quantization)

		# transform from yuv to to rgb
		image_ret_padded = self.yuv2rgb(image_idct)

		# un-pad
		image_rgb = image_ret_padded[:, :, :image_ret_padded.shape[2] - pad_height,
					:image_ret_padded.shape[3] - pad_width].clone()

		return image_rgb * 2 / 255 - 1

	def subsampling(self, image, subsample):
		if subsample == 2:
			split_num = image.shape[2] // 8
			image_block = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
			for i in range(8):
				if i % 2 == 1: image_block[:, 1:3, i, :] = image_block[:, 1:3, i - 1, :]
			for j in range(8):
				if j % 2 == 1: image_block[:, 1:3, :, j] = image_block[:, 1:3, :, j - 1]
			image = torch.cat(torch.cat(image_block.chunk(split_num, 0), 3).chunk(split_num, 0), 2)
		return image


class Jpeg(JpegBasic):
	def __init__(self, Q, subsample=0):
		super(Jpeg, self).__init__()

		# quantization table
		self.Q = Q
		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

		# subsample
		self.subsample = subsample

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		# [-1,1] to [0,255], rgb2yuv, dct
		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

		# quantization
		image_quantization = self.std_quantization(image_dct, self.scale_factor)

		# reverse quantization
		image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

		# idct, yuv2rgb, [0,255] to [-1,1]
		noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
		return noised_image.clamp(-1, 1)


class JpegSS(JpegBasic):
	def __init__(self, Q, subsample=0):
		super(JpegSS, self).__init__()

		# quantization table
		self.Q = Q
		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

		# subsample
		self.subsample = subsample

	def round_ss(self, x):
		cond = torch.tensor((torch.abs(x) < 0.5), dtype=torch.float).to(x.device)
		return cond * (x ** 3) + (1 - cond) * x

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		# [-1,1] to [0,255], rgb2yuv, dct
		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

		# quantization
		image_quantization = self.std_quantization(image_dct, self.scale_factor, self.round_ss)

		# reverse quantization
		image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

		# idct, yuv2rgb, [0,255] to [-1,1]
		noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
		return noised_image.clamp(-1, 1)


class JpegMask(JpegBasic):
	def __init__(self, Q, subsample=0):
		super(JpegMask, self).__init__()

		# quantization table
		self.Q = Q
		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

		# subsample
		self.subsample = subsample

	def round_mask(self, x):
		mask = torch.zeros(1, 3, 8, 8).to(x.device)
		mask[:, 0:1, :5, :5] = 1
		mask[:, 1:3, :3, :3] = 1
		mask = mask.repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)
		return x * mask

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		# [-1,1] to [0,255], rgb2yuv, dct
		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

		# mask
		image_mask = self.round_mask(image_dct)

		# idct, yuv2rgb, [0,255] to [-1,1]
		noised_image = self.idct_rgb(image_mask, pad_width, pad_height)
		return noised_image.clamp(-1, 1)

class JpegSS_diffround(JpegBasic):
	def __init__(self, Q, subsample=0):
		super(JpegSS_diffround, self).__init__()

		# quantization table
		self.Q = Q
		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

		# subsample
		self.subsample = subsample

	def round_ss(self, x):
		cond = torch.tensor((torch.abs(x) < 0.5), dtype=torch.float).to(x.device)
		return cond * (x ** 3) + (1 - cond) * x

	def diff_round(self, x):
		sign = torch.ones_like(x)
		sign[torch.floor(x) % 2 == 0] = -1
		y = sign * torch.cos(x * torch.pi) / 2
		out = y + 0.5 + torch.floor(x)
		return out

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		# [-1,1] to [0,255], rgb2yuv, dct
		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

		# quantization
		image_quantization = self.std_quantization(image_dct, self.scale_factor, self.diff_round)

		# reverse quantization
		image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

		# idct, yuv2rgb, [0,255] to [-1,1]
		noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
		return noised_image.clamp(-1, 1)


class JpegMask_diffround(JpegBasic):
	def __init__(self, Q, subsample=0):
		super(JpegMask_diffround, self).__init__()

		# quantization table
		self.Q = Q
		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

		# subsample
		self.subsample = subsample

	def round_mask(self, x):
		mask = torch.zeros(1, 3, 8, 8).to(x.device)
		mask[:, 0:1, :5, :5] = 1
		mask[:, 1:3, :3, :3] = 1
		mask = mask.repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)
		return x * mask
	def diff_round(self, x):
		sign = torch.ones_like(x)
		sign[torch.floor(x) % 2 == 0] = -1
		y = sign * torch.cos(x * torch.pi) / 2
		out = y + 0.5 + torch.floor(x)
		return out

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		# [-1,1] to [0,255], rgb2yuv, dct
		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

		# mask
		image_mask = self.diff_round(image_dct)

		# idct, yuv2rgb, [0,255] to [-1,1]
		noised_image = self.idct_rgb(image_mask, pad_width, pad_height)
		return noised_image.clamp(-1, 1)


Y_QM_50 = numpy.mat([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])
UV_QM_50 = numpy.mat([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 98, 99, 99],
    [24, 26, 56, 99, 99, 99, 100, 99],
    [47, 66, 99, 99, 99, 99, 99, 98],
    [100, 99, 99, 99, 99, 99, 99, 99],
    [99, 98, 99, 99, 99, 99, 99, 99],
    [99, 99, 100, 99, 99, 99, 99, 99],
    [99, 99, 99, 98, 99, 99, 99, 99]
])
'''
UV_QM_50 = numpy.mat([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])
'''
def Y_QM(q):
    if (q < 50):
        S = 5000 / q
    else:
        S = 200 - 2 * q
    return ((S * Y_QM_50 + 50.) / 100.)
def UV_QM(q):
    if (q < 50):
        S = 5000 / q
    else:
        S = 200 - 2 * q
    return ((S * UV_QM_50 + 50.) / 100.)
def DCT(img):
    C_temp = numpy.zeros(img.shape)
    dst = numpy.zeros(img.shape)
    m, n = img.shape
    N = n
    C_temp[0, :] = 1 * numpy.sqrt(1 / N)

    for i in range(1, m):
        for j in range(n):
            C_temp[i, j] = numpy.cos(numpy.pi * i * (2 * j + 1) / (2 * N)
                                     ) * numpy.sqrt(2 / N)

    dst = numpy.dot(C_temp, img)
    dst = numpy.dot(dst, numpy.transpose(C_temp))
    return dst
def IDCT(img):
    C_temp = numpy.zeros(img.shape)
    dst = numpy.zeros(img.shape)
    m, n = img.shape
    N = n
    C_temp[0, :] = 1 * numpy.sqrt(1 / N)

    for i in range(1, m):
        for j in range(n):
            C_temp[i, j] = numpy.cos(numpy.pi * i * (2 * j + 1) / (2 * N)
                                     ) * numpy.sqrt(2 / N)

    dst = numpy.dot(C_temp, img)
    dst = numpy.dot(dst, numpy.transpose(C_temp))
    img_recor = numpy.dot(numpy.transpose(C_temp), dst)
    img_recor1 = numpy.dot(img_recor, C_temp)
    return img_recor1
def JPEG_Mask(img, q):
    stride = 8

    img_Y = img[:, :, 0]
    img_U = img[:, :, 1]
    img_V = img[:, :, 2]

    h, w, c = img.shape
    img_Y_JPEG = numpy.zeros((h, w))
    img_U_JPEG = numpy.zeros((h, w))
    img_V_JPEG = numpy.zeros((h, w))
    img_JPEG = numpy.zeros((h, w, c))

    for x in range(h // stride):
        for y in range(w // stride):
            # sub_image_zeros = numpy.zeros((stride,stride),dtype=numpy.float32)
            Y_block = img_Y[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride]
            U_block = img_U[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride]
            V_block = img_V[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride]
            # sub_image = sub_image_zeros + block  # [8x8]
            Y_DCT_block = DCT(Y_block)
            U_DCT_block = DCT(U_block)
            V_DCT_block = DCT(V_block)

            if (q > 0 and q < 100):

                Y_IDCT_block = cv2.idct(
                    numpy.array(numpy.multiply(numpy.mat(numpy.rint(Y_DCT_block / Y_QM(q))), Y_QM(q))))
                U_IDCT_block = cv2.idct(
                    numpy.array(numpy.multiply(numpy.mat(numpy.rint(U_DCT_block / UV_QM(q))), UV_QM(q))))
                V_IDCT_block = cv2.idct(
                    numpy.array(numpy.multiply(numpy.mat(numpy.rint(V_DCT_block / UV_QM(q))), UV_QM(q))))
            else:
                for i in range(stride):
                    for j in range(stride):
                        if (i > 4 or j > 4):
                            Y_DCT_block[i, j] = 0
                        if (i > 2 or j > 2):
                            U_DCT_block[i, j] = 0
                            V_DCT_block[i, j] = 0
                Y_IDCT_block = cv2.idct(Y_DCT_block)
                U_IDCT_block = cv2.idct(U_DCT_block)
                V_IDCT_block = cv2.idct(V_DCT_block)

            img_Y_JPEG[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride] = img_Y_JPEG[
                                                                                   x * stride:(x + 1) * stride,
                                                                                   y * stride:(
                                                                                                          y + 1) * stride] + Y_IDCT_block
            img_U_JPEG[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride] = img_U_JPEG[
                                                                                   x * stride:(x + 1) * stride,
                                                                                   y * stride:(
                                                                                                          y + 1) * stride] + U_IDCT_block
            img_V_JPEG[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride] = img_V_JPEG[
                                                                                   x * stride:(x + 1) * stride,
                                                                                   y * stride:(
                                                                                                          y + 1) * stride] + V_IDCT_block

    img_JPEG[:, :, 0] = img_Y_JPEG
    img_JPEG[:, :, 1] = img_U_JPEG
    img_JPEG[:, :, 2] = img_V_JPEG

    img_JPEG = numpy.uint8(img_JPEG)
    return img_JPEG

class Jpeg_compression(nn.Module):
	def __init__(self, Q):
		super(Jpeg_compression, self).__init__()
		self.Q = Q

	def forward(self, image_and_cover):
		image, cover = image_and_cover
		batch_encoded_image = image.cpu().detach().numpy() * 255
		# batch_encoded_image = torch.tensor(batch_encoded_image).cuda()
		batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
		for idx in range(batch_encoded_image.shape[0]):
			encoded_image = batch_encoded_image[idx]
			# noise_image = JPEG2.JPEG_Mask(encoded_image,Q)
			noise_image = JPEG_Mask(encoded_image, self.Q)
			noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
			if (idx == 0):
				batch_noise_image = noise_image.unsqueeze(0)
			else:
				batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
		batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
		return batch_noise_image / 255







class RandomJpeg(JpegBasic):
	"""
	JPEG with random compression quality parameter
	"""

	def __init__(self, min_quality=50, max_quality=100, subsample=2, padding=False):
		super(RandomJpeg, self).__init__()
		self.max_quality = max_quality
		self.min_quality = min_quality
		# subsample
		self.subsample = subsample
		self.padding = padding

	def forward(self, image_and_cover):
		# quantization table

		self.Q = np.random.randint(self.min_quality, self.max_quality)

		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

		image, cover = image_and_cover

		if self.padding:
			h = np.random.randint(0, 8)
			w = np.random.randint(0, 8)
			image = nn.ReflectionPad2d((w, w, h, h))(image)

		# [-1,1] to [0,255], rgb2yuv, dct
		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

		# quantization
		image_quantization = self.std_quantization(image_dct, self.scale_factor)

		# reverse quantization
		image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

		# idct, yuv2rgb, [0,255] to [-1,1]
		noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)

		if self.padding:
			noised_image = noised_image[:, :, h:noised_image.shape[2] - h, w:noised_image.shape[3] - w]
		if self.Q == 50:
			print("JPEG")
		return noised_image.clamp(-1, 1)
class RandomJpegSS(JpegBasic):
	"""
	JPEG-SS with random compression quality parameter
	"""

	def __init__(self, min_quality=50, max_quality=100, subsample=2, padding=False):
		super(RandomJpegSS, self).__init__()
		self.min_quality = min_quality
		self.max_quality = max_quality
		# subsample
		self.subsample = subsample
		self.padding = padding

	def round_ss(self, x):
		cond = (torch.abs(x) < 0.5)
		cond = cond.float()
		cond = cond.to(x.device)
		return cond * (x ** 3) + (1 - cond) * x

	def forward(self, image_and_cover):

		# quantization table
		self.Q = np.random.randint(self.min_quality, self.max_quality)
		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

		image, cover = image_and_cover

		if self.padding:
			h = np.random.randint(0, 8)
			w = np.random.randint(0, 8)
			image = nn.ReflectionPad2d((w, w, h, h))(image)

		# self.subsample = np.random.choice(a=[0,2],p=[0.5,0.5])
		# [-1,1] to [0,255], rgb2yuv, dct
		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

		# quantization
		image_quantization = self.std_quantization(image_dct, self.scale_factor, self.round_ss)

		# reverse quantization
		image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

		# idct, yuv2rgb, [0,255] to [-1,1]
		noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)

		if self.padding:
			noised_image = noised_image[:, :, h:noised_image.shape[2] - h, w:noised_image.shape[3] - w]

		return noised_image.clamp(-1, 1)
class RandomJpegMask(JpegBasic):
	"""
	JPEGMask with random compression quality parameter
	"""

	def __init__(self, min_quality=50, max_quality=100, subsample=2, padding=False):
		super(RandomJpegMask, self).__init__()
		self.min_quality = min_quality
		self.max_quality = max_quality
		self.padding = padding

		# subsample
		self.subsample = subsample

	def round_mask(self, x):
		mask = torch.zeros(1, 3, 8, 8).to(x.device)
		mask[:, 0:1, :5, :5] = 1
		mask[:, 1:3, :3, :3] = 1
		mask = mask.repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)
		return x * mask

	def forward(self, image_and_cover):
		# quantization table
		self.Q = np.random.randint(self.min_quality, self.max_quality)
		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

		image, cover = image_and_cover

		if self.padding:
			h = np.random.randint(0, 9)
			w = np.random.randint(0, 9)
			image = nn.ReflectionPad2d((w, w, h, h))(image)

		# self.subsample = np.random.choice(a=[0,2],p=[0.5,0.5])
		# [-1,1] to [0,255], rgb2yuv, dct
		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

		# mask
		image_mask = self.round_mask(image_dct)

		# idct, yuv2rgb, [0,255] to [-1,1]
		noised_image = self.idct_rgb(image_mask, pad_width, pad_height)

		if self.padding:
			noised_image = noised_image[:, :, h:noised_image.shape[2] - h, w:noised_image.shape[3] - w]

		return noised_image.clamp(-1, 1)


class RandomJpegTest(nn.Module):
	"""
	Real JPEG with random compression quality parameter
	"""

	def __init__(self, min_quality=50, max_quality=100, subsample=2, path="temp/"):
		super(RandomJpegTest, self).__init__()
		self.subsample = subsample
		self.path = path
		self.max_quality = max_quality
		self.min_quality = min_quality
		if not os.path.exists(path):
			os.mkdir(path)

	def get_path(self):
		return self.path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".jpg"

	def forward(self, image_and_cover):
		self.Q = np.random.randint(self.min_quality, self.max_quality)
		image, cover_image = image_and_cover

		shape = image.shape
		noised_image = torch.zeros_like(image)

		for i in range(shape[0]):
			single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).to('cpu', torch.uint8).numpy()
			im = Image.fromarray(single_image)

			file = self.get_path()
			while os.path.exists(file):
				file = self.get_path()
			im.save(file, format="JPEG", quality=self.Q, subsampling=self.subsample)
			jpeg = np.array(Image.open(file), dtype=np.uint8)
			os.remove(file)

			transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			])

			noised_image[i] = transform(jpeg).unsqueeze(0).to(image.device)

		return noised_image
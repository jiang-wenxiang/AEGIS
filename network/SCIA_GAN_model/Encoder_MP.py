import torch

from . import *
# from network.blocks.DCTConv import dctconv
# import torch.nn as nn
# import torch
# import numpy as np
# from network.blocks.ConvNet import ConvBNRelu
# from network.blocks.SENet import BottleneckBlock
# from network.blocks.ExpandNet import ExpandNet

class Encoder_Diffusion(nn.Module):
	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
		super(Encoder_Diffusion, self).__init__()
		self.H = H
		self.W = W

		message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		message_se_blocks = max(blocks - message_convT_blocks, 1)
		self.diffusion_length = diffusion_length
		self.diffusion_size = int(diffusion_length ** 0.5)
		self.image_pre_layer = ConvBNRelu(3, channels)
		self.image_first_layer = SENet_Ca(channels, channels, blocks=blocks)
		self.message_duplicate_layer = \
			nn.Sequential(
				nn.Linear(message_length, 128),
				nn.ReLU(inplace=True),
				nn.Linear(128, self.diffusion_length))
			# nn.Linear(message_length, self.diffusion_length)

		self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks-1),#源代码 -1；                 128，128，30   -没有1
			SENet_Sa(channels, channels, blocks=message_se_blocks))
		self.message_first_layer = SENet_Ca(channels, channels, blocks=blocks)
		self.after_concat_layer = ConvBNRelu(2 * channels, channels)
		# self.dense_layer = Denseblock(channels + message_length, channels)
		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

	def forward(self, image, message):
		# first Conv part of Encoder
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor (with diffusion)
		message_duplicate = self.message_duplicate_layer(message)  #30:4 1 16 16   120
		message_image = message_duplicate.view(-1, 1, self.diffusion_size, self.diffusion_size)

		message_pre = self.message_pre_layer(message_image)
		intermediate2 = self.message_first_layer(message_pre)
		# print(f"intermediate1={intermediate1.shape} | intermediate2={intermediate2.shape}")
		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		intermediate3 = self.after_concat_layer(concat1)

		# intermediate3 = self.dense_layer(torch.cat([intermediate3, expanded_message], dim=1))

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Network
		output = self.final_layer(concat2)
		return output


class Encoder_Diffusion1(nn.Module):
	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
		super(Encoder_Diffusion1, self).__init__()
		self.H = H
		self.W = W

		message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		message_se_blocks = max(blocks - message_convT_blocks, 1)
		self.diffusion_length = diffusion_length
		self.diffusion_size = int(diffusion_length ** 0.5)
		self.image_pre_layer = ConvBNRelu(3, channels)
		self.image_first_layer = SENet_Ca(channels, channels, blocks=blocks)
		self.message_duplicate_layer = \
			nn.Sequential(
				nn.Linear(message_length, self.diffusion_length))
			# nn.Linear(message_length, self.diffusion_length)

		self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks),
			SENet_Sa(channels, channels, blocks=message_se_blocks))
		self.message_first_layer = SENet_Ca(channels, channels, blocks=blocks)
		self.after_concat_layer = ConvBNRelu(2 * channels, channels)
		# self.dense_layer = Denseblock(channels + message_length, channels)
		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

	def forward(self, image, message):
		# first Conv part of Encoder
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor (with diffusion)
		message_duplicate = self.message_duplicate_layer(message)  #30:4 1 16 16   120
		message_image = message_duplicate.view(-1, 1, self.diffusion_size, self.diffusion_size)

		message_pre = self.message_pre_layer(message_image)
		intermediate2 = self.message_first_layer(message_pre)
		# print(f"intermediate1={intermediate1.shape} | intermediate2={intermediate2.shape}")
		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		intermediate3 = self.after_concat_layer(concat1)

		# intermediate3 = self.dense_layer(torch.cat([intermediate3, expanded_message], dim=1))

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Network
		output = self.final_layer(concat2)
		return output


class Encoder_Init(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Encoder_Init, self).__init__()
		self.H = H
		self.W = W

		message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		message_se_blocks = max(blocks - message_convT_blocks, 1)

		self.image_pre_layer = ConvBNRelu(3, channels)
		self.image_first_layer = SENet_Ca(channels, channels, blocks=blocks)
		# print(self.image_first_layer)
		self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks),
			SENet_Sa(channels, channels, blocks=message_se_blocks),
		)

		self.message_first_layer = SENet_Ca(channels, channels, blocks=blocks)

		# print(self.message_first_layer)
		self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

	def forward(self, image, message):
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor
		size = int(np.sqrt(message.shape[1]))
		message_image = message.view(-1, 1, size, size)
		# print(message_image.shape)
		message_pre = self.message_pre_layer(message_image)
		intermediate2 = self.message_first_layer(message_pre)
		# print(intermediate2.shape)
		# print(intermediate2.shape)
		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Encoder
		output = self.final_layer(concat2)

		return output


class Encoder_Init1(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Encoder_Init1, self).__init__()
		self.H = H
		self.W = W

		message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		message_se_blocks = max(blocks - message_convT_blocks, 1)

		self.image_pre_layer = ConvBNRelu(3, channels)
		self.image_first_layer = SENet_Ca(channels, channels, blocks=blocks)
		# print(self.image_first_layer)

		self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks),
			SENet_Sa(channels, channels, blocks=message_se_blocks),
		)

		self.message_first_layer = SENet_Ca(channels, channels, blocks=blocks)

		# print(self.message_first_layer)
		self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

	def forward(self, image, message):
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor
		size = int(np.sqrt(message.shape[1]))
		message_image = message.view(-1, 1, size, size)
		# print(message_image.shape)
		message_pre = self.message_pre_layer(message_image)
		intermediate2 = self.message_first_layer(message_pre)
		print(intermediate1.shape)
		print(intermediate2.shape)
		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Encoder
		output = self.final_layer(concat2)

		return output


# Ablation		CASA
class Encoder_Init_CASA(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Encoder_Init_CASA, self).__init__()
		self.H = H
		self.W = W

		message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		message_se_blocks = max(blocks - message_convT_blocks, 1)

		self.image_pre_layer = ConvBNRelu(3, channels)
		self.image_first_layer = SENet_Ca_Ablation_CASA(channels, channels, blocks=blocks)
		# print(self.image_first_layer)
		self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks),
			SENet_Sa_Ablation_CASA(channels, channels, blocks=message_se_blocks),
		)

		self.message_first_layer = SENet_Ca_Ablation_CASA(channels, channels, blocks=blocks)

		# print(self.message_first_layer)
		self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

	def forward(self, image, message):
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor
		size = int(np.sqrt(message.shape[1]))
		message_image = message.view(-1, 1, size, size)
		# print(message_image.shape)
		message_pre = self.message_pre_layer(message_image)
		intermediate2 = self.message_first_layer(message_pre)
		# print(intermediate2.shape)
		# print(intermediate2.shape)
		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Encoder
		output = self.final_layer(concat2)

		return output


# Ablation		SACA
class Encoder_Init_SACA(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Encoder_Init_SACA, self).__init__()
		self.H = H
		self.W = W

		message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		message_se_blocks = max(blocks - message_convT_blocks, 1)

		self.image_pre_layer = ConvBNRelu(3, channels)
		self.image_first_layer = SENet_Ca_Ablation_SACA(channels, channels, blocks=blocks)
		# print(self.image_first_layer)
		self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks),
			SENet_Sa_Ablation_SACA(channels, channels, blocks=message_se_blocks),
		)

		self.message_first_layer = SENet_Ca_Ablation_SACA(channels, channels, blocks=blocks)

		# print(self.message_first_layer)
		self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

	def forward(self, image, message):
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor
		size = int(np.sqrt(message.shape[1]))
		message_image = message.view(-1, 1, size, size)
		# print(message_image.shape)
		message_pre = self.message_pre_layer(message_image)
		intermediate2 = self.message_first_layer(message_pre)
		# print(intermediate2.shape)
		# print(intermediate2.shape)
		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Encoder
		output = self.final_layer(concat2)

		return output

# Ablation		qu SA
class Encoder_Init_CA(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Encoder_Init_CA, self).__init__()
		self.H = H
		self.W = W

		message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		message_se_blocks = max(blocks - message_convT_blocks, 1)

		self.image_pre_layer = ConvBNRelu(3, channels)
		self.image_first_layer = SENet_Ca(channels, channels, blocks=blocks)
		# print(self.image_first_layer)
		self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks),
		)

		self.message_first_layer = SENet_Ca(channels, channels, blocks=blocks)

		# print(self.message_first_layer)
		self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

	def forward(self, image, message):
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor
		size = int(np.sqrt(message.shape[1]))
		message_image = message.view(-1, 1, size, size)
		# print(message_image.shape)
		message_pre = self.message_pre_layer(message_image)
		intermediate2 = self.message_first_layer(message_pre)
		# print(intermediate2.shape)
		# print(intermediate2.shape)
		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Encoder
		output = self.final_layer(concat2)

		return output

# Ablation		qu SA CA
class Encoder_Init_CAA(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Encoder_Init_CAA, self).__init__()
		self.H = H
		self.W = W

		message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		message_se_blocks = max(blocks - message_convT_blocks, 1)

		self.image_pre_layer = ConvBNRelu(3, channels)
		self.image_first_layer = SENet_Ca_Ablation_CAA(channels, channels, blocks=blocks)
		# print(self.image_first_layer)
		self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks),
		)

		self.message_first_layer = SENet_Ca_Ablation_CAA(channels, channels, blocks=blocks)

		# print(self.message_first_layer)
		self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

	def forward(self, image, message):
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor
		size = int(np.sqrt(message.shape[1]))
		message_image = message.view(-1, 1, size, size)
		# print(message_image.shape)
		message_pre = self.message_pre_layer(message_image)
		intermediate2 = self.message_first_layer(message_pre)
		# print(intermediate2.shape)
		# print(intermediate2.shape)
		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Encoder
		output = self.final_layer(concat2)

		return output



if __name__ == '__main__':
	image = torch.randn(1, 3, 128, 128)  # 2-1
	message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], 64)))
	model = Encoder_Diffusion(128,128,64)  # FMRDBplus_out:50756 ResidualDenseBlock_out:84380
	output = model(image, message)
	print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
	print(output.shape)

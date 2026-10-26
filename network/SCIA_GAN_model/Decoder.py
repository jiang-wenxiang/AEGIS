from . import *

# import torch.nn as nn
# import torch
# import numpy as np
# from network.blocks.ConvNet import ConvBNRelu
# from network.blocks.SENet import BottleneckBlock
# from network.blocks.ExpandNet import ExpandNet

class Decoder_Diffusion(nn.Module):
	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
		super(Decoder_Diffusion, self).__init__()

		stride_blocks = int(np.log2(H // int(np.sqrt(diffusion_length))))

		keep_blocks = max(blocks - stride_blocks, 0)

		self.diffusion_length = diffusion_length
		self.diffusion_size = int(self.diffusion_length ** 0.5)

		self.first_layers = nn.Sequential(
			ConvBNRelu(3, channels),
			SENet_decoder(channels, channels, blocks=stride_blocks + 1), #120 +2       #30 +1
			ConvBNRelu(channels * (2 ** stride_blocks), channels))

		self.keep_layers = SENet_Sa(channels, channels, blocks=keep_blocks)

		# self.dense_layers = Denseblock(channels * 2, channels)

		self.final_layer = ConvBNRelu(channels, 1)

		self.message_layer = \
			nn.Sequential(
		 				nn.Linear(self.diffusion_length, 128),
		 				nn.ReLU(inplace=True),
		 				nn.Linear(128, message_length))
			# nn.Linear(self.diffusion_length, message_length)









	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		feature = self.keep_layers(x)
		# feature = self.dense_layers(torch.cat((x, feature), dim=1))
		x = self.final_layer(feature)
		x = x.view(x.shape[0], -1)
		# print(x.shape)
		x = self.message_layer(x)
		# print(x.shape)
		return x


class Decoder_Diffusion1(nn.Module):
	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
		super(Decoder_Diffusion1, self).__init__()

		stride_blocks = int(np.log2(H // int(np.sqrt(diffusion_length))))

		keep_blocks = max(blocks - stride_blocks, 0)

		self.diffusion_length = diffusion_length
		self.diffusion_size = int(self.diffusion_length ** 0.5)

		self.first_layers = nn.Sequential(
			ConvBNRelu(3, channels),
			SENet_decoder(channels, channels, blocks=stride_blocks + 1), #120 +2       #30 +1
			ConvBNRelu(channels * (2 ** stride_blocks), channels))

		self.keep_layers = SENet_Sa(channels, channels, blocks=keep_blocks)

		# self.dense_layers = Denseblock(channels * 2, channels)

		self.final_layer = ConvBNRelu(channels, 1)

		self.message_layer = \
			nn.Sequential(
		 				nn.Linear(self.diffusion_length, message_length))
			# nn.Linear(self.diffusion_length, message_length)









	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		feature = self.keep_layers(x)
		# feature = self.dense_layers(torch.cat((x, feature), dim=1))
		x = self.final_layer(feature)
		x = x.view(x.shape[0], -1)
		# print(x.shape)
		x = self.message_layer(x)
		# print(x.shape)
		return x




class Decoder_Init(nn.Module):
	'''
	Decode the encoded image and get message
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Decoder_Init, self).__init__()

		stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		keep_blocks = max(blocks - stride_blocks, 0)

		self.first_layers = nn.Sequential(
			ConvBNRelu(3, channels),
			SENet_decoder(channels, channels, blocks=stride_blocks + 1),
			ConvBNRelu(channels * (2 ** stride_blocks), channels),
		)
		self.keep_layers = SENet_Sa(channels, channels, blocks=keep_blocks)
		# print(self.keep_layers)
		self.final_layer = ConvBNRelu(channels, 1)

	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		x = self.keep_layers(x)
		x = self.final_layer(x)
		x = x.view(x.shape[0], -1)
		return x

class Decoder_Init1(nn.Module):
	'''
	Decode the encoded image and get message
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Decoder_Init1, self).__init__()

		stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		keep_blocks = max(blocks - stride_blocks, 0)

		self.first_layers = nn.Sequential(
			ConvBNRelu(3, channels),
			SENet_decoder(channels, channels, blocks=stride_blocks + 1),
			ConvBNRelu(channels * (2 ** stride_blocks), channels),
		)
		self.keep_layers = SENet_Sa(channels, channels, blocks=keep_blocks)
		# print(self.keep_layers)
		self.final_layer = ConvBNRelu(channels, 1)

	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		x = self.keep_layers(x)
		x = self.final_layer(x)
		x = x.view(x.shape[0], -1)
		return x

# qu SA
class Decoder_Init_CA(nn.Module):
	'''
	Decode the encoded image and get message
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Decoder_Init_CA, self).__init__()

		stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		keep_blocks = max(blocks - stride_blocks, 0)

		self.first_layers = nn.Sequential(
			ConvBNRelu(3, channels),
			SENet_decoder(channels, channels, blocks=stride_blocks + 1),
			ConvBNRelu(channels * (2 ** stride_blocks), channels),
		)
		# self.keep_layers = SENet_Sa(channels, channels, blocks=keep_blocks)
		# print(self.keep_layers)
		self.final_layer = ConvBNRelu(channels, 1)

	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		# x = self.keep_layers(x)
		x = self.final_layer(x)
		x = x.view(x.shape[0], -1)
		return x
# Ablation		CASA
class Decoder_Init_CASA(nn.Module):
	'''
	Decode the encoded image and get message
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Decoder_Init_CASA, self).__init__()

		stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		keep_blocks = max(blocks - stride_blocks, 0)

		self.first_layers = nn.Sequential(
			ConvBNRelu(3, channels),
			SENet_decoder_Ablation_CASA(channels, channels, blocks=stride_blocks + 1),
			ConvBNRelu(channels * (2 ** stride_blocks), channels),
		)
		self.keep_layers = SENet_Sa_Ablation_CASA(channels, channels, blocks=keep_blocks)
		# print(self.keep_layers)
		self.final_layer = ConvBNRelu(channels, 1)

	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		x = self.keep_layers(x)
		x = self.final_layer(x)
		x = x.view(x.shape[0], -1)
		return x

# Ablation		SACA
class Decoder_Init_SACA(nn.Module):
	'''
	Decode the encoded image and get message
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Decoder_Init_SACA, self).__init__()

		stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		keep_blocks = max(blocks - stride_blocks, 0)

		self.first_layers = nn.Sequential(
			ConvBNRelu(3, channels),
			SENet_decoder_Ablation_SACA(channels, channels, blocks=stride_blocks + 1),
			ConvBNRelu(channels * (2 ** stride_blocks), channels),
		)
		self.keep_layers = SENet_Sa_Ablation_SACA(channels, channels, blocks=keep_blocks)
		# print(self.keep_layers)
		self.final_layer = ConvBNRelu(channels, 1)

	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		x = self.keep_layers(x)
		x = self.final_layer(x)
		x = x.view(x.shape[0], -1)
		return x

# qu SA CA
class Decoder_Init_CAA(nn.Module):
	'''
	Decode the encoded image and get message
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Decoder_Init_CAA, self).__init__()

		stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		keep_blocks = max(blocks - stride_blocks, 0)

		self.first_layers = nn.Sequential(
			ConvBNRelu(3, channels),
			SENet_decoder_Ablation_CAA(channels, channels, blocks=stride_blocks + 1),
			ConvBNRelu(channels * (2 ** stride_blocks), channels),
		)
		# self.keep_layers = SENet_Sa(channels, channels, blocks=keep_blocks)
		# print(self.keep_layers)
		self.final_layer = ConvBNRelu(channels, 1)

	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		# x = self.keep_layers(x)
		x = self.final_layer(x)
		x = x.view(x.shape[0], -1)
		return x





# if __name__ == '__main__':
# 	image = torch.randn(1, 3, 128, 128)  # 2-1
# 	message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], 64)))
# 	model = Decoder_Diffusion(128,128,64)  # FMRDBplus_out:50756 ResidualDenseBlock_out:84380
# 	output = model(image)
# 	print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
# 	print(output.shape)
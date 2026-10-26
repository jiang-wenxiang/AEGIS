# import torch.nn as nn
# import torch.nn.functional as F
# import torch
#
# class BasicBlock(nn.Module):
# 	def __init__(self, in_channels, out_channels, r, drop_rate):
# 		super(BasicBlock, self).__init__()
#
# 		self.downsample = None
# 		if (in_channels != out_channels):
# 			self.downsample = nn.Sequential(
# 				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
# 						  stride=drop_rate, bias=False),
# 				nn.BatchNorm2d(out_channels)
# 			)
#
# 		self.left = nn.Sequential(
# 			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
# 					  stride=drop_rate, bias=False),
# 			nn.BatchNorm2d(out_channels),
# 			nn.LeakyReLU(inplace=True),
# 			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
# 			nn.BatchNorm2d(out_channels),
# 		)
#
# 		self.se = nn.Sequential(
# 			nn.AdaptiveAvgPool2d((1, 1)),
# 			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
# 			nn.LeakyReLU(inplace=True),
# 			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
# 			nn.Sigmoid()
# 		)
#
# 	def forward(self, x):
# 		identity = x
# 		x = self.left(x)
# 		scale = self.se(x)
# 		x = x * scale
#
# 		if self.downsample is not None:
# 			identity = self.downsample(identity)
#
# 		x += identity
# 		x = F.relu(x)
# 		return x
# class BottleneckBlock(nn.Module):
# 	def __init__(self, in_channels, out_channels, r, drop_rate):
# 		super(BottleneckBlock, self).__init__()
#
# 		self.downsample = None
# 		if (in_channels != out_channels):
# 			self.downsample = nn.Sequential(
# 				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
# 						  stride=drop_rate, bias=False),
# 				nn.BatchNorm2d(out_channels)
# 			)
#
# 		self.left = nn.Sequential(
# 			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
# 					  stride=drop_rate, padding=0, bias=False),
# 			nn.BatchNorm2d(out_channels),
# 			nn.LeakyReLU(inplace=True),
# 			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
# 			nn.BatchNorm2d(out_channels),
# 			nn.LeakyReLU(inplace=True),
# 			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
# 			nn.BatchNorm2d(out_channels),
# 		)
#
# 		self.se = nn.Sequential(
# 			nn.AdaptiveAvgPool2d((1, 1)),
# 			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
# 			nn.LeakyReLU(inplace=True),
# 			nn.Conv2d(in_channels=out_channels // r, out_channels=out_channels, kernel_size=1, bias=False),
# 			nn.Sigmoid()
# 		)
#
# 	def forward(self, x):
# 		identity = x
# 		x = self.left(x)
# 		# print('left', x.shape)
# 		scale = self.se(x)
# 		# print('se',scale.shape)
# 		x = x * scale
#
# 		if self.downsample is not None:
# 			identity = self.downsample(identity)
# 			# print('downsample', identity.shape)
# 		x += identity
# 		x = F.relu(x)
# 		return x
# class SENet(nn.Module):
# 	'''
# 	SENet, with BasicBlock and BottleneckBlock
# 	'''
#
# 	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=1):
# 		super(SENet, self).__init__()
#
# 		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate)] if blocks != 0 else []
# 		for _ in range(blocks - 1):
# 			layer = eval(block_type)(out_channels, out_channels, r, drop_rate)
# 			layers.append(layer)
#
# 		self.layers = nn.Sequential(*layers)
#
# 	def forward(self, x):
# 		return self.layers(x)
# class SENet_decoder(nn.Module):
# 	'''
# 	ResNet, with BasicBlock and BottleneckBlock
# 	'''
#
# 	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=2):
# 		super(SENet_decoder, self).__init__()
#
# 		layers = [eval(block_type)(in_channels, out_channels, r, 1)] if blocks != 0 else []
# 		for _ in range(blocks - 1):
# 			layer1 = eval(block_type)(out_channels, out_channels, r, 1)
# 			layers.append(layer1)
# 			layer2 = eval(block_type)(out_channels, out_channels * drop_rate, r, drop_rate)
# 			out_channels *= drop_rate
# 			layers.append(layer2)
#
# 		self.layers = nn.Sequential(*layers)
#
# 	def forward(self, x):
# 		return self.layers(x)

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
class BasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels, r, drop_rate):
		super(BasicBlock, self).__init__()

		self.downsample = None
		if (in_channels != out_channels):
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
						  stride=drop_rate, bias=False),
				nn.BatchNorm2d(out_channels)
			)

		self.left = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
					  stride=drop_rate, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
		)

		self.se = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		identity = x
		x = self.left(x)
		scale = self.se(x)
		x = x * scale

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = F.relu(x)
		return x
class ChannelAttention(nn.Module):
	def __init__(self, channel, reduction=16):
		super().__init__()
		self.maxpool = nn.AdaptiveMaxPool2d(1)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.se = nn.Sequential(
			nn.Conv2d(channel, channel // reduction, 1, bias=False),
			# nn.BatchNorm2d(channel // reduction),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // reduction, channel, 1, bias=False),
			# nn.BatchNorm2d(channel),
			# nn.ReLU(inplace=True),
		)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		max_result = self.maxpool(x)
		avg_result = self.avgpool(x)
		max_out = self.se(max_result)
		avg_out = self.se(avg_result)
		output = self.sigmoid(max_out + avg_out)
		return output

class ChannelAttention_Ablation(nn.Module):
	def __init__(self, channel, reduction=16):
		super().__init__()
		# self.maxpool = nn.AdaptiveMaxPool2d(1)
		# self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.se = nn.Sequential(
			nn.Conv2d(channel, channel // reduction, 1, bias=False),
			# nn.BatchNorm2d(channel // reduction),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // reduction, channel, 1, bias=False),
			# nn.BatchNorm2d(channel),
			# nn.ReLU(inplace=True),
		)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# max_result = self.maxpool(x)
		# avg_result = self.avgpool(x)
		out = self.se(x)
		# avg_out = self.se(avg_result)
		output = self.sigmoid(out)
		return output

class SpatialAttention(nn.Module):
	def __init__(self, kernel_size=7):
		super().__init__()
		self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
		# self.bn = nn.BatchNorm2d(1)
		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		max_result, _ = torch.max(x, dim=1, keepdim=True)
		avg_result = torch.mean(x, dim=1, keepdim=True)
		result = torch.cat([max_result, avg_result], 1)
		output = self.relu(self.conv(result))
		output = self.sigmoid(output)
		return output


class BottleneckBlock(nn.Module):
	def __init__(self, in_channels, out_channels, r, drop_rate, Attention='se'):
		super(BottleneckBlock, self).__init__()

		self.downsample = None
		if (in_channels != out_channels):
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
						  stride=drop_rate, bias=False),
				nn.BatchNorm2d(out_channels)
			)

		self.left = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
					  stride=drop_rate, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
		)

		# self.se = nn.Sequential(
		# 	nn.AdaptiveAvgPool2d((1, 1)),
		# 	nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(in_channels=out_channels // r, out_channels=out_channels, kernel_size=1, bias=False),
		# 	nn.Sigmoid()
		# )
		if Attention=="ca":
			self.a =ChannelAttention(out_channels, reduction=r)
		if Attention=='sa':
			self.a = SpatialAttention(kernel_size=21)
		if Attention=='se':
			self.a = nn.Sequential(
				nn.AdaptiveAvgPool2d((1, 1)),
				nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_channels=out_channels // r, out_channels=out_channels, kernel_size=1, bias=False),
				nn.Sigmoid()
			)

	def forward(self, x):
		identity = x
		x = self.left(x)
		scale = self.a(x)
		x = x * scale

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = F.relu(x)
		return x



class SENet_decoder(nn.Module):
	'''
	ResNet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=2):
		super(SENet_decoder, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, 1, Attention='ca')] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, 1, Attention='ca')
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate, r, drop_rate, Attention='ca')
			out_channels *= drop_rate
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
class SENet_decoder_improve(nn.Module):
	'''
	ResNet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=2):
		super(SENet_decoder_improve, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, 1, drop_rate, Attention='ca')] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, 1, drop_rate, Attention='ca')
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate, r, drop_rate, Attention='ca')
			out_channels *= drop_rate
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
class SENet_Ca(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=1):
		super(SENet_Ca, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate, Attention='ca')] if blocks != 0 else []
		for _ in range(blocks - 2):
			layer1 = eval(block_type)(out_channels, out_channels, r, drop_rate, Attention='ca')
			layers.append(layer1)
		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

class SENet_Sa(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=1):
		super(SENet_Sa, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate, Attention='sa')] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, drop_rate, Attention='sa')
			layers.append(layer1)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

# _Ablation_CASA/SACA
class CASABlock(nn.Module):
	def __init__(self, channel, reduction=16, kernel_size=7):
		super().__init__()
		self.ca=ChannelAttention(channel=channel,reduction=reduction)
		self.sa=SpatialAttention(kernel_size=kernel_size)
	def forward(self, x):
		residual = x
		out = x * self.ca(x)
		out = out * self.sa(out)
		return out + residual
class SACABlock(nn.Module):
	def __init__(self, channel, reduction=16, kernel_size=7):
		super().__init__()
		self.sa = SpatialAttention(kernel_size=kernel_size)
		self.ca=ChannelAttention(channel=channel,reduction=reduction)
	def forward(self, x):
		residual = x
		out = x * self.sa(x)
		out = out * self.ca(out)
		return out + residual

class BottleneckBlock_Ablation(nn.Module):
	def __init__(self, in_channels, out_channels, r, drop_rate, Attention='se'):
		super(BottleneckBlock_Ablation, self).__init__()

		self.downsample = None
		if (in_channels != out_channels):
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
						  stride=drop_rate, bias=False),
				nn.BatchNorm2d(out_channels)
			)

		self.left = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
					  stride=drop_rate, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
		)


		if Attention=="ca":
			self.a =ChannelAttention(out_channels, reduction=r)
		if Attention=="caa":
			self.a =ChannelAttention_Ablation(out_channels, reduction=r)
		if Attention=='sa':
			self.a = SpatialAttention(kernel_size=21)
		if Attention=='se':
			self.a = nn.Sequential(
				nn.AdaptiveAvgPool2d((1, 1)),
				nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_channels=out_channels // r, out_channels=out_channels, kernel_size=1, bias=False),
				nn.Sigmoid()
			)
		if Attention=="casa":
			self.a =CASABlock(out_channels, reduction=r, kernel_size=21)
		if Attention=="saca":
			self.a =SACABlock(out_channels, reduction=r, kernel_size=21)
	def forward(self, x):
		identity = x
		x = self.left(x)
		scale = self.a(x)
		x = x * scale

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = F.relu(x)
		return x

class SENet_decoder_Ablation_CASA(nn.Module):
	'''
	ResNet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock_Ablation", r=8, drop_rate=2):
		super(SENet_decoder_Ablation_CASA, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, 1, Attention='casa')] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, 1, Attention='casa')
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate, r, drop_rate, Attention='casa')
			out_channels *= drop_rate
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
class SENet_Ca_Ablation_CASA(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock_Ablation", r=8, drop_rate=1):
		super(SENet_Ca_Ablation_CASA, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate, Attention='casa')] if blocks != 0 else []
		for _ in range(blocks - 2):
			layer1 = eval(block_type)(out_channels, out_channels, r, drop_rate, Attention='casa')
			layers.append(layer1)
		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
class SENet_Sa_Ablation_CASA(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock_Ablation", r=8, drop_rate=1):
		super(SENet_Sa_Ablation_CASA, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate, Attention='casa')] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, drop_rate, Attention='casa')
			layers.append(layer1)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


class SENet_decoder_Ablation_SACA(nn.Module):
	'''
	ResNet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock_Ablation", r=8, drop_rate=2):
		super(SENet_decoder_Ablation_SACA, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, 1, Attention='saca')] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, 1, Attention='saca')
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate, r, drop_rate, Attention='saca')
			out_channels *= drop_rate
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
class SENet_Ca_Ablation_SACA(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock_Ablation", r=8, drop_rate=1):
		super(SENet_Ca_Ablation_SACA, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate, Attention='saca')] if blocks != 0 else []
		for _ in range(blocks - 2):
			layer1 = eval(block_type)(out_channels, out_channels, r, drop_rate, Attention='saca')
			layers.append(layer1)
		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
class SENet_Sa_Ablation_SACA(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock_Ablation", r=8, drop_rate=1):
		super(SENet_Sa_Ablation_SACA, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate, Attention='saca')] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, drop_rate, Attention='saca')
			layers.append(layer1)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


# _Ablation_CA
class SENet_decoder_Ablation_CAA(nn.Module):
	'''
	ResNet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock_Ablation", r=8, drop_rate=2):
		super(SENet_decoder_Ablation_CAA, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, 1, Attention='caa')] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, 1, Attention='caa')
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate, r, drop_rate, Attention='caa')
			out_channels *= drop_rate
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
class SENet_Ca_Ablation_CAA(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock_Ablation", r=8, drop_rate=1):
		super(SENet_Ca_Ablation_CAA, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate, Attention='caa')] if blocks != 0 else []
		for _ in range(blocks - 2):
			layer1 = eval(block_type)(out_channels, out_channels, r, drop_rate, Attention='caa')
			layers.append(layer1)
		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


# Ablation		SE

class SENet_decoder_SE(nn.Module):
	'''
	ResNet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=2):
		super(SENet_decoder_SE, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, 1, Attention='se')] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, 1, Attention='se')
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate, r, drop_rate, Attention='se')
			out_channels *= drop_rate
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

class SENet_Ca_SE(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=1):
		super(SENet_Ca_SE, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate, Attention='se')] if blocks != 0 else []
		for _ in range(blocks - 2):
			layer1 = eval(block_type)(out_channels, out_channels, r, drop_rate, Attention='se')
			layers.append(layer1)
		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

class SENet_Sa_SE(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=1):
		super(SENet_Sa_SE, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate, Attention='se')] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, drop_rate, Attention='se')
			layers.append(layer1)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

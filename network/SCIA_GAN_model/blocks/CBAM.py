import numpy as np
import torch
from torch import nn
from torch.nn import init



# class ChannelAttention(nn.Module):
#     def __init__(self,channel,reduction=16):
#         super().__init__()
#         self.maxpool=nn.AdaptiveMaxPool2d(1)
#         self.avgpool=nn.AdaptiveAvgPool2d(1)
#         self.se=nn.Sequential(
#             nn.Conv2d(channel,channel//reduction,1,bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel//reduction,channel,1,bias=False)
#         )
#         self.sigmoid=nn.Sigmoid()
#
#     def forward(self, x) :
#         max_result=self.maxpool(x)
#         avg_result=self.avgpool(x)
#         max_out=self.se(max_result)
#         avg_out=self.se(avg_result)
#         output=self.sigmoid(max_out+avg_out)
#         return output
#
# class SpatialAttention(nn.Module):
#     def __init__(self,kernel_size=7):
#         super().__init__()
#         self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
#         self.sigmoid=nn.Sigmoid()
#
#     def forward(self, x) :
#         max_result,_=torch.max(x,dim=1,keepdim=True)
#         avg_result=torch.mean(x,dim=1,keepdim=True)
#         result=torch.cat([max_result,avg_result],1)
#         output=self.conv(result)
#         output=self.sigmoid(output)
#         return output

class ChannelAttention(nn.Module):
	def __init__(self, channel, reduction=16):
		super().__init__()
		self.maxpool = nn.AdaptiveMaxPool2d(1)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.se = nn.Sequential(
			nn.Conv2d(channel, channel // reduction, 1, bias=False),
			nn.BatchNorm2d(channel // reduction),
			nn.LeakyReLU(),
			nn.Conv2d(channel // reduction, channel, 1, bias=False),
			nn.BatchNorm2d(channel),
			nn.LeakyReLU(),
		)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		max_result = self.maxpool(x)
		avg_result = self.avgpool(x)
		max_out = self.se(max_result)
		avg_out = self.se(avg_result)
		output = self.sigmoid(max_out + avg_out)
		return output
class SpatialAttention(nn.Module):
	def __init__(self, kernel_size=7):
		super().__init__()
		self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
		self.bn = nn.BatchNorm2d(1)
		self.relu = nn.LeakyReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		max_result, _ = torch.max(x, dim=1, keepdim=True)
		avg_result = torch.mean(x, dim=1, keepdim=True)
		result = torch.cat([max_result, avg_result], 1)
		output = self.relu(self.bn(self.conv(result)))
		output = self.sigmoid(output)
		return output

class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


if __name__ == '__main__':
    x = torch.randn(1, 64, 128, 128)  # 2-1
    model = CBAMBlock(channel=64,reduction=16,kernel_size=49)  # FMRDBplus_out:50756 ResidualDenseBlock_out:84380
    # model = ResidualDenseBlock_out(64,64) #FMRDBplus_out:50756 ResidualDenseBlock_out:84380
    output = model(x)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(output.shape)

    
import torch.nn as nn
import torch

class ResidualDenseBlock_out(nn.Module):
	def __init__(self, in_channels, out_channels, bias=False):
		super(ResidualDenseBlock_out, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, 32, 1, 1, bias=bias),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(inplace=True))
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels, 32, 3, 1, 1, bias=bias),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(inplace=True))
		#nn.Conv2d(in_channels, 32, 3, 1, 1, bias=bias)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels + 32, 32, 3, 1, 1, bias=bias),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(inplace=True))
		#nn.Conv2d(in_channels + 32, 32, 3, 1, 1, bias=bias)
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels + 2 * 32, 32, 3, 1, 1, bias=bias),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(inplace=True))
		#nn.Conv2d(in_channels + 2 * 32, 32, 3, 1, 1, bias=bias)
		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels + 3 * 32, 32, 3, 1, 1, bias=bias),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(inplace=True))
		#nn.Conv2d(in_channels + 3 * 32, 32, 3, 1, 1, bias=bias)
		self.conv5 = nn.Conv2d(in_channels + 4 * 32, out_channels, 3, 1, 1, bias=bias)
		# self.lrelu = nn.LeakyReLU(inplace=True)
	def forward(self, x):
		# x1 = self.lrelu(self.conv1(x))
        # x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
		x1 = self.conv1(x)
		x2 = self.conv2(torch.cat((x, x1), 1))
		x2 = x2 + self.conv(x)
		x3 = self.conv3(torch.cat((x, x1, x2), 1))
		x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
		x4 = x2 + x4
		x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
		return x5

class ResidualDenseBlock_outS(nn.Module):
	def __init__(self, in_channels, out_channels, bias=False):
		super(ResidualDenseBlock_outS, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, 64, 1, 1, bias=bias),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(inplace=True))
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels, 64, 3, 1, 1, bias=bias),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(inplace=True))
		#nn.Conv2d(in_channels, 32, 3, 1, 1, bias=bias)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels + 64, 64, 3, 1, 1, bias=bias),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(inplace=True))
		#nn.Conv2d(in_channels + 32, 32, 3, 1, 1, bias=bias)
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels + 2 * 64, in_channels, 3, 1, 1, bias=bias),
			nn.BatchNorm2d(in_channels),
			nn.LeakyReLU(inplace=True))
		#nn.Conv2d(in_channels + 2 * 32, 32, 3, 1, 1, bias=bias)
		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels, 32, 3, 4, 1, bias=bias),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(inplace=True))
		self.conv5 = nn.Sequential(
			nn.Conv2d(32, out_channels, 3, 4, 1, bias=bias),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(inplace=True))
		# self.lrelu = nn.LeakyReLU(inplace=True)
	def forward(self, x):
		x1 = self.conv1(x)
		x2 = self.conv2(torch.cat((x, x1), 1))
		x2 = x2 + self.conv(x)
		x3 = self.conv3(torch.cat((x, x1, x2), 1))
		x4 = self.conv4(x3)
		x5 = self.conv5(x4)
		return x5


class Denseblock(nn.Module):
	def __init__(self, in_channels, out_channels, bias=False):
		super(Denseblock, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, 64, 1, 1, bias=bias),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(inplace=True))
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels, 64, 3, 1, 1, bias=bias),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(inplace=True))
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels + 64, 64, 3, 1, 1, bias=bias),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(inplace=True))
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels + 2 * 64, 64, 3, 1, 1, bias=bias),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(inplace=True))
		self.conv4 = nn.Sequential(
			nn.Conv2d(64, out_channels, 3, 1, 1, bias=bias),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(inplace=True)
			# nn.Softmax(dim=1)
		)
	def forward(self, x):
		x1 = self.conv1(x)
		x2 = self.conv2(torch.cat((x, x1), 1))
		x2 = x2 + self.conv(x)
		x3 = self.conv3(torch.cat((x, x1, x2), 1))
		x4 = self.conv4(x3)
		return x4

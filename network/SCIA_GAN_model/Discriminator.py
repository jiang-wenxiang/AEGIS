from . import *


class Discriminator(nn.Module):
	'''
	Adversary to discriminate the cover image and the encoded image
	'''

	def __init__(self, blocks=4, channels=64):
		super(Discriminator, self).__init__()

		self.layers = nn.Sequential(
			ConvNet(3, channels, blocks),
			nn.AdaptiveAvgPool2d(output_size=(1, 1))
		)

		self.linear = nn.Linear(channels, 1)

	def forward(self, image):
		x = self.layers(image)
		x.squeeze_(3).squeeze_(2)
		x = self.linear(x)
		return x
class Discriminator_Init(nn.Module):
	'''
	Adversary to discriminate the cover image and the encoded image
	'''

	def __init__(self, blocks=3, channels=64):
		super(Discriminator_Init, self).__init__()

		self.layers = nn.Sequential(
			ConvNet(3, channels, blocks),
			nn.AdaptiveAvgPool2d(output_size=(1, 1))
		)
		self.linear = nn.Linear(channels, 1)
		# self.sigmoid = torch.nn.Sigmoid()

	def forward(self, image):
		x = self.layers(image)
		x.squeeze_(3).squeeze_(2)
		x = self.linear(x)
		# x = self.sigmoid(x)
		return x

from . import *
from .Encoder_MP import Encoder_Init, Encoder_Init_CASA, Encoder_Init_SACA, Encoder_Init_CA, Encoder_Init_CAA, Encoder_Diffusion, Encoder_Diffusion1, Encoder_Init1
from .Decoder import Decoder_Init, Decoder_Init_CASA, Decoder_Init_SACA, Decoder_Init_CA, Decoder_Init_CAA, Decoder_Diffusion, Decoder_Diffusion1, Decoder_Init1
from ..Noise import Noise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import skimage
# class EncoderDecoder(nn.Module):
# 	def __init__(self, H, W, message_length, noise_layers):
# 		super(EncoderDecoder, self).__init__()
# 		self.encoder = Encoder_Init(H, W, message_length)
# 		self.noise = Noise(noise_layers)
# 		self.decoder = Decoder_Init(H, W, message_length)
# 	def forward(self, image, message):
# 		# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 		# cover_ybr = skimage.color.rgb2ycbcr(image.permute(0, 2, 3, 1).cpu().detach().numpy())
# 		# cover_ybr = torch.from_numpy(cover_ybr / 255.0).permute(0, 3, 1, 2)
# 		# encoded_image = self.encoder(cover_ybr.to(device), message.to(device))
# 		# stego_rgb = skimage.color.ycbcr2rgb((encoded_image * 255.0).permute(0, 2, 3, 1).to(device).data.cpu().numpy())
# 		# stego_rgb = torch.from_numpy(stego_rgb).permute(0, 3, 1, 2)
# 		# noised_image = self.noise([stego_rgb, image])
# 		# noised_image = skimage.color.rgb2ycbcr(noised_image.permute(0, 2, 3, 1).cpu().detach().numpy())
# 		# noised_image = torch.from_numpy(noised_image / 255.0).permute(0, 3, 1, 2)
# 		# decoded_message = self.decoder(noised_image.to(device))
#
# 		encoded_image = self.encoder(image, message)
# 		noised_image = self.noise([encoded_image, image])
# 		# noised_image = (noised_image - image) * 1.5 + image
# 		decoded_message = self.decoder(noised_image)
# 		return encoded_image, noised_image, decoded_message#, decoded_blank_message

# class EncoderDecoder(nn.Module):
# 	def __init__(self, H, W, message_length, noise_layers):
# 		super(EncoderDecoder, self).__init__()
# 		self.encoder = Encoder_Init(H, W, message_length)
# 		self.noise = Noise(noise_layers)
# 		self.decoder = Decoder_Init(H, W, message_length)
# 		self.dwt = common.DWT()
# 		self.iwt = common.IWT()
# 	def forward(self, image, message):
# 		image_input = self.dwt(image)
# 		encoded_image_output = self.encoder(image_input, message)
# 		encoded_image = self.iwt(encoded_image_output)
# 		noised_image = self.noise([encoded_image, image]) #encoded_image, image
# 		noised_image_input = self.dwt(noised_image)
# 		decoded_message = self.decoder(noised_image_input)
# 		return encoded_image, noised_image, decoded_message, image_input, encoded_image_output

# class EncoderDecoder_Diffusion(nn.Module):
# 	'''
# 	A Sequential of Encoder_MP-Noise-Decoder
# 	'''
#
# 	def __init__(self, H, W, message_length, noise_layers):
# 		super(EncoderDecoder_Diffusion, self).__init__()
#
# 		self.encoder = Encoder_Diffusion(H, W, message_length)
# 		self.noise = Noise(noise_layers)
# 		self.decoder = Decoder_Diffusion(H, W, message_length)
# 		# print(self.decoder)
# 		self.dwt = common.DWT()
# 		self.iwt = common.IWT()
# 	def forward(self, image, message):
# 		image_input = self.dwt(image)
# 		encoded_image_output = self.encoder(image_input, message)
# 		encoded_image = self.iwt(encoded_image_output)
# 		noised_image = self.noise([encoded_image, image]) #encoded_image, image
# 		noised_image_input = self.dwt(noised_image)
# 		decoded_message = self.decoder(noised_image_input)
#
# 		return encoded_image, noised_image, decoded_message, image_input, encoded_image_output


class EncoderDecoder_New(nn.Module):
	def __init__(self, H, W, message_length, noise_layers):
		super(EncoderDecoder_New, self).__init__()
		self.encoder = Encoder_Init(H, W, message_length)
		self.noise = Noise(noise_layers)
		self.decoder = Decoder_Init(H, W, message_length)
		# self.dwt = common.DWT()
		# self.iwt = common.IWT()
	def forward(self, image, message):
		# image_input = self.dwt(image)
		encoded_image = self.encoder(image, message)
		# encoded_image = self.iwt(encoded_image_output)
		encoded_image = (encoded_image-image) * 1.508 + image
		noised_image = self.noise([encoded_image, image]) #encoded_image, image
		# noised_image_input = self.dwt(noised_image)
		decoded_message = self.decoder(noised_image)
		return encoded_image, noised_image, decoded_message#, image_input, encoded_image_output

class EncoderDecoder_New1(nn.Module):
	def __init__(self, H, W, message_length, noise_layers):
		super(EncoderDecoder_New1, self).__init__()
		self.encoder = Encoder_Init1(H, W, message_length)
		self.noise = Noise(noise_layers)
		self.decoder = Decoder_Init1(H, W, message_length)
		# self.dwt = common.DWT()
		# self.iwt = common.IWT()
	def forward(self, image, message):
		# image_input = self.dwt(image)
		encoded_image = self.encoder(image, message)
		# encoded_image = self.iwt(encoded_image_output)
		encoded_image = (encoded_image-image) * 1.508 + image
		noised_image = self.noise([encoded_image, image]) #encoded_image, image
		# noised_image_input = self.dwt(noised_image)
		decoded_message = self.decoder(noised_image)
		return encoded_image, noised_image, decoded_message#, image_input, encoded_image_output

class EncoderDecoder_Diffusion_New(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, H, W, message_length, noise_layers):
		super(EncoderDecoder_Diffusion_New, self).__init__()

		self.encoder = Encoder_Diffusion(H, W, message_length)
		self.decoder = Decoder_Diffusion(H, W, message_length)
		self.noise = Noise(noise_layers)


		# print(self.decoder)
	def forward(self, image, message):
		encoded_image = self.encoder(image, message)
		noised_image = self.noise([encoded_image, image])
		decoded_message = self.decoder(noised_image)
		return encoded_image, noised_image, decoded_message
class EncoderDecoder_Diffusion_New1(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, H, W, message_length, noise_layers):
		super(EncoderDecoder_Diffusion_New1, self).__init__()

		self.encoder = Encoder_Diffusion1(H, W, message_length)
		self.decoder = Decoder_Diffusion1(H, W, message_length)
		self.noise = Noise(noise_layers)


		# print(self.decoder)
	def forward(self, image, message):
		encoded_image = self.encoder(image, message)
		noised_image = self.noise([encoded_image, image])
		decoded_message = self.decoder(noised_image)
		return encoded_image, noised_image, decoded_message

# from .model_ARWGAN.encoder import Encoder as Encoder_ARWGAN
# from .model_ARWGAN.decoder import Decoder as Decoder_ARWGAN
#
# class EncoderDecoder_ARWGAN(nn.Module):
# 	def __init__(self, noise_layers):
# 		super(EncoderDecoder_ARWGAN, self).__init__()
#
# 		self.encoder = Encoder_ARWGAN()
# 		self.noise = Noise(noise_layers)
# 		self.decoder = Decoder_ARWGAN()
#
# 	def forward(self, image, message):
# 		encoded_image = self.encoder(image, message)
# 		noised_image = self.noise([encoded_image, image])
# 		decoded_message = self.decoder(noised_image)
# 		return encoded_image, noised_image, decoded_message




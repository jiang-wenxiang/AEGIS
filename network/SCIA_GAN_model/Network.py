from .Encoder_MP_Decoder import *
from .Discriminator import Discriminator, Discriminator_Init
import torchvision
from network import pytorch_ssim
import network.Unet_common as common
from network.attention.MODELS.model_resnet import *
from Evaluation import ssim_loss, calculate_psnr, decoded_message_error_rate_batch
import skimage
from network.ybr import ycbcr_images
from network.dwt_dct import DWTForward_Init
# import config as c
# import network.Unet_common as common

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
class GWLoss(nn.Module):
    """Gradient Weighted Loss"""
    def __init__(self, reduction='mean'):
        super(GWLoss, self).__init__()
        self.w = 4
        self.reduction = reduction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float)
        self.weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=sobel_y, requires_grad=False)
    def forward(self, x1, x2):
        b, c, w, h = x1.shape
        weight_x = self.weight_x.expand(c, 1, 3, 3).type_as(x1)
        weight_y = self.weight_y.expand(c, 1, 3, 3).type_as(x1)
        Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        # loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
        loss = (1 + self.w * dx) * (1 + self.w * dy) * torch.abs(x1 - x2)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)

dwt = common.DWT()
iwt = common.IWT()

print('No A loss				#############   MSE HFW')  #  you *0.5
# class Network_New:  # No
# 	def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
# 				 only_decoder=False):
# 	# def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
# 	# 			 only_decoder=False):
# 		# device
# 		self.device = device
#
# 		# network
# 		if not with_diffusion:
# 			self.encoder_decoder = EncoderDecoder_New(H, W, message_length, noise_layers).to(device)
# 		else:
# 			print('Diffusion')
# 			self.encoder_decoder = EncoderDecoder_Diffusion_New(H, W, message_length, noise_layers).to(device)
#
# 		self.discriminator = Discriminator_Init().to(device)
#
# 		self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
# 		self.discriminator = torch.nn.DataParallel(self.discriminator)
#
# 		if only_decoder:
# 			for p in self.encoder_decoder.module.encoder.parameters():
# 				p.requires_grad = False
#
# 		# mark "cover" as 1, "encoded" as 0
# 		self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
# 		self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)
#
# 		# optimizer
# 		# print(lr)
# 		self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
# 		self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
# 		# self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
# 		# self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
# 		# self.weight_scheduler_encoder_decoder = \
# 		# 	torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_encoder_decoder, mode='min',factor= 0.98,patience=625,verbose=False,threshold=0.0001,threshold_mode='rel',cooldown=50,min_lr=1e-10,eps=1e-8)#
# 		#torch.optim.lr_scheduler.StepLR(self.opt_encoder_decoder, weight_step, gamma=gamma)
# 		# self.weight_scheduler_discriminator = torch.optim.lr_scheduler.StepLR(self.opt_discriminator, weight_step, gamma=gamma)
#
# 		# loss function
# 		self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
# 		self.criterion_MSE = nn.MSELoss().to(device)
#
# 		self.dwt = common.DWT()
# 		self.iwt = common.IWT()
#
# 		self.L1_Charbonnier = L1_Charbonnier_loss()
# 		self.GW = GWLoss()
# 		self.SSIM = pytorch_ssim.SSIM(window_size=11)
#
# 		# weight of encoder-decoder loss
# 		self.discriminator_weight = 0.0001
# 		self.encoder_weight = 1 #1 9
# 		self.decoder_weight = 25#20 15
#
# 	def train(self, images: torch.Tensor, messages: torch.Tensor):
# 		self.encoder_decoder.train()
# 		self.discriminator.train()
# 		with torch.enable_grad():
# 			# use device to9 compute
# 			# print(self.opt_discriminator.param_groups[0]['lr'])
#
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
#
# 			'''
# 			train discriminator
# 			'''
# 			self.opt_discriminator.zero_grad()
# 			# RAW : target label for image should be "cover"(1)
# 			d_label_cover = self.discriminator(images)
# 			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
# 			d_cover_loss.backward()
# 			# GAN : target label for encoded image should be "encoded"(0)
# 			d_label_encoded = self.discriminator(encoded_images.detach())
# 			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
# 			# loss_discriminator
# 			d_encoded_loss.backward()
#
# 			# loss_discriminator = 0.5 * d_cover_loss + 0.5 * d_encoded_loss
# 			# loss_discriminator.backward()
# 			self.opt_discriminator.step()
# 			# self.weight_scheduler_discriminator.step(loss_discriminator)
# 			'''
# 			train encoder and decoder
# 			'''
# 			self.opt_encoder_decoder.zero_grad()
# 			# GAN : target label for encoded image should be "cover"(0)
# 			g_label_decoded = self.discriminator(encoded_images)
# 			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
# 			# print(f" {d_encoded_loss} | {d_cover_loss} | {g_loss_on_discriminator}")
# 			# RAW : the encoded image should be similar to cover image
# 			loss_mse_rgb = self.criterion_MSE(encoded_images, images)
# 			# loss_ssim_rgb = 0.1 * (1 - self.SSIM(encoded_images, images))
# 			# loss_gw_low = 0.0001 * self.GW(self.dwt(encoded_images).narrow(1, 0, 3), self.dwt(images).narrow(1, 0, 3))
# 			loss_charbonnier_high = 2 * self.criterion_MSE(self.dwt(encoded_images).narrow(1, 3, 9),self.dwt(images).narrow(1, 3, 9))
# 			g_loss_on_encoder = (loss_mse_rgb + loss_charbonnier_high)# * 0.5
# 			# print(f" {g_loss_on_encoder} | {loss_mse_rgb} | {loss_ssim_rgb} | {loss_gw_low} | SSIM ={loss_charbonnier_high}")
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)
#
# 			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
# 					 self.decoder_weight * g_loss_on_decoder
# 			# print(f' {self.discriminator_weight * g_loss_on_discriminator} | {self.encoder_weight * g_loss_on_encoder} | {self.decoder_weight * g_loss_on_decoder}')
# 			g_loss.backward()
# 			self.opt_encoder_decoder.step()
# 			# self.weight_scheduler_encoder_decoder.step(g_loss)
#
# 			# psnr	ssim	decoded message error rate
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1),keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": g_loss_on_discriminator,
# 			"g_loss_on_encoder": g_loss_on_encoder,
# 			"g_loss_on_decoder": g_loss_on_decoder,
# 			"d_cover_loss": d_cover_loss,
# 			"d_encoded_loss": d_encoded_loss
# 		}
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def train_only_decoder(self, images: torch.Tensor, messages: torch.Tensor):
# 		# for p in self.encoder_decoder.module.encoder.parameters():
# 		# 	p.requires_grad = False
# 		# # for p in self.encoder_decoder.module.noise.parameters():
# 		# # 	p.requires_grad = False
# 		# self.encoder_decoder.module.encoder.eval()
# 		# # self.encoder_decoder.module.noise.eval()
# 		# self.encoder_decoder.module.decoder.train()
# 		self.encoder_decoder.train()
#
# 		with torch.enable_grad():
# 			# use device to compute
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
# 			'''
# 			train encoder and decoder
# 			'''
# 			# for name, param in self.encoder_decoder.module.encoder.named_parameters():
# 			# 	print(name, param.requires_grad)
# 			# for name, param in self.encoder_decoder.module.decoder.named_parameters():
# 			# 	print(name, param.requires_grad)
# 			self.opt_encoder_decoder.zero_grad()
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss = self.criterion_MSE(decoded_messages, messages)
#
# 			# print(g_loss)
# 			g_loss.backward()
# 			self.opt_encoder_decoder.step()
#
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 		'''
# 		decoded message error rate
# 		'''
# 		error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 		error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": 0.,
# 			"g_loss_on_encoder": 0.,
# 			"g_loss_on_decoder": g_loss,
# 			"d_cover_loss": 0.,
# 			"d_encoded_loss": 0.
# 		}
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def validation(self, images: torch.Tensor, messages: torch.Tensor):
# 		self.encoder_decoder.eval()
# 		self.discriminator.eval()
#
# 		with torch.no_grad():
# 			# use device to compute
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			# encoded_images, noised_images, decoded_messages, image_input, encoded_image_output = self.encoder_decoder(images, messages)
#
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
#
# 			'''
# 			validate discriminator
# 			'''
# 			# RAW : target label for image should be "cover"(1)
# 			d_label_cover = self.discriminator(images)
# 			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
#
# 			# GAN : target label for encoded image should be "encoded"(0)
# 			d_label_encoded = self.discriminator(encoded_images.detach())
# 			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
#
# 			'''
# 			validate encoder and decoder
# 			'''
#
# 			# GAN : target label for encoded image should be "cover"(0)
# 			g_label_decoded = self.discriminator(encoded_images)
# 			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
#
# 			# RAW : the encoded image should be similar to cover image
# 			loss_mse_rgb = self.criterion_MSE(encoded_images, images)
# 			# loss_ssim_rgb = 0.1 * (1 - self.SSIM(encoded_images, images))
# 			# loss_gw_low = 0.0001 * self.GW(self.dwt(encoded_images).narrow(1, 0, 3), self.dwt(images).narrow(1, 0, 3))
# 			loss_charbonnier_high = 2 * self.criterion_MSE(self.dwt(encoded_images).narrow(1, 3, 9), self.dwt(images).narrow(1, 3, 9))
# 			g_loss_on_encoder = (loss_mse_rgb + loss_charbonnier_high)# * 0.5
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)
#
# 			# full loss
# 			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
# 					 self.decoder_weight * g_loss_on_decoder
#
# 			# psnr	ssim	decoded message error rate
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": g_loss_on_discriminator,
# 			"g_loss_on_encoder": g_loss_on_encoder,
# 			"g_loss_on_decoder": g_loss_on_decoder,
# 			"d_cover_loss": d_cover_loss,
# 			"d_encoded_loss": d_encoded_loss
# 		}
#
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def decoded_message_error_rate(self, message, decoded_message):
# 		length = message.shape[0]
#
# 		message = message.gt(0.5)
# 		decoded_message = decoded_message.gt(0.5)
#
# 		error_rate = float(sum(message != decoded_message)) / length
# 		return error_rate
#
# 	def decoded_message_error_rate_batch(self, messages, decoded_messages):
# 		error_rate = 0.0
# 		batch_size = len(messages)
# 		for i in range(batch_size):
# 			error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
# 		print(decoded_messages[0])
# 		error_rate /= batch_size
# 		return error_rate
#
# 	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
# 		torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
# 		torch.save(self.discriminator.module.state_dict(), path_discriminator)
#
# 	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
# 		self.load_model_ed(path_encoder_decoder)
# 		self.load_model_dis(path_discriminator)
#
# 	def load_model_ed(self, path_encoder_decoder: str):
# 		self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))
#
# 	def load_model_dis(self, path_discriminator: str):
# 		self.discriminator.module.load_state_dict(torch.load(path_discriminator))

print('No A loss				#############   MSE HFW')  #  wu *0.5
# class Network_New:		#No Attention loss
# 	def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
# 				 only_decoder=False):
# 	# def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
# 	# 			 only_decoder=False):
# 		# device
# 		self.device = device
#
# 		# network
# 		if not with_diffusion:
# 			self.encoder_decoder = EncoderDecoder_New(H, W, message_length, noise_layers).to(device)
# 		else:
# 			print('Diffusion')
# 			self.encoder_decoder = EncoderDecoder_Diffusion_New(H, W, message_length, noise_layers).to(device)
#
# 		self.discriminator = Discriminator_Init().to(device)
#
# 		self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
# 		self.discriminator = torch.nn.DataParallel(self.discriminator)
#
# 		if only_decoder:
# 			for p in self.encoder_decoder.module.encoder.parameters():
# 				p.requires_grad = False
#
# 		# mark "cover" as 1, "encoded" as 0
# 		self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
# 		self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)
#
# 		# optimizer
# 		# print(lr)
# 		self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
# 		self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
# 		# self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
# 		# self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
# 		# self.weight_scheduler_encoder_decoder = \
# 		# 	torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_encoder_decoder, mode='min',factor= 0.98,patience=625,verbose=False,threshold=0.0001,threshold_mode='rel',cooldown=50,min_lr=1e-10,eps=1e-8)#
# 		#torch.optim.lr_scheduler.StepLR(self.opt_encoder_decoder, weight_step, gamma=gamma)
# 		# self.weight_scheduler_discriminator = torch.optim.lr_scheduler.StepLR(self.opt_discriminator, weight_step, gamma=gamma)
#
# 		# loss function
# 		self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
# 		self.criterion_MSE = nn.MSELoss().to(device)
#
# 		self.dwt = common.DWT()
# 		self.iwt = common.IWT()
#
# 		self.L1_Charbonnier = L1_Charbonnier_loss()
# 		self.GW = GWLoss()
# 		self.SSIM = pytorch_ssim.SSIM(window_size=11)
#
# 		# weight of encoder-decoder loss
# 		self.discriminator_weight = 0.0001
# 		self.encoder_weight = 1 #1 9
# 		self.decoder_weight = 25#20 15
#
# 	def train(self, images: torch.Tensor, messages: torch.Tensor):
# 		self.encoder_decoder.train()
# 		self.discriminator.train()
# 		with torch.enable_grad():
# 			# use device to9 compute
# 			# print(self.opt_discriminator.param_groups[0]['lr'])
#
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
#
# 			'''
# 			train discriminator
# 			'''
# 			self.opt_discriminator.zero_grad()
# 			# RAW : target label for image should be "cover"(1)
# 			d_label_cover = self.discriminator(images)
# 			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
# 			d_cover_loss.backward()
# 			# GAN : target label for encoded image should be "encoded"(0)
# 			d_label_encoded = self.discriminator(encoded_images.detach())
# 			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
# 			# loss_discriminator
# 			d_encoded_loss.backward()
#
# 			# loss_discriminator = 0.5 * d_cover_loss + 0.5 * d_encoded_loss
# 			# loss_discriminator.backward()
# 			self.opt_discriminator.step()
# 			# self.weight_scheduler_discriminator.step(loss_discriminator)
# 			'''
# 			train encoder and decoder
# 			'''
# 			self.opt_encoder_decoder.zero_grad()
# 			# GAN : target label for encoded image should be "cover"(0)
# 			g_label_decoded = self.discriminator(encoded_images)
# 			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
# 			# print(f" {d_encoded_loss} | {d_cover_loss} | {g_loss_on_discriminator}")
# 			# RAW : the encoded image should be similar to cover image
# 			loss_mse_rgb = self.criterion_MSE(encoded_images, images)
# 			# loss_ssim_rgb = 0.1 * (1 - self.SSIM(encoded_images, images))
# 			# loss_gw_low = 0.0001 * self.GW(self.dwt(encoded_images).narrow(1, 0, 3), self.dwt(images).narrow(1, 0, 3))
# 			loss_charbonnier_high = 2 * self.criterion_MSE(self.dwt(encoded_images).narrow(1, 3, 9),self.dwt(images).narrow(1, 3, 9))
# 			g_loss_on_encoder = (loss_mse_rgb + loss_charbonnier_high)# * 0.5
# 			# print(f" {g_loss_on_encoder} | {loss_mse_rgb} | {loss_ssim_rgb} | {loss_gw_low} | SSIM ={loss_charbonnier_high}")
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)
#
# 			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
# 					 self.decoder_weight * g_loss_on_decoder
# 			# print(f' {self.discriminator_weight * g_loss_on_discriminator} | {self.encoder_weight * g_loss_on_encoder} | {self.decoder_weight * g_loss_on_decoder}')
# 			g_loss.backward()
# 			self.opt_encoder_decoder.step()
# 			# self.weight_scheduler_encoder_decoder.step(g_loss)
#
# 			# psnr	ssim	decoded message error rate
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1),keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": g_loss_on_discriminator,
# 			"g_loss_on_encoder": g_loss_on_encoder,
# 			"g_loss_on_decoder": g_loss_on_decoder,
# 			"d_cover_loss": d_cover_loss,
# 			"d_encoded_loss": d_encoded_loss
# 		}
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def train_only_decoder(self, images: torch.Tensor, messages: torch.Tensor):
# 		# for p in self.encoder_decoder.module.encoder.parameters():
# 		# 	p.requires_grad = False
# 		# # for p in self.encoder_decoder.module.noise.parameters():
# 		# # 	p.requires_grad = False
# 		# self.encoder_decoder.module.encoder.eval()
# 		# # self.encoder_decoder.module.noise.eval()
# 		# self.encoder_decoder.module.decoder.train()
# 		self.encoder_decoder.train()
#
# 		with torch.enable_grad():
# 			# use device to compute
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
# 			'''
# 			train encoder and decoder
# 			'''
# 			# for name, param in self.encoder_decoder.module.encoder.named_parameters():
# 			# 	print(name, param.requires_grad)
# 			# for name, param in self.encoder_decoder.module.decoder.named_parameters():
# 			# 	print(name, param.requires_grad)
# 			self.opt_encoder_decoder.zero_grad()
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss = self.criterion_MSE(decoded_messages, messages)
#
# 			# print(g_loss)
# 			g_loss.backward()
# 			self.opt_encoder_decoder.step()
#
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 		'''
# 		decoded message error rate
# 		'''
# 		error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 		error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": 0.,
# 			"g_loss_on_encoder": 0.,
# 			"g_loss_on_decoder": g_loss,
# 			"d_cover_loss": 0.,
# 			"d_encoded_loss": 0.
# 		}
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def validation(self, images: torch.Tensor, messages: torch.Tensor):
# 		self.encoder_decoder.eval()
# 		self.discriminator.eval()
#
# 		with torch.no_grad():
# 			# use device to compute
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			# encoded_images, noised_images, decoded_messages, image_input, encoded_image_output = self.encoder_decoder(images, messages)
#
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
#
# 			'''
# 			validate discriminator
# 			'''
# 			# RAW : target label for image should be "cover"(1)
# 			d_label_cover = self.discriminator(images)
# 			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
#
# 			# GAN : target label for encoded image should be "encoded"(0)
# 			d_label_encoded = self.discriminator(encoded_images.detach())
# 			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
#
# 			'''
# 			validate encoder and decoder
# 			'''
#
# 			# GAN : target label for encoded image should be "cover"(0)
# 			g_label_decoded = self.discriminator(encoded_images)
# 			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
#
# 			# RAW : the encoded image should be similar to cover image
# 			loss_mse_rgb = self.criterion_MSE(encoded_images, images)
# 			# loss_ssim_rgb = 0.1 * (1 - self.SSIM(encoded_images, images))
# 			# loss_gw_low = 0.0001 * self.GW(self.dwt(encoded_images).narrow(1, 0, 3), self.dwt(images).narrow(1, 0, 3))
# 			loss_charbonnier_high = 2 * self.criterion_MSE(self.dwt(encoded_images).narrow(1, 3, 9), self.dwt(images).narrow(1, 3, 9))
# 			g_loss_on_encoder = (loss_mse_rgb + loss_charbonnier_high)# * 0.5
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)
#
# 			# full loss
# 			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
# 					 self.decoder_weight * g_loss_on_decoder
#
# 			# psnr	ssim	decoded message error rate
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": g_loss_on_discriminator,
# 			"g_loss_on_encoder": g_loss_on_encoder,
# 			"g_loss_on_decoder": g_loss_on_decoder,
# 			"d_cover_loss": d_cover_loss,
# 			"d_encoded_loss": d_encoded_loss
# 		}
#
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def decoded_message_error_rate(self, message, decoded_message):
# 		length = message.shape[0]
#
# 		message = message.gt(0.5)
# 		decoded_message = decoded_message.gt(0.5)
#
# 		error_rate = float(sum(message != decoded_message)) / length
# 		return error_rate
#
# 	def decoded_message_error_rate_batch(self, messages, decoded_messages):
# 		error_rate = 0.0
# 		batch_size = len(messages)
# 		for i in range(batch_size):
# 			error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
# 		print(decoded_messages[0])
# 		error_rate /= batch_size
# 		return error_rate
#
# 	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
# 		torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
# 		torch.save(self.discriminator.module.state_dict(), path_discriminator)
#
# 	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
# 		self.load_model_ed(path_encoder_decoder)
# 		self.load_model_dis(path_discriminator)
#
# 	def load_model_ed(self, path_encoder_decoder: str):
# 		self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))
#
# 	def load_model_dis(self, path_discriminator: str):
# 		self.discriminator.module.load_state_dict(torch.load(path_discriminator))

print('No Hfw loss    No A loss	#############  	MSE')  #  wu *0.5
# class Network_New: #No Hfw loss    No A loss
# 	def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
# 				 only_decoder=False):
# 	# def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
# 	# 			 only_decoder=False):
# 		# device
# 		self.device = device
#
# 		# network
# 		if not with_diffusion:
# 			self.encoder_decoder = EncoderDecoder_New(H, W, message_length, noise_layers).to(device)
# 		else:
# 			print('Diffusion')
# 			self.encoder_decoder = EncoderDecoder_Diffusion_New(H, W, message_length, noise_layers).to(device)
#
# 		self.discriminator = Discriminator_Init().to(device)
#
# 		self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
# 		self.discriminator = torch.nn.DataParallel(self.discriminator)
#
# 		if only_decoder:
# 			for p in self.encoder_decoder.module.encoder.parameters():
# 				p.requires_grad = False
#
# 		# mark "cover" as 1, "encoded" as 0
# 		self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
# 		self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)
#
# 		# optimizer
# 		# print(lr)
# 		self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
# 		self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
# 		# self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
# 		# self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
# 		# self.weight_scheduler_encoder_decoder = \
# 		# 	torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_encoder_decoder, mode='min',factor= 0.98,patience=625,verbose=False,threshold=0.0001,threshold_mode='rel',cooldown=50,min_lr=1e-10,eps=1e-8)#
# 		#torch.optim.lr_scheduler.StepLR(self.opt_encoder_decoder, weight_step, gamma=gamma)
# 		# self.weight_scheduler_discriminator = torch.optim.lr_scheduler.StepLR(self.opt_discriminator, weight_step, gamma=gamma)
#
# 		# loss function
# 		self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
# 		self.criterion_MSE = nn.MSELoss().to(device)
#
# 		self.dwt = common.DWT()
# 		self.iwt = common.IWT()
#
# 		self.L1_Charbonnier = L1_Charbonnier_loss()
# 		self.GW = GWLoss()
# 		self.SSIM = pytorch_ssim.SSIM(window_size=11)
#
# 		# weight of encoder-decoder loss
# 		self.discriminator_weight = 0.0001
# 		self.encoder_weight = 1 #1 9
# 		self.decoder_weight = 15#20 15
#
# 	def train(self, images: torch.Tensor, messages: torch.Tensor):
# 		self.encoder_decoder.train()
# 		self.discriminator.train()
# 		with torch.enable_grad():
# 			# use device to9 compute
# 			# print(self.opt_discriminator.param_groups[0]['lr'])
#
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
#
# 			'''
# 			train discriminator
# 			'''
# 			self.opt_discriminator.zero_grad()
# 			# RAW : target label for image should be "cover"(1)
# 			d_label_cover = self.discriminator(images)
# 			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
# 			d_cover_loss.backward()
# 			# GAN : target label for encoded image should be "encoded"(0)
# 			d_label_encoded = self.discriminator(encoded_images.detach())
# 			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
# 			# loss_discriminator
# 			d_encoded_loss.backward()
#
# 			# loss_discriminator = 0.5 * d_cover_loss + 0.5 * d_encoded_loss
# 			# loss_discriminator.backward()
# 			self.opt_discriminator.step()
# 			# self.weight_scheduler_discriminator.step(loss_discriminator)
# 			'''
# 			train encoder and decoder
# 			'''
# 			self.opt_encoder_decoder.zero_grad()
# 			# GAN : target label for encoded image should be "cover"(0)
# 			g_label_decoded = self.discriminator(encoded_images)
# 			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
# 			# print(f" {d_encoded_loss} | {d_cover_loss} | {g_loss_on_discriminator}")
# 			# RAW : the encoded image should be similar to cover image
# 			loss_mse_rgb = self.criterion_MSE(encoded_images, images)
# 			# loss_ssim_rgb = 0.1 * (1 - self.SSIM(encoded_images, images))
# 			# loss_gw_low = 0.0001 * self.GW(self.dwt(encoded_images).narrow(1, 0, 3), self.dwt(images).narrow(1, 0, 3))
#
# 			#loss_charbonnier_high = 2 * self.criterion_MSE(self.dwt(encoded_images).narrow(1, 3, 9),self.dwt(images).narrow(1, 3, 9))
# 			g_loss_on_encoder = loss_mse_rgb #(loss_mse_rgb + loss_charbonnier_high) * 0.5
# 			# print(f" {g_loss_on_encoder} | {loss_mse_rgb} | {loss_ssim_rgb} | {loss_gw_low} | SSIM ={loss_charbonnier_high}")
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)
#
# 			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
# 					 self.decoder_weight * g_loss_on_decoder
# 			# print(f' {self.discriminator_weight * g_loss_on_discriminator} | {self.encoder_weight * g_loss_on_encoder} | {self.decoder_weight * g_loss_on_decoder}')
# 			g_loss.backward()
# 			self.opt_encoder_decoder.step()
# 			# self.weight_scheduler_encoder_decoder.step(g_loss)
#
# 			# psnr	ssim	decoded message error rate
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1),keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": g_loss_on_discriminator,
# 			"g_loss_on_encoder": g_loss_on_encoder,
# 			"g_loss_on_decoder": g_loss_on_decoder,
# 			"d_cover_loss": d_cover_loss,
# 			"d_encoded_loss": d_encoded_loss
# 		}
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def train_only_decoder(self, images: torch.Tensor, messages: torch.Tensor):
# 		# for p in self.encoder_decoder.module.encoder.parameters():
# 		# 	p.requires_grad = False
# 		# # for p in self.encoder_decoder.module.noise.parameters():
# 		# # 	p.requires_grad = False
# 		# self.encoder_decoder.module.encoder.eval()
# 		# # self.encoder_decoder.module.noise.eval()
# 		# self.encoder_decoder.module.decoder.train()
# 		self.encoder_decoder.train()
#
# 		with torch.enable_grad():
# 			# use device to compute
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
# 			'''
# 			train encoder and decoder
# 			'''
# 			# for name, param in self.encoder_decoder.module.encoder.named_parameters():
# 			# 	print(name, param.requires_grad)
# 			# for name, param in self.encoder_decoder.module.decoder.named_parameters():
# 			# 	print(name, param.requires_grad)
# 			self.opt_encoder_decoder.zero_grad()
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss = self.criterion_MSE(decoded_messages, messages)
#
# 			# print(g_loss)
# 			g_loss.backward()
# 			self.opt_encoder_decoder.step()
#
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 		'''
# 		decoded message error rate
# 		'''
# 		error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 		error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": 0.,
# 			"g_loss_on_encoder": 0.,
# 			"g_loss_on_decoder": g_loss,
# 			"d_cover_loss": 0.,
# 			"d_encoded_loss": 0.
# 		}
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def validation(self, images: torch.Tensor, messages: torch.Tensor):
# 		self.encoder_decoder.eval()
# 		self.discriminator.eval()
#
# 		with torch.no_grad():
# 			# use device to compute
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			# encoded_images, noised_images, decoded_messages, image_input, encoded_image_output = self.encoder_decoder(images, messages)
#
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
#
# 			'''
# 			validate discriminator
# 			'''
# 			# RAW : target label for image should be "cover"(1)
# 			d_label_cover = self.discriminator(images)
# 			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
#
# 			# GAN : target label for encoded image should be "encoded"(0)
# 			d_label_encoded = self.discriminator(encoded_images.detach())
# 			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
#
# 			'''
# 			validate encoder and decoder
# 			'''
#
# 			# GAN : target label for encoded image should be "cover"(0)
# 			g_label_decoded = self.discriminator(encoded_images)
# 			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
#
# 			# RAW : the encoded image should be similar to cover image
# 			loss_mse_rgb = self.criterion_MSE(encoded_images, images)
# 			# loss_ssim_rgb = 0.1 * (1 - self.SSIM(encoded_images, images))
# 			# loss_gw_low = 0.0001 * self.GW(self.dwt(encoded_images).narrow(1, 0, 3), self.dwt(images).narrow(1, 0, 3))
#
# 			# loss_charbonnier_high = 2 * self.criterion_MSE(self.dwt(encoded_images).narrow(1, 3, 9), self.dwt(images).narrow(1, 3, 9))
# 			g_loss_on_encoder = loss_mse_rgb #(loss_mse_rgb + loss_charbonnier_high) * 0.5
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)
#
# 			# full loss
# 			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
# 					 self.decoder_weight * g_loss_on_decoder
#
# 			# psnr	ssim	decoded message error rate
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": g_loss_on_discriminator,
# 			"g_loss_on_encoder": g_loss_on_encoder,
# 			"g_loss_on_decoder": g_loss_on_decoder,
# 			"d_cover_loss": d_cover_loss,
# 			"d_encoded_loss": d_encoded_loss
# 		}
#
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def decoded_message_error_rate(self, message, decoded_message):
# 		length = message.shape[0]
#
# 		message = message.gt(0.5)
# 		decoded_message = decoded_message.gt(0.5)
#
# 		error_rate = float(sum(message != decoded_message)) / length
# 		return error_rate
#
# 	def decoded_message_error_rate_batch(self, messages, decoded_messages):
# 		error_rate = 0.0
# 		batch_size = len(messages)
# 		for i in range(batch_size):
# 			error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
# 		print(decoded_messages[0])
# 		error_rate /= batch_size
# 		return error_rate
#
# 	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
# 		torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
# 		torch.save(self.discriminator.module.state_dict(), path_discriminator)
#
# 	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
# 		self.load_model_ed(path_encoder_decoder)
# 		self.load_model_dis(path_discriminator)
#
# 	def load_model_ed(self, path_encoder_decoder: str):
# 		self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))
#
# 	def load_model_dis(self, path_discriminator: str):
# 		self.discriminator.module.load_state_dict(torch.load(path_discriminator))#

print('Yes Hfw loss    Yes A loss#############   MSE HFW')  #  wu *0.5
MSE=True
class Network_New:
	def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
				 only_decoder=False):
	# def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
	# 			 only_decoder=False):
		# device
		self.device = device

		# network
		if not with_diffusion:
			self.encoder_decoder = EncoderDecoder_New(H, W, message_length, noise_layers).to(device)
			# self.encoder_decoder = EncoderDecoder_New1(H, W, message_length, noise_layers).to(device)
		else:
			print('Diffusion_Dense')
			self.encoder_decoder = EncoderDecoder_Diffusion_New(H, W, message_length, noise_layers).to(device)
			# self.encoder_decoder = EncoderDecoder_Diffusion_New1(H, W, message_length, noise_layers).to(device)#120
		self.discriminator = Discriminator_Init().to(device)

		self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
		self.discriminator = torch.nn.DataParallel(self.discriminator)
		self.Attention_model = ResidualNet('ImageNet', 50, 1000, 'CBAM')
		self.Attention_optimizer = torch.optim.SGD(self.Attention_model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
		self.Attention_model = torch.nn.DataParallel(self.Attention_model, device_ids=list(range(1)))
		self.Attention_model = self.Attention_model.cuda()
		self.Attention_checkpoint = torch.load('/media/WRL/D/WRL/MBRS-main/network/attention/checkpoints/RESNET50_CBAM_new_name_wrap.pth')
		self.Attention_model.load_state_dict(self.Attention_checkpoint['state_dict'])
		# self.Attention_optimizer.load_state_dict(self.Attention_checkpoint['optimizer'])
		for p in self.Attention_model.parameters():
			p.requires_grad = False

		if only_decoder:
			for p in self.encoder_decoder.module.encoder.parameters():
				p.requires_grad = False

		# mark "cover" as 1, "encoded" as 0
		self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
		self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

		# optimizer
		# print(lr)
		self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
		self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
		# self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
		# self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
		# self.weight_scheduler_encoder_decoder = \
		# 	torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_encoder_decoder, mode='min',factor= 0.98,patience=625,verbose=False,threshold=0.0001,threshold_mode='rel',cooldown=50,min_lr=1e-10,eps=1e-8)#
		#torch.optim.lr_scheduler.StepLR(self.opt_encoder_decoder, weight_step, gamma=gamma)
		# self.weight_scheduler_discriminator = torch.optim.lr_scheduler.StepLR(self.opt_discriminator, weight_step, gamma=gamma)

		# loss function
		self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
		self.criterion_MSE = nn.MSELoss().to(device)

		self.dwt = common.DWT()
		self.iwt = common.IWT()

		self.L1_Charbonnier = L1_Charbonnier_loss()
		self.GW = GWLoss()
		self.SSIM = pytorch_ssim.SSIM(window_size=11)

		# weight of encoder-decoder loss
		self.discriminator_weight = 0.0001
		self.encoder_weight = 1 #1 9
		self.decoder_weight = 5# 5				com:10    JPEG 5
		# self.encoder_weight = 0.75  # 1 9
		# self.decoder_weight = 1.5  # 15
	def train(self, images: torch.Tensor, messages: torch.Tensor):
		self.encoder_decoder.train()
		self.discriminator.train()
		self.Attention_model.eval()
		with torch.enable_grad():
			# use device to9 compute
			# print(self.opt_discriminator.param_groups[0]['lr'])

			images, messages = images.to(self.device), messages.to(self.device)
			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

			'''
			train discriminator
			'''
			self.opt_discriminator.zero_grad()
			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(images)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
			d_cover_loss.backward()
			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
			# loss_discriminator
			d_encoded_loss.backward()

			# loss_discriminator = 0.5 * d_cover_loss + 0.5 * d_encoded_loss
			# loss_discriminator.backward()
			self.opt_discriminator.step()
			# self.weight_scheduler_discriminator.step(loss_discriminator)
			'''
			train encoder and decoder
			'''
			self.opt_encoder_decoder.zero_grad()
			# GAN : target label for encoded image should be "cover"(0)
			g_label_decoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
			# print(f" {d_encoded_loss} | {d_cover_loss} | {g_loss_on_discriminator}")
			# RAW : the encoded image should be similar to cover image
			loss_mse_rgb = self.criterion_MSE(encoded_images, images)
			# loss_ssim_rgb = 0.1 * (1 - self.SSIM(encoded_images, images))

			# loss_gw_low = 0.0001 * self.GW(self.dwt(encoded_images).narrow(1, 0, 3), self.dwt(images).narrow(1, 0, 3))
			loss_charbonnier_high = 2 * self.criterion_MSE(self.dwt(encoded_images).narrow(1, 3, 9),self.dwt(images).narrow(1, 3, 9))

			target_feature = self.Attention_model(images)
			output_feature = self.Attention_model(encoded_images)
			loss_ssim_Attention = self.criterion_MSE(output_feature, target_feature)
			g_loss_on_encoder = loss_mse_rgb + loss_charbonnier_high + 0.5 * loss_ssim_Attention



			# print(f" {g_loss_on_encoder} | {loss_mse_rgb} | {loss_ssim_Attention} | SSIM ={loss_charbonnier_high}")
			# RESULT : the decoded message should be similar to the raw message
			if MSE == True:
				g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)   #初始用的MSE
			else:
				g_loss_on_decoder = self.criterion_BCE(decoded_messages, messages)
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
					 self.decoder_weight * g_loss_on_decoder
			# print(f' {self.discriminator_weight * g_loss_on_discriminator} | {self.encoder_weight * g_loss_on_encoder} | {self.decoder_weight * g_loss_on_decoder}')
			g_loss.backward()
			self.opt_encoder_decoder.step()
			# self.weight_scheduler_encoder_decoder.step(g_loss)

			# psnr	ssim	decoded message error rate
			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
			# psnr = calculate_psnr(encoded_images.detach(), images)
			# ssim = ssim_loss(encoded_images.detach(), images)
			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1),keepdim=True) / len(messages[0])).float()

		result = {
			"error_rate_corrective": error_rate_corrective,
			"error_rate_refine": error_rate_refine,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder": g_loss_on_encoder,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}
		return result, (images, encoded_images, noised_images, messages, decoded_messages)

	def train_only_decoder(self, images: torch.Tensor, messages: torch.Tensor):
		# for p in self.encoder_decoder.module.encoder.parameters():
		# 	p.requires_grad = False

		# self.encoder_decoder.module.encoder.eval()
		# # self.encoder_decoder.module.noise.eval()
		# self.encoder_decoder.module.decoder.train()
		self.encoder_decoder.train()

		with torch.enable_grad():
			# use device to compute
			images, messages = images.to(self.device), messages.to(self.device)
			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
			'''
			train encoder and decoder
			'''
			# for name, param in self.encoder_decoder.module.encoder.named_parameters():
			# 	print(name, param.requires_grad)
			# for name, param in self.encoder_decoder.module.decoder.named_parameters():
			# 	print(name, param.requires_grad)
			self.opt_encoder_decoder.zero_grad()
			# RESULT : the decoded message should be similar to the raw message
			g_loss = self.criterion_MSE(decoded_messages, messages)

			# print(g_loss)
			g_loss.backward()
			self.opt_encoder_decoder.step()

			# psnr = calculate_psnr(encoded_images.detach(), images)
			# ssim = ssim_loss(encoded_images.detach(), images)
			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
		'''
		decoded message error rate
		'''
		error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
		error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()

		result = {
			"error_rate_corrective": error_rate_corrective,
			"error_rate_refine": error_rate_refine,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
			"g_loss_on_discriminator": 0.,
			"g_loss_on_encoder": 0.,
			"g_loss_on_decoder": g_loss,
			"d_cover_loss": 0.,
			"d_encoded_loss": 0.
		}
		return result, (images, encoded_images, noised_images, messages, decoded_messages)

	def validation(self, images: torch.Tensor, messages: torch.Tensor):
		self.encoder_decoder.eval()
		self.discriminator.eval()

		self.Attention_model.eval()
		with torch.no_grad():
			# use device to compute
			images, messages = images.to(self.device), messages.to(self.device)
			# encoded_images, noised_images, decoded_messages, image_input, encoded_image_output = self.encoder_decoder(images, messages)

			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

			'''
			validate discriminator
			'''
			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(images)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])

			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])

			'''
			validate encoder and decoder
			'''

			# GAN : target label for encoded image should be "cover"(0)
			g_label_decoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

			# RAW : the encoded image should be similar to cover image
			loss_mse_rgb = self.criterion_MSE(encoded_images, images)
			# loss_ssim_rgb = 0.1 * (1 - self.SSIM(encoded_images, images))

			# loss_gw_low = 0.0001 * self.GW(self.dwt(encoded_images).narrow(1, 0, 3), self.dwt(images).narrow(1, 0, 3))
			loss_charbonnier_high = 2 * self.criterion_MSE(self.dwt(encoded_images).narrow(1, 3, 9), self.dwt(images).narrow(1, 3, 9))

			target_feature = self.Attention_model(images)
			output_feature = self.Attention_model(encoded_images)
			loss_ssim_Attention = self.criterion_MSE(output_feature, target_feature)
			g_loss_on_encoder = loss_mse_rgb + loss_charbonnier_high + 0.5 * loss_ssim_Attention
			# RESULT : the decoded message should be similar to the raw message
			if MSE == True:
				g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)   #初始用的MSE
			else:
				g_loss_on_decoder = self.criterion_BCE(decoded_messages, messages)
			# full loss
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
					 self.decoder_weight * g_loss_on_decoder

			# psnr	ssim	decoded message error rate
			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
			# psnr = calculate_psnr(encoded_images.detach(), images)
			# ssim = ssim_loss(encoded_images.detach(), images)
			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()

		result = {
			"error_rate_corrective": error_rate_corrective,
			"error_rate_refine": error_rate_refine,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder": g_loss_on_encoder,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}

		return result, (images, encoded_images, noised_images, messages, decoded_messages)

	def decoded_message_error_rate(self, message, decoded_message):
		length = message.shape[0]

		message = message.gt(0.5)
		decoded_message = decoded_message.gt(0.5)

		error_rate = float(sum(message != decoded_message)) / length
		return error_rate
	def decoded_message_error_rate_batch(self, messages, decoded_messages):
		error_rate = 0.0
		batch_size = len(messages)
		for i in range(batch_size):
			error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
		print(decoded_messages[0])
		error_rate /= batch_size
		return error_rate
	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
		torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
		torch.save(self.discriminator.module.state_dict(), path_discriminator)
	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
		self.load_model_ed(path_encoder_decoder)
		self.load_model_dis(path_discriminator)
	def load_model_ed(self, path_encoder_decoder: str):
		self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))
	def load_model_dis(self, path_discriminator: str):
		self.discriminator.module.load_state_dict(torch.load(path_discriminator))

from network.model_ARWGAN.discriminator import Discriminator as Discriminator_ARWGAN
from network.model_ARWGAN.SSIM import SSIM as SSIM_ARWGAN
from network.model_ARWGAN.vgg_loss import VGGLoss
class Network_ARWGAN:
	def __init__(self, noise_layers, device, use_vgg=False):
	# def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
	# 			 only_decoder=False):
		# device
		self.device = device

		self.encoder_decoder = EncoderDecoder_ARWGAN(noise_layers).to(device)
		self.discriminator = Discriminator_ARWGAN().to(device)

		self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
		self.discriminator = torch.nn.DataParallel(self.discriminator)

		self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
		self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

		if use_vgg:
			self.vgg_loss = VGGLoss(3, 1, False)
			self.vgg_loss.to(device)
		else:
			self.vgg_loss = None
		# self.config = configuration
		self.device = device
		self.ssim_loss = SSIM_ARWGAN()
		self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
		self.mse_loss = nn.MSELoss().to(device)
		self.adversarial_weight = 0.001
		self.mse_weight = 0.7
		self.ssim_weight = 0.1
		self.decode_weight = 1.5

		# Defined the labels used for training the discriminator/adversarial loss
		self.cover_label = 1
		self.encoded_label = 0

	def train_on_batch(self, batch: list):
		images, messages = batch
		batch_size = images.shape[0]

		self.encoder_decoder.train()
		self.discriminator.train()
		with torch.enable_grad():
			# ---------------- Train the discriminator -----------------------------
			self.optimizer_discrim.zero_grad()
			# train on cover

			d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
			d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
			g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

			d_on_cover = self.discriminator(images)
			d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, (d_target_label_cover).float())
			d_loss_on_cover.backward()

			# train on fake
			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
			d_on_encoded = self.discriminator(encoded_images.detach())
			d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, (d_target_label_encoded).float())

			d_loss_on_encoded.backward()
			self.optimizer_discrim.step()

			# --------------Train the generator (encoder-decoder) ---------------------
			self.optimizer_enc_dec.zero_grad()
			d_on_encoded_for_enc = self.discriminator(encoded_images)
			g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc.float(), g_target_label_encoded.float())

			if self.vgg_loss == None:
				g_loss_enc = self.mse_loss(encoded_images, images)

			else:
				vgg_on_cov = self.vgg_loss(images)
				vgg_on_enc = self.vgg_loss(encoded_images)
				g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)
			g_loss_enc_ssim = self.ssim_loss(encoded_images, images)
			g_loss_dec = self.mse_loss(decoded_messages, messages)
			g_loss = self.adversarial_weight * g_loss_adv + self.ssim_weight * (
					1 - g_loss_enc_ssim) + self.mse_weight * (g_loss_enc) + self.decode_weight * (g_loss_dec)

			g_loss.backward()

			self.optimizer_enc_dec.step()
			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
			error_rate_refine = (
						torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(
					messages[0])).float()

		result = {
			"error_rate_corrective": error_rate_corrective,
			"error_rate_refine": error_rate_refine.item(),
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
		}
		return result, (images, encoded_images, noised_images, messages, decoded_messages)

	def validate_on_batch(self, batch: list):
		images, messages = batch

		batch_size = images.shape[0]

		self.encoder_decoder.eval()
		self.discriminator.eval()
		with torch.no_grad():

			d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
			d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
			g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

			d_on_cover = self.discriminator(images)
			d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())

			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

			d_on_encoded = self.discriminator(encoded_images)
			d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())

			d_on_encoded_for_enc = self.discriminator(encoded_images)
			g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc.float(), g_target_label_encoded.float())
			g_loss_enc_ssim = self.ssim_loss(images, encoded_images)
			if self.vgg_loss is None:
				g_loss_enc = self.mse_loss(encoded_images, images)
			else:
				vgg_on_cov = self.vgg_loss(images)
				vgg_on_enc = self.vgg_loss(encoded_images)
				g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)
			g_loss_dec = self.mse_loss(decoded_messages, messages)
			g_loss = self.adversarial_weight * g_loss_adv + self.ssim_weight * (
					1 - g_loss_enc_ssim) + self.mse_weight * (g_loss_enc) + self.decode_weight * (g_loss_dec)

			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()

		result = {
			"error_rate_corrective": error_rate_corrective,
			"error_rate_refine": error_rate_refine.item(),
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
		}
		return result, (images, encoded_images, noised_images, messages, decoded_messages)

	def to_stirng(self):
		return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))

	def decoded_message_error_rate(self, message, decoded_message):
		length = message.shape[0]

		message = message.gt(0.5)
		decoded_message = decoded_message.gt(0.5)

		error_rate = float(sum(message != decoded_message)) / length
		return error_rate

	def decoded_message_error_rate_batch(self, messages, decoded_messages):
		error_rate = 0.0
		batch_size = len(messages)
		for i in range(batch_size):
			error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
		print(decoded_messages[0])
		error_rate /= batch_size
		return error_rate

	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
		torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
		torch.save(self.discriminator.module.state_dict(), path_discriminator)

	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
		self.load_model_ed(path_encoder_decoder)
		self.load_model_dis(path_discriminator)

	def load_model_ed(self, path_encoder_decoder: str):
		self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))

	def load_model_dis(self, path_discriminator: str):
		self.discriminator.module.load_state_dict(torch.load(path_discriminator))

print('No Hfw loss    Yes A loss#############   MSE HFW')  #  wu *0.5
# class Network_New:
# 	def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
# 				 only_decoder=False):
# 	# def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
# 	# 			 only_decoder=False):
# 		# device
# 		self.device = device
#
# 		# network
# 		if not with_diffusion:
# 			self.encoder_decoder = EncoderDecoder_New(H, W, message_length, noise_layers).to(device)
# 		else:
# 			print('Diffusion')
# 			self.encoder_decoder = EncoderDecoder_Diffusion_New(H, W, message_length, noise_layers).to(device)
#
# 		self.discriminator = Discriminator_Init().to(device)
#
# 		self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
# 		self.discriminator = torch.nn.DataParallel(self.discriminator)
#
# 		self.Attention_model = ResidualNet('ImageNet', 50, 1000, 'CBAM')
# 		self.Attention_optimizer = torch.optim.SGD(self.Attention_model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
# 		self.Attention_model = torch.nn.DataParallel(self.Attention_model, device_ids=list(range(1)))
# 		self.Attention_model = self.Attention_model.cuda()
# 		self.Attention_checkpoint = torch.load('/media/WRL/D/WRL/attention-module-master/checkpoints/RESNET50_CBAM_new_name_wrap.pth')
# 		self.Attention_model.load_state_dict(self.Attention_checkpoint['state_dict'])
# 		# self.Attention_optimizer.load_state_dict(self.Attention_checkpoint['optimizer'])
#
# 		if only_decoder:
# 			for p in self.encoder_decoder.module.encoder.parameters():
# 				p.requires_grad = False
#
# 		# mark "cover" as 1, "encoded" as 0
# 		self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
# 		self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)
#
# 		# optimizer
# 		# print(lr)
# 		self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
# 		self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
# 		# self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
# 		# self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
# 		# self.weight_scheduler_encoder_decoder = \
# 		# 	torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_encoder_decoder, mode='min',factor= 0.98,patience=625,verbose=False,threshold=0.0001,threshold_mode='rel',cooldown=50,min_lr=1e-10,eps=1e-8)#
# 		#torch.optim.lr_scheduler.StepLR(self.opt_encoder_decoder, weight_step, gamma=gamma)
# 		# self.weight_scheduler_discriminator = torch.optim.lr_scheduler.StepLR(self.opt_discriminator, weight_step, gamma=gamma)
#
# 		# loss function
# 		self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
# 		self.criterion_MSE = nn.MSELoss().to(device)
#
# 		self.dwt = common.DWT()
# 		self.iwt = common.IWT()
#
# 		self.L1_Charbonnier = L1_Charbonnier_loss()
# 		self.GW = GWLoss()
# 		self.SSIM = pytorch_ssim.SSIM(window_size=11)
#
# 		# weight of encoder-decoder loss
# 		self.discriminator_weight = 0.0001
# 		self.encoder_weight = 1 #1 9
# 		self.decoder_weight = 15#20 15
#
# 	def train(self, images: torch.Tensor, messages: torch.Tensor):
# 		self.encoder_decoder.train()
# 		self.discriminator.train()
# 		self.Attention_model.eval()
# 		with torch.enable_grad():
# 			# use device to9 compute
# 			# print(self.opt_discriminator.param_groups[0]['lr'])
#
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
#
# 			'''
# 			train discriminator
# 			'''
# 			self.opt_discriminator.zero_grad()
# 			# RAW : target label for image should be "cover"(1)
# 			d_label_cover = self.discriminator(images)
# 			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
# 			d_cover_loss.backward()
# 			# GAN : target label for encoded image should be "encoded"(0)
# 			d_label_encoded = self.discriminator(encoded_images.detach())
# 			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
# 			# loss_discriminator
# 			d_encoded_loss.backward()
#
# 			# loss_discriminator = 0.5 * d_cover_loss + 0.5 * d_encoded_loss
# 			# loss_discriminator.backward()
# 			self.opt_discriminator.step()
# 			# self.weight_scheduler_discriminator.step(loss_discriminator)
# 			'''
# 			train encoder and decoder
# 			'''
# 			self.opt_encoder_decoder.zero_grad()
# 			# GAN : target label for encoded image should be "cover"(0)
# 			g_label_decoded = self.discriminator(encoded_images)
# 			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
# 			# print(f" {d_encoded_loss} | {d_cover_loss} | {g_loss_on_discriminator}")
# 			# RAW : the encoded image should be similar to cover image
# 			loss_mse_rgb = self.criterion_MSE(encoded_images, images)
# 			# loss_ssim_rgb = 0.1 * (1 - self.SSIM(encoded_images, images))
#
# 			target_feature = self.Attention_model(images)
# 			output_feature = self.Attention_model(encoded_images)
# 			loss_ssim_Attention = (1 - self.SSIM(output_feature, target_feature)) * 0.5
#
# 			# loss_gw_low = 0.0001 * self.GW(self.dwt(encoded_images).narrow(1, 0, 3), self.dwt(images).narrow(1, 0, 3))
# 			#loss_charbonnier_high = 2 * self.criterion_MSE(self.dwt(encoded_images).narrow(1, 3, 9),self.dwt(images).narrow(1, 3, 9))
# 			g_loss_on_encoder = (loss_mse_rgb + loss_ssim_Attention)# * 0.5
# 			# print(f" {g_loss_on_encoder} | {loss_mse_rgb} | {loss_ssim_Attention} | SSIM ={loss_charbonnier_high}")
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)
#
# 			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
# 					 self.decoder_weight * g_loss_on_decoder
# 			# print(f' {self.discriminator_weight * g_loss_on_discriminator} | {self.encoder_weight * g_loss_on_encoder} | {self.decoder_weight * g_loss_on_decoder}')
# 			g_loss.backward()
# 			self.opt_encoder_decoder.step()
# 			# self.weight_scheduler_encoder_decoder.step(g_loss)
#
# 			# psnr	ssim	decoded message error rate
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1),keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": g_loss_on_discriminator,
# 			"g_loss_on_encoder": g_loss_on_encoder,
# 			"g_loss_on_decoder": g_loss_on_decoder,
# 			"d_cover_loss": d_cover_loss,
# 			"d_encoded_loss": d_encoded_loss
# 		}
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def train_only_decoder(self, images: torch.Tensor, messages: torch.Tensor):
# 		# for p in self.encoder_decoder.module.encoder.parameters():
# 		# 	p.requires_grad = False
# 		# # for p in self.encoder_decoder.module.noise.parameters():
# 		# # 	p.requires_grad = False
# 		# self.encoder_decoder.module.encoder.eval()
# 		# # self.encoder_decoder.module.noise.eval()
# 		# self.encoder_decoder.module.decoder.train()
# 		self.encoder_decoder.train()
#
# 		with torch.enable_grad():
# 			# use device to compute
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
# 			'''
# 			train encoder and decoder
# 			'''
# 			# for name, param in self.encoder_decoder.module.encoder.named_parameters():
# 			# 	print(name, param.requires_grad)
# 			# for name, param in self.encoder_decoder.module.decoder.named_parameters():
# 			# 	print(name, param.requires_grad)
# 			self.opt_encoder_decoder.zero_grad()
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss = self.criterion_MSE(decoded_messages, messages)
#
# 			# print(g_loss)
# 			g_loss.backward()
# 			self.opt_encoder_decoder.step()
#
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 		'''
# 		decoded message error rate
# 		'''
# 		error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 		error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": 0.,
# 			"g_loss_on_encoder": 0.,
# 			"g_loss_on_decoder": g_loss,
# 			"d_cover_loss": 0.,
# 			"d_encoded_loss": 0.
# 		}
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def validation(self, images: torch.Tensor, messages: torch.Tensor):
# 		self.encoder_decoder.eval()
# 		self.discriminator.eval()
# 		self.Attention_model.eval()
# 		with torch.no_grad():
# 			# use device to compute
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			# encoded_images, noised_images, decoded_messages, image_input, encoded_image_output = self.encoder_decoder(images, messages)
#
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
#
# 			'''
# 			validate discriminator
# 			'''
# 			# RAW : target label for image should be "cover"(1)
# 			d_label_cover = self.discriminator(images)
# 			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
#
# 			# GAN : target label for encoded image should be "encoded"(0)
# 			d_label_encoded = self.discriminator(encoded_images.detach())
# 			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
#
# 			'''
# 			validate encoder and decoder
# 			'''
#
# 			# GAN : target label for encoded image should be "cover"(0)
# 			g_label_decoded = self.discriminator(encoded_images)
# 			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
#
# 			# RAW : the encoded image should be similar to cover image
# 			loss_mse_rgb = self.criterion_MSE(encoded_images, images)
# 			# loss_ssim_rgb = 0.1 * (1 - self.SSIM(encoded_images, images))
# 			target_feature = self.Attention_model(images)
# 			output_feature = self.Attention_model(encoded_images)
# 			loss_ssim_Attention = (1 - self.SSIM(output_feature, target_feature)) * 0.5
#
# 			# loss_gw_low = 0.0001 * self.GW(self.dwt(encoded_images).narrow(1, 0, 3), self.dwt(images).narrow(1, 0, 3))
# 			#loss_charbonnier_high = 2 * self.criterion_MSE(self.dwt(encoded_images).narrow(1, 3, 9), self.dwt(images).narrow(1, 3, 9))
# 			g_loss_on_encoder = (loss_mse_rgb + loss_ssim_Attention)# * 0.5
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)
#
# 			# full loss
# 			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
# 					 self.decoder_weight * g_loss_on_decoder
#
# 			# psnr	ssim	decoded message error rate
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": g_loss_on_discriminator,
# 			"g_loss_on_encoder": g_loss_on_encoder,
# 			"g_loss_on_decoder": g_loss_on_decoder,
# 			"d_cover_loss": d_cover_loss,
# 			"d_encoded_loss": d_encoded_loss
# 		}
#
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def decoded_message_error_rate(self, message, decoded_message):
# 		length = message.shape[0]
#
# 		message = message.gt(0.5)
# 		decoded_message = decoded_message.gt(0.5)
#
# 		error_rate = float(sum(message != decoded_message)) / length
# 		return error_rate
#
# 	def decoded_message_error_rate_batch(self, messages, decoded_messages):
# 		error_rate = 0.0
# 		batch_size = len(messages)
# 		for i in range(batch_size):
# 			error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
# 		print(decoded_messages[0])
# 		error_rate /= batch_size
# 		return error_rate
#
# 	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
# 		torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
# 		torch.save(self.discriminator.module.state_dict(), path_discriminator)
#
# 	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
# 		self.load_model_ed(path_encoder_decoder)
# 		self.load_model_dis(path_discriminator)
#
# 	def load_model_ed(self, path_encoder_decoder: str):
# 		self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))
#
# 	def load_model_dis(self, path_discriminator: str):
# 		self.discriminator.module.load_state_dict(torch.load(path_discriminator))

print('Yes Hfw loss    No A loss#############   MSE HFW')  #  wu *0.5
# class Network_New:
# 	def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
# 				 only_decoder=False):
# 	# def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
# 	# 			 only_decoder=False):
# 		# device
# 		self.device = device
#
# 		# network
# 		if not with_diffusion:
# 			self.encoder_decoder = EncoderDecoder_New(H, W, message_length, noise_layers).to(device)
# 		else:
# 			print('Diffusion')
# 			self.encoder_decoder = EncoderDecoder_Diffusion_New(H, W, message_length, noise_layers).to(device)
#
# 		self.discriminator = Discriminator_Init().to(device)
#
# 		self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
# 		self.discriminator = torch.nn.DataParallel(self.discriminator)
#
# 		self.Attention_model = ResidualNet('ImageNet', 50, 1000, 'CBAM')
# 		self.Attention_optimizer = torch.optim.SGD(self.Attention_model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
# 		self.Attention_model = torch.nn.DataParallel(self.Attention_model, device_ids=list(range(1)))
# 		self.Attention_model = self.Attention_model.cuda()
# 		self.Attention_checkpoint = torch.load('/media/WRL/D/WRL/attention-module-master/checkpoints/RESNET50_CBAM_new_name_wrap.pth')
# 		self.Attention_model.load_state_dict(self.Attention_checkpoint['state_dict'])
# 		# self.Attention_optimizer.load_state_dict(self.Attention_checkpoint['optimizer'])
#
# 		if only_decoder:
# 			for p in self.encoder_decoder.module.encoder.parameters():
# 				p.requires_grad = False
#
# 		# mark "cover" as 1, "encoded" as 0
# 		self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
# 		self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)
#
# 		# optimizer
# 		# print(lr)
# 		self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
# 		self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
# 		# self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
# 		# self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
# 		# self.weight_scheduler_encoder_decoder = \
# 		# 	torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_encoder_decoder, mode='min',factor= 0.98,patience=625,verbose=False,threshold=0.0001,threshold_mode='rel',cooldown=50,min_lr=1e-10,eps=1e-8)#
# 		#torch.optim.lr_scheduler.StepLR(self.opt_encoder_decoder, weight_step, gamma=gamma)
# 		# self.weight_scheduler_discriminator = torch.optim.lr_scheduler.StepLR(self.opt_discriminator, weight_step, gamma=gamma)
#
# 		# loss function
# 		self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
# 		self.criterion_MSE = nn.MSELoss().to(device)
#
# 		self.dwt = common.DWT()
# 		self.iwt = common.IWT()
#
# 		self.L1_Charbonnier = L1_Charbonnier_loss()
# 		self.GW = GWLoss()
# 		self.SSIM = pytorch_ssim.SSIM(window_size=11)
#
# 		# weight of encoder-decoder loss
# 		self.discriminator_weight = 0.0001
# 		self.encoder_weight = 1 #1 9
# 		self.decoder_weight = 15#20 15
#
# 	def train(self, images: torch.Tensor, messages: torch.Tensor):
# 		self.encoder_decoder.train()
# 		self.discriminator.train()
# 		self.Attention_model.eval()
# 		with torch.enable_grad():
# 			# use device to9 compute
# 			# print(self.opt_discriminator.param_groups[0]['lr'])
#
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
#
# 			'''
# 			train discriminator
# 			'''
# 			self.opt_discriminator.zero_grad()
# 			# RAW : target label for image should be "cover"(1)
# 			d_label_cover = self.discriminator(images)
# 			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
# 			d_cover_loss.backward()
# 			# GAN : target label for encoded image should be "encoded"(0)
# 			d_label_encoded = self.discriminator(encoded_images.detach())
# 			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
# 			# loss_discriminator
# 			d_encoded_loss.backward()
#
# 			# loss_discriminator = 0.5 * d_cover_loss + 0.5 * d_encoded_loss
# 			# loss_discriminator.backward()
# 			self.opt_discriminator.step()
# 			# self.weight_scheduler_discriminator.step(loss_discriminator)
# 			'''
# 			train encoder and decoder
# 			'''
# 			self.opt_encoder_decoder.zero_grad()
# 			# GAN : target label for encoded image should be "cover"(0)
# 			g_label_decoded = self.discriminator(encoded_images)
# 			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
# 			# print(f" {d_encoded_loss} | {d_cover_loss} | {g_loss_on_discriminator}")
# 			# RAW : the encoded image should be similar to cover image
# 			loss_mse_rgb = self.criterion_MSE(encoded_images, images)
# 			# loss_ssim_rgb = 0.1 * (1 - self.SSIM(encoded_images, images))
#
# 			# target_feature = self.Attention_model(images)
# 			# output_feature = self.Attention_model(encoded_images)
# 			# loss_ssim_Attention = (1 - self.SSIM(output_feature, target_feature)) * 0.5
#
# 			# loss_gw_low = 0.0001 * self.GW(self.dwt(encoded_images).narrow(1, 0, 3), self.dwt(images).narrow(1, 0, 3))
# 			loss_charbonnier_high = 2 * self.criterion_MSE(self.dwt(encoded_images).narrow(1, 3, 9),self.dwt(images).narrow(1, 3, 9))
# 			g_loss_on_encoder = (loss_mse_rgb + loss_charbonnier_high)# * 0.5
# 			# print(f" {g_loss_on_encoder} | {loss_mse_rgb} | {loss_ssim_Attention} | SSIM ={loss_charbonnier_high}")
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)
#
# 			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
# 					 self.decoder_weight * g_loss_on_decoder
# 			# print(f' {self.discriminator_weight * g_loss_on_discriminator} | {self.encoder_weight * g_loss_on_encoder} | {self.decoder_weight * g_loss_on_decoder}')
# 			g_loss.backward()
# 			self.opt_encoder_decoder.step()
# 			# self.weight_scheduler_encoder_decoder.step(g_loss)
#
# 			# psnr	ssim	decoded message error rate
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1),keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": g_loss_on_discriminator,
# 			"g_loss_on_encoder": g_loss_on_encoder,
# 			"g_loss_on_decoder": g_loss_on_decoder,
# 			"d_cover_loss": d_cover_loss,
# 			"d_encoded_loss": d_encoded_loss
# 		}
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def train_only_decoder(self, images: torch.Tensor, messages: torch.Tensor):
# 		# for p in self.encoder_decoder.module.encoder.parameters():
# 		# 	p.requires_grad = False
# 		# # for p in self.encoder_decoder.module.noise.parameters():
# 		# # 	p.requires_grad = False
# 		# self.encoder_decoder.module.encoder.eval()
# 		# # self.encoder_decoder.module.noise.eval()
# 		# self.encoder_decoder.module.decoder.train()
# 		self.encoder_decoder.train()
#
# 		with torch.enable_grad():
# 			# use device to compute
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
# 			'''
# 			train encoder and decoder
# 			'''
# 			# for name, param in self.encoder_decoder.module.encoder.named_parameters():
# 			# 	print(name, param.requires_grad)
# 			# for name, param in self.encoder_decoder.module.decoder.named_parameters():
# 			# 	print(name, param.requires_grad)
# 			self.opt_encoder_decoder.zero_grad()
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss = self.criterion_MSE(decoded_messages, messages)
#
# 			# print(g_loss)
# 			g_loss.backward()
# 			self.opt_encoder_decoder.step()
#
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 		'''
# 		decoded message error rate
# 		'''
# 		error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 		error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": 0.,
# 			"g_loss_on_encoder": 0.,
# 			"g_loss_on_decoder": g_loss,
# 			"d_cover_loss": 0.,
# 			"d_encoded_loss": 0.
# 		}
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def validation(self, images: torch.Tensor, messages: torch.Tensor):
# 		self.encoder_decoder.eval()
# 		self.discriminator.eval()
# 		self.Attention_model.eval()
# 		with torch.no_grad():
# 			# use device to compute
# 			images, messages = images.to(self.device), messages.to(self.device)
# 			# encoded_images, noised_images, decoded_messages, image_input, encoded_image_output = self.encoder_decoder(images, messages)
#
# 			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
#
# 			'''
# 			validate discriminator
# 			'''
# 			# RAW : target label for image should be "cover"(1)
# 			d_label_cover = self.discriminator(images)
# 			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
#
# 			# GAN : target label for encoded image should be "encoded"(0)
# 			d_label_encoded = self.discriminator(encoded_images.detach())
# 			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
#
# 			'''
# 			validate encoder and decoder
# 			'''
#
# 			# GAN : target label for encoded image should be "cover"(0)
# 			g_label_decoded = self.discriminator(encoded_images)
# 			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
#
# 			# RAW : the encoded image should be similar to cover image
# 			loss_mse_rgb = self.criterion_MSE(encoded_images, images)
# 			# loss_ssim_rgb = 0.1 * (1 - self.SSIM(encoded_images, images))
# 			# target_feature = self.Attention_model(images)
# 			# output_feature = self.Attention_model(encoded_images)
# 			# loss_ssim_Attention = (1 - self.SSIM(output_feature, target_feature)) * 0.5
#
# 			# loss_gw_low = 0.0001 * self.GW(self.dwt(encoded_images).narrow(1, 0, 3), self.dwt(images).narrow(1, 0, 3))
# 			loss_charbonnier_high = 2 * self.criterion_MSE(self.dwt(encoded_images).narrow(1, 3, 9), self.dwt(images).narrow(1, 3, 9))
# 			g_loss_on_encoder = (loss_mse_rgb + loss_charbonnier_high)# * 0.5
# 			# RESULT : the decoded message should be similar to the raw message
# 			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)
#
# 			# full loss
# 			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
# 					 self.decoder_weight * g_loss_on_decoder
#
# 			# psnr	ssim	decoded message error rate
# 			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
# 			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")
# 			# psnr = calculate_psnr(encoded_images.detach(), images)
# 			# ssim = ssim_loss(encoded_images.detach(), images)
# 			error_rate_corrective = decoded_message_error_rate_batch(messages, decoded_messages)
# 			error_rate_refine = (torch.sum(torch.abs(messages.float() - decoded_messages.float()), dim=(0, 1), keepdim=True) / len(messages[0])).float()
#
# 		result = {
# 			"error_rate_corrective": error_rate_corrective,
# 			"error_rate_refine": error_rate_refine,
# 			"psnr": psnr,
# 			"ssim": ssim,
# 			"g_loss": g_loss,
# 			"g_loss_on_discriminator": g_loss_on_discriminator,
# 			"g_loss_on_encoder": g_loss_on_encoder,
# 			"g_loss_on_decoder": g_loss_on_decoder,
# 			"d_cover_loss": d_cover_loss,
# 			"d_encoded_loss": d_encoded_loss
# 		}
#
# 		return result, (images, encoded_images, noised_images, messages, decoded_messages)
#
# 	def decoded_message_error_rate(self, message, decoded_message):
# 		length = message.shape[0]
#
# 		message = message.gt(0.5)
# 		decoded_message = decoded_message.gt(0.5)
#
# 		error_rate = float(sum(message != decoded_message)) / length
# 		return error_rate
#
# 	def decoded_message_error_rate_batch(self, messages, decoded_messages):
# 		error_rate = 0.0
# 		batch_size = len(messages)
# 		for i in range(batch_size):
# 			error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
# 		print(decoded_messages[0])
# 		error_rate /= batch_size
# 		return error_rate
#
# 	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
# 		torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
# 		torch.save(self.discriminator.module.state_dict(), path_discriminator)
#
# 	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
# 		self.load_model_ed(path_encoder_decoder)
# 		self.load_model_dis(path_discriminator)
#
# 	def load_model_ed(self, path_encoder_decoder: str):
# 		self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))
#
# 	def load_model_dis(self, path_discriminator: str):
# 		self.discriminator.module.load_state_dict(torch.load(path_discriminator))





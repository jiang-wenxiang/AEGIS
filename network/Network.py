from .Encoder_MP_Decoder import *
from .Discriminator import Discriminator


class Network:

	def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
				 only_decoder=False):
		# device
		self.device = device

		# 1. 网络组件初始化
		# 如果不使用扩散模型，加载标准的 Encoder-Decoder 架构
		# noise_layers 被传入 EncoderDecoder，说明攻击模拟层是在网络前向传播内部动态调用的
		if not with_diffusion:
			self.encoder_decoder = EncoderDecoder(H, W, message_length, noise_layers).to(device)
		else:
			self.encoder_decoder = EncoderDecoder_Diffusion(H, W, message_length, noise_layers).to(device)

		# 初始化判别器 (用于对抗训练，提升图像视觉质量)
		self.discriminator = Discriminator().to(device)

		# 启用数据并行 (DataParallel)，支持多 GPU 训练
		self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
		self.discriminator = torch.nn.DataParallel(self.discriminator)

		# 特殊模式：只训练解码器
		# 如果开启，冻结编码器参数，不计算梯度。通常用于在固定编码方式下增强解码器的鲁棒性。
		if only_decoder:
			for p in self.encoder_decoder.module.encoder.parameters():
				p.requires_grad = False

		# 2. 定义 GAN 标签
		# 1 代表 "Cover" (原图/真实图)，0 代表 "Encoded" (水印图/伪造图)
		self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
		self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

		# 3. 优化器配置
		# 分别为生成部分(EC)和判别部分(D)配置 Adam 优化器
		print(lr)
		self.opt_encoder_decoder = torch.optim.Adam(
			filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
		self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

		# 4. 损失函数定义
		self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)  # 二元交叉熵，用于 GAN 的分类任务
		self.criterion_MSE = nn.MSELoss().to(device)            # 均方误差，用于图像像素差异和水印差异

		# 5. 损失权重 (关键超参数)
		# 决定了模型更看重哪项指标：
		# discriminator_weight: 对抗性权重 (隐蔽性)
		# encoder_weight: 图像像素相似度权重 (隐蔽性)
		# decoder_weight: 水印恢复准确度权重 (鲁棒性) - 设为 10 说明目前非常看重解码准确率
		self.discriminator_weight = 0.0001
		self.encoder_weight = 1
		self.decoder_weight = 10

	def train(self, images: torch.Tensor, messages: torch.Tensor):
		self.encoder_decoder.train()
		self.discriminator.train()

		with torch.enable_grad():
			# 数据迁移到 GPU
			images, messages = images.to(self.device), messages.to(self.device)

			# 前向传播：
			# encoded_images: 加了水印的图
			# noised_images: 加水印并经过噪声层攻击后的图 (解码器的输入)
			# decoded_messages: 从攻击后图像解出的信息
			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

			'''
			Step A: 训练判别器 (Train Discriminator)
			目标：准确区分原图 (Label=1) 和 水印图 (Label=0)
			'''
			self.opt_discriminator.zero_grad()

			# 1. 让判别器看原图，希望它输出 1
			d_label_cover = self.discriminator(images)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
			d_cover_loss.backward()

			# 2. 让判别器看水印图 (detach截断梯度，不更新生成器)，希望它输出 0
			d_label_encoded = self.discriminator(encoded_images.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
			d_encoded_loss.backward()

			self.opt_discriminator.step()

			'''
			Step B: 训练生成网络 (Train Encoder & Decoder)
			目标：生成高质量图像骗过判别器，同时保证信息能被解码
			'''
			self.opt_encoder_decoder.zero_grad()

			# Loss 1: 对抗损失 (欺骗判别器)
			# 把水印图给判别器，这次希望判别器被骗，认为它是原图 (Label=1)
			g_label_decoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

			# Loss 2: 图像感知损失 (隐蔽性)
			# 水印图和原图在像素级应该尽可能一致
			g_loss_on_encoder = self.criterion_MSE(encoded_images, images)

			# Loss 3: 消息解码损失 (鲁棒性)
			# 解出的水印和原始水印应该尽可能一致
			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

			# full loss：加权求和
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
					 self.decoder_weight * g_loss_on_decoder

			g_loss.backward()
			self.opt_encoder_decoder.step()

			# 计算 SSIM
			# 注意：这里使用的是 losses.ssim_loss，它是越小越好。
			# 这里的公式 `loss` 是为了将其反转回 SSIM 指标 (0~1)。
			psnr = kornia.metrics.psnr(encoded_images.detach(), images, 2)

			# ssim
			ssim = kornia.metrics.ssim(encoded_images.detach(), images, window_size=5).mean().item()

		'''
		计算误码率 (BER)
		'''
		error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

		# 打包返回结果
		result = {
			"error_rate": error_rate,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder": g_loss_on_encoder,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}
		return result


	def train_only_decoder(self, images: torch.Tensor, messages: torch.Tensor):
		self.encoder_decoder.train()

		with torch.enable_grad():
			# use device to compute
			images, messages = images.to(self.device), messages.to(self.device)
			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

			'''
			train encoder and decoder
			'''
			self.opt_encoder_decoder.zero_grad()

			# RESULT : the decoded message should be similar to the raw message
			g_loss = self.criterion_MSE(decoded_messages, messages)

			g_loss.backward()
			self.opt_encoder_decoder.step()

			# psnr
			psnr = kornia.metrics.psnr(encoded_images.detach(), images, 2)

			# ssim
			ssim = kornia.metrics.ssim(encoded_images.detach(), images, window_size=5).mean().item()

		'''
		decoded message error rate
		'''
		error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

		result = {
			"error_rate": error_rate,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
			"g_loss_on_discriminator": 0.,
			"g_loss_on_encoder": 0.,
			"g_loss_on_decoder": 0.,
			"d_cover_loss": 0.,
			"d_encoded_loss": 0.
		}
		return result

	def validation(self, images: torch.Tensor, messages: torch.Tensor):
		# 验证模式：只计算，不反向传播更新梯度
		self.encoder_decoder.eval()
		self.discriminator.eval()

		with torch.no_grad():
			# use device to compute
			images, messages = images.to(self.device), messages.to(self.device)
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
			g_loss_on_encoder = self.criterion_MSE(encoded_images, images)

			# RESULT : the decoded message should be similar to the raw message
			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

			# full loss
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
					 self.decoder_weight * g_loss_on_decoder

			# psnr
			psnr = kornia.metrics.psnr(encoded_images.detach(), images, 2)

			# ssim
			ssim = kornia.metrics.ssim(encoded_images.detach(), images, window_size=5).mean().item()

		'''
		decoded message error rate
		'''
		error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

		result = {
			"error_rate": error_rate,
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

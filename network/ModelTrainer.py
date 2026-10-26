from .DWT_DCT_SVD.DWT_DCT_SVD_Wrappers import DWT_DCT_SVD_EncoderDecoder
from .Encoder_MP_Decoder import *
from .Discriminator import Discriminator
import lpips  # pip install lpips
from .Basic_model.options import getDefaultHiddenConfig
from .Basic_model.options import get_DL_AEGIS_Config
from .Basic_model.options import get_AEGISnoAttentionConfig
from .Basic_model.options import get_GL_AEGIS_Config
from .Basic_model.options import get_SP_AEGIS_Config
from .Basic_model.options import get_AEGISnoAttention_noRes_Config
from .Basic_model.encoder_decoder import EncoderDecoder as HiDDeN_EncoderDecoder
from .MaskWM.models.Mask_Model import WatermarkModel
from .Trustmark.EncoderDecoder_TrustMark import TrustMarkEncoder, EncoderDecoder_TrustMark
from .StegaStamp.StegaStamp_Models import StegaStamp_EncoderDecoder
from .EditGuard.EditGuard_Model import EditGuard_Model
from .OmniGuard.OmniGuard_Model import OmniGuard_Model
from .ReDMark.ReDMark_Models import ReDMark_EncoderDecoder
from .RivaGAN.RivaGAN_Wrappers import RivaGAN_EncoderDecoder

from .SCIA_GAN_model.Encoder_MP_Decoder import EncoderDecoder_New, EncoderDecoder_Diffusion_New
from .UniversalDct.Encoder_Universal_DCT_Decoder import EncoderDecoder_Universal_DCT
from .EDBFormer.Encoder_EDBFormer_Decoder import EncoderDecoder_EDBFormer


class Network:

    def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr,
                 model_name="MBRS",
                 only_decoder=False,
                 with_discriminator=True,
                 discriminator_weight=0.0001, encoder_weight=1, decoder_weight=10, use_lpips_loss=False,
                 opt_discriminator_RMSprop=False, opt_lr=0.00001, use_WGAN=False, encoder_lr=None,
                 decoder_lr=None, mask_weight=0.0001):

        # device
        self.device = device
        self.with_discriminator = with_discriminator

        # 注意：所有模型都应接收 noise_layers, 以保证攻击环境一致
        if model_name == "MBRS":
            self.encoder_decoder = EncoderDecoder(H, W, message_length, noise_layers).to(device)
        elif model_name == "MBRS_Diffusion":
            self.encoder_decoder = EncoderDecoder_Diffusion(H, W, message_length, noise_layers).to(device)
        elif model_name == "HiDDeN":
            self.config = getDefaultHiddenConfig(H, W, message_length, model_name)
            self.encoder_decoder = HiDDeN_EncoderDecoder(self.config, noise_layers).to(device)
        elif model_name == "DWT_DCT_SVD":
            self.encoder_decoder = DWT_DCT_SVD_EncoderDecoder(H, W, message_length, noise_layers).to(device)
        elif model_name == "StegaStamp":
            self.encoder_decoder = StegaStamp_EncoderDecoder(H, W, message_length, noise_layers).to(device)
        elif model_name == "EditGuard":
            self.encoder_decoder = EditGuard_Model(H, W, message_length, noise_layers).to(device)
        elif model_name == "OmniGuard":
            self.encoder_decoder = OmniGuard_Model(H, W, message_length, noise_layers).to(device)
        elif model_name == "ReDMark":
            self.encoder_decoder = ReDMark_EncoderDecoder(H, W, message_length, noise_layers).to(device)
        elif model_name == "RivaGAN":
            self.encoder_decoder = RivaGAN_EncoderDecoder(H, W, message_length, noise_layers).to(device)
        elif model_name == "AEGIS_NANS":
            self.config = get_AEGISnoAttention_noRes_Config(H, W, message_length, model_name)
            self.encoder_decoder = HiDDeN_EncoderDecoder(self.config, noise_layers).to(device)
        elif model_name == "AEGIS_NA":
            self.config = get_AEGISnoAttentionConfig(H, W, message_length, model_name)
            self.encoder_decoder = HiDDeN_EncoderDecoder(self.config, noise_layers).to(device)
        elif model_name == "SP_AEGIS":
            self.config = get_SP_AEGIS_Config(H, W, message_length, model_name)
            self.encoder_decoder = HiDDeN_EncoderDecoder(self.config, noise_layers).to(device)
        elif model_name == "DL_AEGIS":
            self.config = get_DL_AEGIS_Config(H, W, message_length, model_name)
            self.encoder_decoder = HiDDeN_EncoderDecoder(self.config, noise_layers).to(device)
        elif model_name == "GL_AEGIS":
            self.config = get_GL_AEGIS_Config(H, W, message_length, model_name)
            self.encoder_decoder = HiDDeN_EncoderDecoder(self.config, noise_layers).to(device)
        elif model_name == "SCIA_GAN_with_Diffusion":
            self.encoder_decoder = EncoderDecoder_Diffusion_New(H, W, message_length, noise_layers).to(device)
        elif model_name == "SCIA_GAN":
            self.encoder_decoder = EncoderDecoder_New(H, W, message_length, noise_layers).to(device)
        elif model_name == "MaskWM":
            self.encoder_decoder = WatermarkModel(H, W, message_length, noise_layers)
        elif model_name == "Universal_DCT":
            self.encoder_decoder = EncoderDecoder_Universal_DCT(H, W, message_length, noise_layers)
        elif model_name == "TrustMark":
            self.encoder_decoder = EncoderDecoder_TrustMark(H, W, message_length, noise_layers).to(device)
        elif model_name == "EDBFormer":
            self.encoder_decoder = EncoderDecoder_EDBFormer(H, W, message_length, noise_layers).to(device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.model_name = model_name
        self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)

        if self.with_discriminator:
            self.discriminator = Discriminator().to(device)
            self.discriminator = torch.nn.DataParallel(self.discriminator)
            # 判别器优化器
            if opt_discriminator_RMSprop:
                self.opt_discriminator = torch.optim.RMSprop(self.discriminator.parameters(), lr=opt_lr)
            else:
                self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=opt_lr)
        else:
            self.discriminator = None
            self.opt_discriminator = None
        # self.discriminator = torch.nn.DataParallel(self.discriminator)

        if only_decoder:
            for p in self.encoder_decoder.module.encoder.parameters():
                p.requires_grad = False

        if self.with_discriminator:
            self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
            self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)
            self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
        else:
            self.label_cover = None
            self.label_encoded = None
            self.criterion_BCE = None

        print("lr:", lr)

        self.use_WGAN = use_WGAN

        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr

        if model_name == "DWT_DCT_SVD" or model_name == "TrustMark":
            self.opt_encoder_decoder = None
        else:
            if self.encoder_lr is None or self.decoder_lr is None:
                self.opt_encoder_decoder = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
                # self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

            else:
                self.optimizer_encoder = torch.optim.Adam(self.encoder_decoder.module.encoder.parameters(), lr=encoder_lr)

                self.optimizer_decoder = torch.optim.Adam(self.encoder_decoder.module.decoder.parameters(), lr=decoder_lr)

        self.criterion_MSG = nn.BCEWithLogitsLoss().to(device)
        self.criterion_MSE = nn.MSELoss().to(device)

        self.discriminator_weight = discriminator_weight
        self.encoder_weight = encoder_weight
        self.decoder_weight = decoder_weight
        self.mask_weight = mask_weight

        self.use_lpips_loss = use_lpips_loss

        self.lpips_fn = lpips.LPIPS(net='alex').to(device)

    def train(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.train()

        if self.with_discriminator:
            self.discriminator.train()

        with torch.enable_grad():
            images, messages = images.to(self.device), messages.to(self.device)

            if "MaskWM" in self.model_name:
                encoded_images, noised_images, decoded_messages, mask_gt, mask_pd = self.encoder_decoder(images, messages)
            else:
                encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            if "AEGIS" in self.model_name:
                fused_result, res_result = decoded_messages
                spa_decoded_messages, dwt_decoded_messages, dct_decoded_messages, svd_decoded_messages = res_result
                decoded_messages = fused_result

            d_cover_loss = 0.0
            d_encoded_loss = 0.0

            if self.with_discriminator:
                if self.use_WGAN:
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                    self.opt_discriminator.zero_grad()

                    d_label_cover = self.discriminator(images)
                    d_label_encoded = self.discriminator(encoded_images.detach())

                    # 判别器的目标是拉大真假图像的打分差距：最小化 -(Real的得分 - Fake的得分)
                    d_loss = -torch.mean(d_label_cover) + torch.mean(d_label_encoded)
                    d_loss.backward()

                    self.opt_discriminator.step()

                    d_cover_loss = torch.mean(d_label_cover)
                    d_encoded_loss = torch.mean(d_label_encoded)

                else:
                    self.opt_discriminator.zero_grad()

                    d_label_cover = self.discriminator(images)
                    d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
                    # 判别器直接反向传播训练
                    d_cover_loss.backward()

                    # 2. 让判别器看水印图 (detach截断梯度，不更新生成器)，希望它输出 0
                    d_label_encoded = self.discriminator(encoded_images.detach())
                    d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
                    # 判别器直接反向传播训练
                    d_encoded_loss.backward()

                    self.opt_discriminator.step()

            if self.encoder_lr is None or self.decoder_lr is None:
                self.opt_encoder_decoder.zero_grad()
            else:
                self.optimizer_encoder.zero_grad()
                self.optimizer_decoder.zero_grad()

            g_loss_on_discriminator = 0.0
            if self.with_discriminator:
                g_label_decoded = self.discriminator(encoded_images)

                if self.use_WGAN:
                    g_loss_on_discriminator = -torch.mean(g_label_decoded)

                else:
                    g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

            g_loss_on_encoder = self.criterion_MSE(encoded_images, images)

            if self.use_lpips_loss:
                lpips_loss = self.lpips_fn(encoded_images, images).mean()
                g_loss_on_encoder = g_loss_on_encoder + lpips_loss

            if "AEGIS" in self.model_name:
                g_loss_decoder = self.criterion_MSG(decoded_messages, messages)
                spa_loss_dec = self.criterion_MSG(spa_decoded_messages, messages)
                dwt_loss_dec = self.criterion_MSG(dwt_decoded_messages, messages)
                dct_loss_dec = self.criterion_MSG(dct_decoded_messages, messages)
                svd_loss_dec = self.criterion_MSG(svd_decoded_messages, messages)

                g_loss_on_decoder = g_loss_decoder + \
                                    self.config.spa_decoder_loss * spa_loss_dec + \
                                    self.config.dwt_decoder_loss * dwt_loss_dec + \
                                    self.config.dct_decoder_loss * dct_loss_dec + \
                                    self.config.svd_decoder_loss * svd_loss_dec
            elif "MaskWM" in self.model_name:
                g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages) \
                           + self.mask_weight * self.criterion_MSE(mask_gt, mask_pd)

            else:
                g_loss_on_decoder = self.criterion_MSG(decoded_messages, messages)

            g_loss = self.discriminator_weight * g_loss_on_discriminator + \
                     self.encoder_weight * g_loss_on_encoder + \
                     self.decoder_weight * g_loss_on_decoder

            g_loss.backward()

            if self.encoder_lr is None or self.decoder_lr is None:
                self.opt_encoder_decoder.step()
            else:
                self.optimizer_encoder.step()
                self.optimizer_decoder.step()
            # psnr
            psnr = kornia.metrics.psnr(encoded_images.detach(), images, 2)
            # ssim
            ssim = kornia.metrics.ssim(encoded_images.detach(), images, window_size=5).mean().item()
            # lpips
            lpips_val = self.lpips_fn(encoded_images.detach(), images).mean().item()

        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        def to_scalar(val):
            return val.item() if isinstance(val, torch.Tensor) else float(val)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_val,
            "g_loss": to_scalar(g_loss),
            "g_loss_on_discriminator": to_scalar(g_loss_on_discriminator),
            "g_loss_on_encoder": to_scalar(g_loss_on_encoder),
            "g_loss_on_decoder": to_scalar(g_loss_on_decoder),
            "d_cover_loss": to_scalar(d_cover_loss),
            "d_encoded_loss": to_scalar(d_encoded_loss)
        }
        return result

    def update_lr(self, lr):
        print("update lr:", lr)

        if self.encoder_lr is None or self.decoder_lr is None:
            # 正确做法：直接修改现有优化器的学习率，保留动量等历史状态
            for param_group in self.opt_encoder_decoder.param_groups:
                param_group['lr'] = lr
        else:
            encoder_lr, decoder_lr = lr
            for param_group in self.optimizer_encoder.param_groups:
                param_group['lr'] = encoder_lr
            for param_group in self.optimizer_decoder.param_groups:
                param_group['lr'] = decoder_lr

    def update_curriculum(self, is_warmup: bool, d_weight: float, e_weight: float, noise_intensity: float):
        self.discriminator_weight = d_weight
        self.encoder_weight = e_weight

        if is_warmup:
            self.discriminator_weight = 0.0
            self.encoder_weight = 0.0

        def apply_noise_intensity(module):
            if hasattr(module, 'set_intensity'):
                module.set_intensity(noise_intensity)

        self.encoder_decoder.apply(apply_noise_intensity)

    def train_only_decoder(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.train()

        base_model = self.encoder_decoder.module if hasattr(self.encoder_decoder, 'module') else self.encoder_decoder

        if 'NTR' in self.model_name:
            encoder_module = base_model.system.backbone
        else:
            encoder_module = base_model.encoder

        encoder_module.eval()

        for p in encoder_module.parameters():
            p.requires_grad = False
            p.grad = None
        # ==============================================================================

        with torch.enable_grad():
            images, messages = images.to(self.device), messages.to(self.device)

            if "MaskWM" in self.model_name:
                encoded_images, noised_images, decoded_messages, mask_gt, mask_pd = self.encoder_decoder(images, messages)
            else:
                encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            if "Muti_HiDDeN" in self.model_name:
                fused_result, res_result = decoded_messages
                spa_decoded_messages, dwt_decoded_messages, dct_decoded_messages, svd_decoded_messages = res_result
                decoded_messages = fused_result

            if self.encoder_lr is None or self.decoder_lr is None:
                self.opt_encoder_decoder.zero_grad(set_to_none=True)
            else:
                self.optimizer_encoder.zero_grad(set_to_none=True)
                self.optimizer_decoder.zero_grad(set_to_none=True)

            if "Muti_HiDDeN" in self.model_name:
                g_loss_decoder = self.criterion_MSG(decoded_messages, messages)
                spa_loss_dec = self.criterion_MSG(spa_decoded_messages, messages)
                dwt_loss_dec = self.criterion_MSG(dwt_decoded_messages, messages)
                dct_loss_dec = self.criterion_MSG(dct_decoded_messages, messages)
                svd_loss_dec = self.criterion_MSG(svd_decoded_messages, messages)

                g_loss_on_decoder = g_loss_decoder + \
                                    self.config.spa_decoder_loss * spa_loss_dec + \
                                    self.config.dwt_decoder_loss * dwt_loss_dec + \
                                    self.config.dct_decoder_loss * dct_loss_dec + \
                                    self.config.svd_decoder_loss * svd_loss_dec
            else:
                g_loss_on_decoder = self.criterion_MSG(decoded_messages, messages)

            g_loss = self.decoder_weight * g_loss_on_decoder

            g_loss.backward()

            if self.encoder_lr is None or self.decoder_lr is None:
                self.opt_encoder_decoder.step()
            else:
                self.optimizer_encoder.step()
                self.optimizer_decoder.step()

            # psnr
            psnr = kornia.metrics.psnr(encoded_images.detach(), images, 2)

            # ssim
            ssim = kornia.metrics.ssim(encoded_images.detach(), images, window_size=5).mean().item()

            # lpips
            lpips_val = self.lpips_fn(encoded_images.detach(), images).mean().item()

        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_val,
            "g_loss": g_loss,
            "g_loss_on_discriminator": 0.,
            "g_loss_on_encoder": 0.,
            "g_loss_on_decoder": g_loss_on_decoder,
            "d_cover_loss": 0.,
            "d_encoded_loss": 0.
        }
        return result

    def validation(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.eval()

        if self.with_discriminator:
            self.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            if "MaskWM" in self.model_name:
                encoded_images, noised_images, decoded_messages, mask_gt, mask_pd = self.encoder_decoder(images, messages)
            else:
                encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            if "Muti_HiDDeN" in self.model_name:
                fused_result, res_result = decoded_messages
                spa_decoded_messages, dwt_decoded_messages, dct_decoded_messages, svd_decoded_messages = res_result
                decoded_messages = fused_result

            d_cover_loss = 0.0
            d_encoded_loss = 0.0

            if self.with_discriminator:
                d_label_cover = self.discriminator(images)
                d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])

                d_label_encoded = self.discriminator(encoded_images.detach())
                d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])

            g_loss_on_discriminator = 0.0
            if self.with_discriminator:
                g_label_decoded = self.discriminator(encoded_images)
                g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

            g_loss_on_encoder = self.criterion_MSE(encoded_images, images)

            if self.use_lpips_loss:
                lpips_loss = self.lpips_fn(encoded_images, images).mean()
                g_loss_on_encoder = g_loss_on_encoder + lpips_loss

            if "Muti_HiDDeN" in self.model_name:
                g_loss_decoder = self.criterion_MSG(decoded_messages, messages)
                spa_loss_dec = self.criterion_MSG(spa_decoded_messages, messages)
                dwt_loss_dec = self.criterion_MSG(dwt_decoded_messages, messages)
                dct_loss_dec = self.criterion_MSG(dct_decoded_messages, messages)
                svd_loss_dec = self.criterion_MSG(svd_decoded_messages, messages)

                g_loss_on_decoder = g_loss_decoder + \
                                    self.config.spa_decoder_loss * spa_loss_dec + \
                                    self.config.dwt_decoder_loss * dwt_loss_dec + \
                                    self.config.dct_decoder_loss * dct_loss_dec + \
                                    self.config.svd_decoder_loss * svd_loss_dec
            else:
                g_loss_on_decoder = self.criterion_MSG(decoded_messages, messages)

            # full loss
            g_loss = self.discriminator_weight * g_loss_on_discriminator + \
                     self.encoder_weight * g_loss_on_encoder + \
                     self.decoder_weight * g_loss_on_decoder

            # psnr
            psnr = kornia.metrics.psnr(encoded_images.detach(), images, 2)

            # ssim
            ssim = kornia.metrics.ssim(encoded_images.detach(), images, window_size=5).mean().item()

            # lpips
            lpips_val = self.lpips_fn(encoded_images.detach(), images).mean().item()

        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_val,
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

    def save_model(self, path_encoder_decoder: str, path_discriminator: str = None):
        torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)

        if self.with_discriminator:
            torch.save(self.discriminator.module.state_dict(), path_discriminator)

    def load_model(self, path_encoder_decoder: str, path_discriminator: str = None):
        self.load_model_ed(path_encoder_decoder)

        if self.with_discriminator:
            self.load_model_dis(path_discriminator)

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))

    def load_model_dis(self, path_discriminator: str):
        assert self.with_discriminator, "with_discriminator is " + str(self.with_discriminator) + \
                                        ", but using load_model_dis func"
        if self.discriminator is not None:
            self.discriminator.module.load_state_dict(torch.load(path_discriminator))

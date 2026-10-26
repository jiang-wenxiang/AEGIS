from sympy import false

from .settings import *
import os
import time

'''
params setting
'''
settings = JsonConfig()
# settings.load_json_file("./settings/SCIA_GAN_train_settings_256.json")
# settings.load_json_file("./settings/MaskWM_train_settings_256.json")
# settings.load_json_file("./settings/Universal_DCT_train_settings_256.json")
# settings.load_json_file("./settings/EDBFormer_train_settings_256.json")
# settings.load_json_file("./settings/SCIA_GAN_with_Diffusion_train_settings_256.json")
# settings.load_json_file("./settings/SP_AEGIS_256_oi.json")
# settings.load_json_file("./settings/SP_AEGIS_256_pn.json")
# settings.load_json_file("./settings/SP_AEGIS_256_ps.json")
# settings.load_json_file("./settings/SP_AEGIS_256_psi.json")
settings.load_json_file("./settings/SP_AEGIS_256_phc.json")
# settings.load_json_file("./settings/HiDDeN_train_settings_256.json")
# settings.load_json_file("./settings/StegaStamp_train_settings.json")
# settings.load_json_file("./settings/GL_Res_Muti_HiDDeN_train_settings.json")
# settings.load_json_file("./settings/EditGuard_train_settings.json")
# settings.load_json_file("./settings/EditGuard_train_settings_256.json")
# settings.load_json_file("./settings/OmniGuard_train_settings_256.json")
# settings.load_json_file("./settings/MBRS_train_settings_256.json")
# settings.load_json_file("./settings/ReDMark_train_settings.json")
# settings.load_json_file("./settings/DWT_DCT_SVD_settings_256.json")
# settings.load_json_file("./settings/RivaGAN_train_settings_256.json")

only_decoder = settings.only_decoder

model_name = settings.model_name
project_name = settings.project_name
dataset_path = settings.dataset_path
epoch_number = settings.epoch_number
batch_size = settings.batch_size
train_continue = settings.train_continue
train_continue_path = settings.train_continue_path
train_continue_epoch = settings.train_continue_epoch
save_images_number = settings.save_images_number

# StegaStamp 新增需要解析参数
use_curriculum = getattr(settings, 'use_curriculum', False)        # 【核心开关】True: 动态增强 (StegaStamp专用) | False: 静态常量 (其他方法)
warmup_steps = getattr(settings, 'warmup_steps', 0)               # 解码器预热步数
ramp_steps = getattr(settings, 'ramp_steps', 0)                      # 爬升总步数
max_d_weight = getattr(settings, 'max_d_weight', settings.discriminator_weight)
max_e_weight = getattr(settings, 'max_e_weight', settings.encoder_weight)
max_noise_intensity = getattr(settings, 'max_noise_intensity', 1.0)

use_lpips_loss = getattr(settings, 'use_lpips_loss', False)
opt_discriminator_RMSprop = getattr(settings, 'opt_discriminator_RMSprop', False)
use_WGAN = getattr(settings, 'use_WGAN', False)

lr = settings.lr
decay_lr = getattr(settings, 'decay_lr', False)
opt_lr = getattr(settings, 'opt_lr', lr)

encoder_lr = getattr(settings, 'encoder_lr', None)
decoder_lr = getattr(settings, 'decoder_lr', None)

mask_weight = getattr(settings, 'mask_weight', 0.0)

H, W, message_length = settings.H, settings.W, settings.message_length,
noise_layers = settings.noise_layers

discriminator_weight = settings.discriminator_weight
encoder_weight = settings.encoder_weight
decoder_weight = settings.decoder_weight

with_discriminator = settings.with_discriminator

'''
file preparing
'''
full_project_name = model_name + "_" + project_name + "_m" + str(message_length)
# for noise in noise_layers:
# 	full_project_name += "_" + noise
# result_folder = "results/" + time.strftime(full_project_name + "__%Y_%m_%d__%H_%M_%S", time.localtime()) + "/"
result_folder = "results/" + full_project_name + "/"

if train_continue:
	wandb_name = settings.wandb_name
else:
	wandb_name = time.strftime(full_project_name + "__%Y_%m_%d__%H_%M_%S", time.localtime()) + "/"

if not os.path.exists(result_folder): os.mkdir(result_folder)
if not os.path.exists(result_folder + "images/"): os.mkdir(result_folder + "images/")
if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")

with open(result_folder + "/train_params.txt", "w") as file:
	content = "-----------------------" + \
			  time.strftime("Date: %Y/%m/%d %H:%M:%S", time.localtime()) + \
			  "-----------------------\n"

	for item in settings.get_items():
		content += item[0] + " = " + str(item[1]) + "\n"
	print(content)
	file.write(content)

with open(result_folder + "/train_log.txt", "w") as file:
	content = "-----------------------" + \
			  time.strftime("Date: %Y/%m/%d %H:%M:%S", time.localtime()) + \
			  "-----------------------\n"
	file.write(content)

with open(result_folder + "/val_log.txt", "w") as file:
	content = "-----------------------" + \
			  time.strftime("Date: %Y/%m/%d %H:%M:%S", time.localtime()) + \
			  "-----------------------\n"
	file.write(content)

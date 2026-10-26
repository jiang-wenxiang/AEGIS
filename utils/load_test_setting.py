try:
	from .settings import *
except ImportError:
	from settings import *

import os
import time

'''
params setting
'''
filename = "/root/autodl-tmp/MBRS-main/results/MBRS_m64/test_settings.json"
settings = JsonConfig()
settings.load_json_file(filename)

model_name = settings.model_name
dataset_path = settings.dataset_path
batch_size = 1
model_epoch = settings.model_epoch
strength_factor = settings.strength_factor
save_images_number = settings.save_images_number
lr = 1e-3
H, W, message_length = settings.H, settings.W, settings.message_length
noise_layers = settings.noise_layers

discriminator_weight = settings.discriminator_weight
encoder_weight = settings.encoder_weight
decoder_weight = settings.decoder_weight

with_discriminator = settings.with_discriminator

result_folder = "./results/" + settings.result_folder
test_base = "/test_"
for layer in settings.noise_layers:
	test_base += layer + "_"

test_param = result_folder + test_base + "s{}_params.json".format(strength_factor)
test_log = result_folder + test_base + "s{}_log.txt".format(strength_factor)
with open(test_param, "w") as file:
	content = ""
	for item in settings.get_items():
		content += item[0] + " = " + str(item[1]) + "\n"
	print(content)

	with open(filename, "r") as setting_file:
		content = setting_file.read()
		file.write(content)

with open(test_log, "w") as file:
	content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
														time.localtime()) + "-----------------------\n"
	file.write(content)

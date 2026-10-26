import random

import tqdm
from torch.utils.data import DataLoader
from utils import *
from network.ModelTrainer import *

from utils.load_train_setting import *

import wandb
from rich.progress import Progress

'''
train
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True

setup_seed(7)

wandb.init(
		project="AEGIS",
		name=wandb_name,
		config=settings
	)

network = Network(H, W, message_length, noise_layers, device, batch_size, lr, model_name, only_decoder,
				  with_discriminator = with_discriminator,
				  discriminator_weight=discriminator_weight,
				  encoder_weight=encoder_weight,
				  decoder_weight=decoder_weight,
				  use_lpips_loss=use_lpips_loss,
				  opt_discriminator_RMSprop=opt_discriminator_RMSprop,
				  opt_lr=opt_lr,
				  use_WGAN=use_WGAN, encoder_lr=encoder_lr, decoder_lr=decoder_lr,
				  mask_weight=mask_weight)

train_dataset = MBRSDataset(os.path.join(dataset_path, "train"), H, W)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = MBRSDataset(os.path.join(dataset_path, "test"), H, W)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

if train_continue:
	EC_path = "results/" + train_continue_path + "/models/EC_" + str(train_continue_epoch) + ".pth"
	D_path = "results/" + train_continue_path + "/models/D_" + str(train_continue_epoch) + ".pth"
	network.load_model(EC_path, D_path)

print("\nStart training : \n\n")

saved_iterations = np.random.choice(np.arange(len(val_dataloader)), size=save_images_number, replace=False)
saved_all = None


def denormalize(tensor):
	return (tensor + 1) / 2

def get_current_value(step, max_value, ramp_steps=0, use_curriculum=False):
	if not use_curriculum or ramp_steps <= 0:
		return max_value
	if step <= 0:
		return 0.0
	return min(max_value * (step / ramp_steps), max_value)

all_start_time = time.time()
global_step = train_continue_epoch * len(train_dataloader) if train_continue else 0

if decoder_lr is None or encoder_lr is None:
	initial_lr = lr
else:
	initial_lr = [encoder_lr, decoder_lr]

gamma = 0.5 ** (1 / 100)  # 0.9930925

save_every_epoch = True

for epoch in range(epoch_number):

	print("-" * 15)

	epoch += train_continue_epoch if train_continue else 0

	running_result = {
		"error_rate": 0.0,  # 比特错误率 (BER)
		"psnr": 0.0,  # 峰值信噪比 (图像质量)
		"ssim": 0.0,  # 结构相似性 (图像质量)
		"lpips": 0.0,
		"g_loss": 0.0,  # 生成器总 Loss
		"g_loss_on_discriminator": 0.0,  # 对抗 Loss
		"g_loss_on_encoder": 0.0,  # 编码 Loss
		"g_loss_on_decoder": 0.0,  # 解码 Loss
		"d_cover_loss": 0.0,  # 判别器识别原图的 Loss
		"d_encoded_loss": 0.0  # 判别器识别水印图的 Loss
	}

	start_time = time.time()

	if decay_lr:
		if decoder_lr is None or encoder_lr is None:
			current_lr = initial_lr * (gamma ** epoch)
		else:
			current_lr = [lr * (gamma ** epoch) for lr in initial_lr]

		network.update_lr(current_lr)

	run_num = 0
	with Progress() as progress:
		task1 = progress.add_task("[green]训练中("+str(global_step+1)+")", total=len(train_dataloader))
		for _, images, in enumerate(train_dataloader):
			image = images.to(device)

			message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

			global_step += 1

			is_warmup = use_curriculum and (global_step <= warmup_steps)

			current_d_weight = get_current_value(global_step - warmup_steps, max_d_weight, ramp_steps, use_curriculum)
			current_e_weight = get_current_value(global_step - warmup_steps, max_e_weight, ramp_steps, use_curriculum)
			current_noise_intensity = get_current_value(global_step - warmup_steps, max_noise_intensity, ramp_steps, use_curriculum)

			network.update_curriculum(
				is_warmup=is_warmup,
				d_weight=current_d_weight,
				e_weight=current_e_weight,
				noise_intensity=current_noise_intensity
			)

			result = network.train(image, message) if not only_decoder else network.train_only_decoder(image, message)

			to_scalar = lambda x: x.detach().item() if torch.is_tensor(x) else float(x)

			for key in result:
				running_result[key] += to_scalar(result[key])

			run_num += 1
			progress.update(task1, advance=1)

	seconds = int(time.time() - start_time)
	minutes = seconds // 60
	seconds = seconds % 60
	content = "Epoch " + str(epoch) + ", train duration: " + f"{minutes} min {seconds} s" + "\n"
	for key in running_result:
		content += key + "=" + str(round(running_result[key] / run_num, 4)) + ", "
	content += "\n"

	with open(result_folder + "/train_log.txt", "a") as file:
		file.write(content)
	print(content)


	if epoch % 10 == 0:
		val_result = {
			"error_rate": 0.0,
			"psnr": 0.0,
			"ssim": 0.0,
			"lpips": 0.0,
			"g_loss": 0.0,
			"g_loss_on_discriminator": 0.0,
			"g_loss_on_encoder": 0.0,
			"g_loss_on_decoder": 0.0,
			"d_cover_loss": 0.0,
			"d_encoded_loss": 0.0
		}

		start_time = time.time()

		val_num = 0
		saved_all = None

		with Progress() as progress:
			task2 = progress.add_task("[blue] 验证中", total=len(val_dataloader))
			for i, images in enumerate(val_dataloader):
				image = images.to(device)
				message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

				result, (images, encoded_images, noised_images, messages, decoded_messages) = network.validation(image, message)

				to_scalar = lambda x: x.detach().item() if torch.is_tensor(x) else float(x)

				for key in result:
					val_result[key] += to_scalar(result[key])

				val_num += 1

				if i in saved_iterations:
					if saved_all is None:
						saved_all = get_random_images(image, encoded_images, noised_images)
					else:
						saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

					if epoch % 10 == 0 and epoch != 0:
						img_tensor = image[0].detach().cpu()
						enc_tensor = encoded_images[0].detach().cpu()
						noi_tensor = noised_images[0].detach().cpu()

						img_vis_data = denormalize(img_tensor).clamp(0, 1)
						enc_vis_data = denormalize(enc_tensor).clamp(0, 1)
						noi_vis_data = denormalize(noi_tensor).clamp(0, 1)

						img_viz = wandb.Image(img_vis_data, caption=f"Original_Ep{epoch}_It{i}")
						enc_viz = wandb.Image(enc_vis_data, caption=f"Encoded_Ep{epoch}_It{i}")
						noi_viz = wandb.Image(noi_vis_data, caption=f"Noised_Ep{epoch}_It{i}")

						wandb.log({
							f"Visualization/{i}_Original": img_viz,
							f"Visualization/{i}_Encoded": enc_viz,
							f"Visualization/{i}_Noised": noi_viz,
							"epoch": epoch  # 绑定 epoch 方便时间轴对齐
						}, step=epoch)
				progress.update(task2, advance=1)

		save_images(saved_all, epoch, result_folder + "images/", resize_to=(W, H))

		wandb.log({
			"Val/Error_Rate": val_result['error_rate'] / val_num,
			"Val/PSNR": val_result['psnr'] / val_num,
			"Val/SSIM": val_result['ssim'] / val_num,
			"Val/LPIPS": val_result['lpips'] / val_num,
			"Val/Loss": val_result['g_loss'] / val_num,
			"Val/Loss_discriminator": val_result['g_loss_on_discriminator'] / val_num,
			"Val/Loss_encoder": val_result['g_loss_on_encoder'] / val_num,
			"Val/Loss_decoder": val_result['g_loss_on_decoder'] / val_num,
			"epoch": epoch
		}, step=epoch)

		seconds = int(time.time() - start_time)
		minutes = seconds // 60
		seconds = seconds % 60
		content = "Epoch " + str(epoch) + ", test duration: " + f"{minutes} min {seconds} s" + "\n"
		for key in val_result:
			content += key + "=" + str(round(val_result[key] / val_num, 4)) + ", "
		content += "\n"

		with open(result_folder + "/val_log.txt", "a") as file:
			file.write(content)
		print(content)

	wandb.log({
		"Train/Error_Rate": running_result['error_rate'] / run_num,
		"Train/PSNR": running_result['psnr'] / run_num,
		"Train/SSIM": running_result['ssim'] / run_num,
		"Train/LPIPS": running_result['lpips'] / run_num,
		"Train/Loss": running_result['g_loss'] / run_num,
		"Train/Loss_discriminator": running_result['g_loss_on_discriminator'] / run_num,
		"Train/Loss_encoder": running_result['g_loss_on_encoder'] / run_num,
		"Train/Loss_decoder": running_result['g_loss_on_decoder'] / run_num,
		"epoch": epoch
	}, step=epoch)

	if (epoch % 100 == 0) or (epoch == (train_continue_epoch + epoch_number - 1 if train_continue else epoch_number - 1)):

		path_model = result_folder + "models/"

		path_encoder_decoder = path_model + "EC_" + str(epoch) + ".pth"
		path_discriminator = path_model + "D_" + str(epoch) + ".pth"
		network.save_model(path_encoder_decoder, path_discriminator)
	elif save_every_epoch:
		path_model = result_folder + "models/"
		path_encoder_decoder = path_model + "EC_last.pth"
		path_discriminator = path_model + "D_last.pth"
		network.save_model(path_encoder_decoder, path_discriminator)

seconds = int(time.time() - all_start_time)
hour = seconds // 60 // 60
minutes = seconds // 60 % 60
seconds = seconds % 60
content = "All training duration: " + f"{hour}:{minutes}:{seconds}" + "\n"
print("\n" + content)

wandb.summary["final_train_error_rate"] = round(running_result['error_rate'] / run_num, 4)
wandb.summary["final_train_psnr"] = round(running_result['psnr'] / run_num, 4)
wandb.summary["final_val_error_rate"] = round(val_result['error_rate'] / val_num, 4)
wandb.summary["final_val_psnr"] = round(val_result['psnr'] / val_num, 4)
wandb.summary["Duration"] = round(time.time() - all_start_time, 2)
wandb.finish()

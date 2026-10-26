# benchmark_screencam.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import os
import json
import copy
import lpips  # 引入 lpips 库[cite: 2]

# 引入原有的工具类[cite: 2]
from utils import *
from network.ModelTrainer import *
from utils.plot_radar import plot_radar_chart, plot_line_chart
from utils.visualizer import save_attention_maps

import torchvision

# 引入我们刚刚封装的独立硬件模块
from hardware_control import ScreenCameraHardware

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

dataset_path = "datasets/"
batch_size = 4
H, W, message_length = 256, 256, 64

SAVE_DIR = "results_screencam"
RADAR_FILE_PATH = os.path.join(SAVE_DIR, "detailed_radar_data.json")


models_to_benchmark = [
    {
        "name": "MaskWM",
        "model_name": "MaskWM",
        "checkpoint": "results/MaskWM_256x256_m64/coco_s1/models/EC_100.pth"
    },
    {
        "name": "SCIA-GAN",
        "model_name": "SCIA_GAN",
        "checkpoint": "results/SCIA_GAN_256x256_m64/coco_s1/models/EC_100.pth"
    },
    {
        "name": "Universal-DCT",
        "model_name": "Universal_DCT",
        "checkpoint": "results/Universal_DCT_256x256_m64/coco_s1/models/EC_100.pth"
    },
    {
        "name": "SP_AEGIS",
        "model_name": "SP_AEGIS",
        "checkpoint": "results/SP_AEGIS_256x256_m64/models_pro_noise_SS_cocoS1_11/models/EC_500.pth"
    },
    {
        "name": "HiDDeN",
        "model_name": "HiDDeN",
        "checkpoint": "results/HiDDeN_256x256_m64/coco_s1/models/EC_200.pth"
    },
    {
        "name": "MBRS",
        "model_name": "MBRS",
        "checkpoint": "results/MBRS_256x256_m64/coco_s1/models/EC_100.pth"
    },
    {
        "name": "StegaStamp",
        "model_name": "StegaStamp",
        "checkpoint": "results/StegaStamp_256x256_m64/coco_s1/models/EC_200.pth"
    },
    {
        "name": "EditGuard",
        "model_name": "EditGuard",
        "checkpoint": "results/EditGuard_256x256_m64/coco_s1/models/EC_100.pth"
    },
    {
        "name": "DWT_DCT_SVD",
        "model_name": "DWT_DCT_SVD",
        "training_free": True,
    },
    {
        "name": "TrustMark",
        "model_name": "TrustMark",
        "training_free": True,
    },
    {
        "name": "EDBFormer",
        "model_name": "EDBFormer",
        "checkpoint": "results/EDBFormer_256x256_m64/coco_s1/models/EC_300.pth"
    },
]


def load_radar_data(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    return {}


def save_radar_data(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_physical_waypoints(json_path="watermark_waypoints.json"):
    """加载需要测试的机械臂位姿序列"""
    if not os.path.exists(json_path):
        # 如果文件不存在，生成几个模拟位姿用于测试代码
        return [{"pose": [1500] * 5, "id": "Center"}, {"pose": [1300] * 5, "id": "Left"}]
    with open(json_path, 'r') as f:
        return json.load(f)


physical_poses = load_physical_waypoints()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = MBRSDataset(os.path.join(dataset_path, "test"), H, W)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("Initializing Loss Functions (MSE, LPIPS)...")
criterion_MSE = nn.MSELoss().to(device)
criterion_LPIPS = lpips.LPIPS(net='alex').to(device)
criterion_LPIPS.eval()

detailed_radar_data = load_radar_data(RADAR_FILE_PATH)
if "ScreenCam" not in detailed_radar_data:
    detailed_radar_data["ScreenCam"] = {"intensities": [], "models": {}}

hardware = ScreenCameraHardware(camera_index=0)
num_poses = hardware.num_poses

total_images = len(test_dataset)
images_per_pose = total_images // num_poses

global_metrics = {
    model_cfg["name"]: {"error_rate": 0.0, "lpips": 0.0, "psnr": 0.0, "ssim": 0.0, "count": 0}
    for model_cfg in models_to_benchmark
}

print("\n" + "=" * 50)
print(f" 🚀 开始混合自动化测试 | 共 {total_images} 张图像")
print("=" * 50)

custom_progress = Progress(
    SpinnerColumn(spinner_name="hearts", style="bold red"),
    TextColumn("[bold magenta]{task.description}"),
    BarColumn(bar_width=40, style="#ff69b4", complete_style="green"),
    TaskProgressColumn(),
    " ⏱️ 耗时:", TimeElapsedColumn(),
    " ⏳ 剩余:", TimeRemainingColumn(),
)

with custom_progress as progress:
    pose_task = progress.add_task(f"[bold red]🦾 机械臂总进度 (共 {num_poses} 个位姿)...", total=num_poses)

    for pose_idx in range(num_poses):
        pose_id = f"Pose_{pose_idx + 1}"
        if pose_id not in detailed_radar_data["ScreenCam"]["intensities"]:
            detailed_radar_data["ScreenCam"]["intensities"].append(pose_id)

        progress.console.print(f"\n[bold yellow]✨ 正在前往位姿 {pose_idx + 1}/{num_poses} 并进行震定与视觉校验...[/]")
        hardware.move_to_pose_and_verify(pose_idx)

        start_idx = pose_idx * images_per_pose
        end_idx = start_idx + images_per_pose if pose_idx < num_poses - 1 else total_images
        subset_indices = list(range(start_idx, end_idx))
        pose_dataset = Subset(test_dataset, subset_indices)
        pose_dataloader = DataLoader(pose_dataset, batch_size=batch_size, shuffle=False)

        for model_config in models_to_benchmark:
            model_name = model_config["name"]

            if model_name not in detailed_radar_data["ScreenCam"]["models"]:
                detailed_radar_data["ScreenCam"]["models"][model_name] = []

            model_task = progress.add_task(
                f"[bold cyan] 📸 测试模型 [{model_name}]...",
                total=len(pose_dataloader)
            )

            current_network = Network(H, W, message_length, noise_layers=[],
                                      device=device, batch_size=batch_size, lr=0,
                                      model_name=model_config['model_name'])

            if not model_config.get("training_free", False):
                current_network.load_model_ed(model_config['checkpoint'])

            current_network.encoder_decoder.eval()
            core_model = current_network.encoder_decoder.module if isinstance(current_network.encoder_decoder,
                                                                              torch.nn.DataParallel) else current_network.encoder_decoder

            metrics_sum = {"error_rate": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "encoder_loss": 0.0,
                           "decoder_loss": 0.0}
            batch_count = 0

            with torch.no_grad():
                for batch_i, images in enumerate(pose_dataloader):
                    images = images.to(device)
                    messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], message_length))).to(device)

                    if hasattr(core_model, 'encode'):
                        encoded_images = core_model.encode(images, messages)
                    else:
                        encoded_images = core_model.encoder(images, messages)

                    global_start_idx = start_idx + (batch_i * batch_size)
                    noised_images = hardware.display_and_capture_batch(encoded_images, global_start_idx, model_name)
                    noised_images = noised_images.to(device)

                    if hasattr(core_model, 'decode'):
                        decoded_messages = core_model.decode(noised_images)
                    else:
                        decoded_messages = core_model.decoder(noised_images)

                    if "MaskWM" in model_config['model_name']:
                        decoded_messages, mask_pd = decoded_messages

                    if "Muti_HiDDeN" in model_config['model_name']:
                        fused_result, res_result = decoded_messages
                        decoded_messages = fused_result

                    error_rate = current_network.decoded_message_error_rate_batch(messages, decoded_messages)
                    psnr = kornia.metrics.psnr(encoded_images, images, 2).mean().item()
                    ssim = kornia.metrics.ssim(encoded_images, images, 5).mean().item()
                    lpips_val = criterion_LPIPS(encoded_images, images).mean().item()
                    enc_loss = criterion_MSE(encoded_images, images).item()
                    dec_loss = criterion_MSE(decoded_messages, messages).item()

                    metrics_sum["error_rate"] += error_rate
                    metrics_sum["psnr"] += psnr
                    metrics_sum["ssim"] += ssim
                    metrics_sum["lpips"] += lpips_val
                    metrics_sum["encoder_loss"] += enc_loss
                    metrics_sum["decoder_loss"] += dec_loss
                    batch_count += 1

                    actual_batch_size = images.size(0)
                    global_metrics[model_name]["error_rate"] += error_rate * actual_batch_size
                    global_metrics[model_name]["lpips"] += lpips_val * actual_batch_size
                    global_metrics[model_name]["psnr"] += psnr * actual_batch_size
                    global_metrics[model_name]["ssim"] += ssim * actual_batch_size
                    global_metrics[model_name]["count"] += actual_batch_size

                    for b_idx in range(actual_batch_size):
                        global_idx = global_start_idx + b_idx
                        if global_idx % 50 == 0:
                            single_err = current_network.decoded_message_error_rate_batch(
                                messages[b_idx:b_idx + 1], decoded_messages[b_idx:b_idx + 1]
                            )
                            single_acc = 1.0 - single_err

                            model_save_dir = os.path.join(SAVE_DIR, model_name)
                            os.makedirs(model_save_dir, exist_ok=True)
                            base_filename = os.path.join(model_save_dir, f"img_idx_{global_idx:04d}")

                            def denorm(t): return (t + 1.0) / 2.0

                            torchvision.utils.save_image(denorm(images[b_idx]), f"{base_filename}_1_original.png")
                            torchvision.utils.save_image(denorm(encoded_images[b_idx]),
                                                         f"{base_filename}_2_encoded.png")
                            torchvision.utils.save_image(denorm(noised_images[b_idx]), f"{base_filename}_3_camera.png")

                            orig_msg_list = messages[b_idx].detach().cpu().numpy().tolist()
                            dec_msg_list = decoded_messages[b_idx].detach().cpu().numpy().tolist()

                            with open(f"{base_filename}_4_info.txt", "w", encoding="utf-8") as f:
                                f.write(f"测试模型: {model_name}\n")
                                f.write(f"全局索引: {global_idx}\n")
                                f.write(f"物理位姿: {pose_id}\n")
                                f.write(f"单图解码准确率 (Accuracy): {single_acc * 100:.2f}%\n")
                                f.write(f"单图误码率 (BER): {single_err:.4f}\n")
                                f.write("-" * 40 + "\n")
                                # 限制浮点数输出长度，方便人工对比
                                dec_formatted = [round(x, 4) for x in dec_msg_list]
                                f.write(f"原始 Message : {orig_msg_list}\n")
                                f.write(f"解码 Message : {dec_formatted}\n")
                    progress.advance(model_task)

            progress.remove_task(model_task)

            if batch_count > 0:
                avg_err = metrics_sum["error_rate"] / batch_count
                accuracy = 1.0 - avg_err

                pose_list_idx = detailed_radar_data["ScreenCam"]["intensities"].index(pose_id)
                model_data_list = detailed_radar_data["ScreenCam"]["models"][model_name]
                while len(model_data_list) <= pose_list_idx:
                    model_data_list.append(0.0)
                model_data_list[pose_list_idx] = accuracy

        progress.advance(pose_task)

        save_radar_data(detailed_radar_data, RADAR_FILE_PATH)

hardware.close()

print(f"\nGenerating Radar Charts for ScreenCam to {SAVE_DIR}...")
save_dir = os.path.join(SAVE_DIR, "radar_charts")
os.makedirs(save_dir, exist_ok=True)

data_block = detailed_radar_data["ScreenCam"]
categories = data_block["intensities"]
model_results = data_block["models"]

plot_radar_chart(
    model_results,
    categories,
    filename=os.path.join(save_dir, "abs_radar_ScreenCam.svg"),
    title="Screen Camera Attack"
)

print("\nScreen-Cam Physical Benchmark Completed!")

print("\nScreen-Cam Physical Benchmark Completed!")

print("\n" + "=" * 80)
print(" 🚀 所有物理位姿测试执行完毕，全局平均指标如下：")
print("=" * 80)
for model_name, metrics in global_metrics.items():
    total_count = metrics["count"]
    if total_count > 0:
        avg_err = metrics["error_rate"] / total_count
        avg_lpips = metrics["lpips"] / total_count
        avg_psnr = metrics["psnr"] / total_count
        avg_ssim = metrics["ssim"] / total_count

        print(
            f" 模型 [{model_name:^18}] | 全局平均 BER: {avg_err:.4f} | LPIPS: {avg_lpips:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")
print("=" * 80)
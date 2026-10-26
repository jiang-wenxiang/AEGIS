import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from network.ModelTrainer import *
import os
from network.Noise import Noise
import wandb
import copy
import lpips

# 引入您之前保存的 plot_radar.py
from utils.plot_radar import plot_radar_chart, plot_line_chart
from utils.visualizer import save_attention_maps

import json
import os
from benchmark_extensions import (
    benchmark_combined_noises,
    benchmark_model_complexity,
    build_combined_noise_schedule,
    plot_cross_method_combined_boxplots
)

def save_radar_data(data, filepath="results/detailed_radar_data.json"):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:

        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[Info] Data successfully saved to {filepath}")

def load_radar_data(filepath="results/detailed_radar_data.json"):

    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[Info] Successfully loaded existing data from {filepath}")
        return data
    else:
        print(f"[Info] No existing data found at {filepath}. Initializing empty dictionary.")
        return {}

dataset_path = "datasets/"
batch_size = 4
H, W, message_length = 256, 256, 64

FORCE_RETEST = False
RUN_COMPLEXITY = True
TIMING_WARMUP = 20
TIMING_REPEATS = 100
COMBINATION_REPEATS = 5

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

# models_to_benchmark = [
#     {
#         "name": "SP_AEGIS_128x128_m64",
#         "model_name": "SP_AEGIS",
#         "checkpoint": "results/SP_AEGIS_128x128_m64/models/EC_368.pth",
#         "ablation_mask": [1, 1, 1, 1]
#     },
#     {
#         "name": "SP_AEGIS_128x128_m64",
#         "model_name": "SP_AEGIS",
#         "checkpoint": "results/SP_AEGIS_128x128_m64/models/EC_368.pth",
#         "ablation_mask": [0, 1, 1, 1]
#     },
#     {
#         "name": "SP_AEGIS_128x128_m64",
#         "model_name": "SP_AEGIS",
#         "checkpoint": "results/SP_AEGIS_128x128_m64/models/EC_368.pth",
#         "ablation_mask": [1, 0, 1, 1]
#     },
#     {
#         "name": "SP_AEGIS_128x128_m64",
#         "model_name": "SP_AEGIS",
#         "checkpoint": "results/SP_AEGIS_128x128_m64/models/EC_368.pth",
#         "ablation_mask": [1, 1, 0, 1]
#     },
#     {
#         "name": "SP_AEGIS_128x128_m64",
#         "model_name": "SP_AEGIS",
#         "checkpoint": "results/SP_AEGIS_128x128_m64/models/EC_368.pth",
#         "ablation_mask": [1, 1, 1, 0]
#     },
#     # ... 添加更多
# ]

# models_to_benchmark = [
#     {
#         "name": "DWT_DCT_SVD_m64",
#         "model_name": "DWT_DCT_SVD",
#         "training_free": True,
#     },
# ]

noise_schedule = []

# # ------------------------------------------------------
# # 0. 几何攻击：缩放 (Scaling)
# # 测试点数：8 个 (1.0, 1.2, ... 0.8)
# # -----------------------------------------------------
# for sc in [1.0, 1.2, 1.4, 1.6, 1.8, 0.4, 0.6, 0.8]:
#     config_str = f"Scaling({sc})"
#     noise_schedule.append({
#         "type": "Scaling",
#         "intensity": sc,
#         "config_list": [config_str]
#     })
#
# # ------------------------------------------------------
# # 1. 几何攻击：旋转 (Rotation)
# # 测试点数：18 个 (0, 10, ... 180)
# # -----------------------------------------------------
# for angle in range(0, 180, 10):
#     config_str = f"Rotation({angle})" if angle > 0 else "Identity()"
#     noise_schedule.append({
#         "type": "Rotation",
#         "intensity": angle,
#         "config_list": [config_str]
#     })
#
# # ------------------------------------------------------
# # 2. 扭曲攻击 (Warp) [新增]
# # 测试点: 0.0, 0.1 ... 1.0 (共 8 个点)
# # 假设 Warp(0.2) 中的参数越大，扭曲越严重
# # ------------------------------------------------------
# for i in range(0, 8):
#     strength = round(i * 0.125, 3)  # 0.0, 0.1, ..., 1.0
#     config_str = f"Warp({strength})" if strength > 0 else "Identity()"
#     noise_schedule.append({
#         "type": "Warp",
#         "intensity": strength, # X轴: 扭曲强度
#         "config_list": [config_str]
#     })
#
# # ------------------------------------------------------
# # 3. 压缩攻击：JPEG
# # 测试点数：10 个 (Q=100 到 Q=10)
# # ------------------------------------------------------
# for q in range(100, 9, -10):
#     config_str = f"Jpeg({q})" if q < 100 else "Identity()"
#     noise_schedule.append({
#         "type": "JPEG",
#         "intensity": 100 - q, # intensity: 0(无损) -> 90(强压)
#         "config_list": [config_str]
#     })
#
# # ------------------------------------------------------
# # 4. 几何攻击：裁剪 (Crop)
# # 测试点数：8 个 (保留 100% -> 保留 30%)
# # ------------------------------------------------------
# # 裁剪掉的比例: 0%, 10%, ... 70%
# for i in range(0, 8):
#     p = (100 - i * 10) / 100.0 # 保留比例
#     # Crop(height_ratio, width_ratio)
#     config_str = f"Crop({p:.2f}, {p:.2f})" if i > 0 else "Identity()"
#     noise_schedule.append({
#         "type": "Crop",
#         "intensity": i / 100.0,  # intensity: 0.0 -> 0.6
#         "config_list": [config_str]
#     })
#
# # ------------------------------------------------------
# # 5. 遮挡攻击 (Cropout) [类似 Dropout/Cutout]
# # 测试点: 0.0 -> 0.7 (共 8 个点)
# # 假设 Cropout(0.3, 0.3) 意味着挖掉 30% 的区域
# # ------------------------------------------------------
# for i in range(0, 8):
#     # p 从 0.0 增到 0.8 (步长 0.1)
#     p = round(i * 0.1, 1)
#
#     config_str = f"Cropout({p}, {p})" if p > 0 else "Identity()"
#     noise_schedule.append({
#         "type": "Cropout",
#         "intensity": p,  # X轴: 挖掉的比例
#         "config_list": [config_str]
#     })
#
# # ------------------------------------------------------
# # 6. 高斯模糊/滤波 (GF)
# # 测试点: 0.0 -> 2.0 (共 8 个点)
# # 假设 GF(1.0) 里的参数是 sigma (浮点数)
# # ------------------------------------------------------
# for i in range(0, 8):
#     sigma = round(i * 0.25, 2)  # 0.0, 0.2, ..., 2.0
#
#     config_str = f"GF({sigma})" if sigma > 0 else "Identity()"
#     noise_schedule.append({
#         "type": "GF",  # Gaussian Filter
#         "intensity": sigma,  # X轴: Sigma值
#         "config_list": [config_str]
#     })
#
# # ------------------------------------------------------
# # 7. 高斯噪声 (GN) [新增]
# # 类名: GN(var)
# # 测试范围: 标准差 std = 0.0 -> 0.2 (即方差 var = 0.0 -> 0.04)
# # 共 8 个测试点
# # ------------------------------------------------------
# for i in range(0, 8):
#     # 设定标准差步长为 0.02 (0.0, 0.02, ... 0.20)
#     std = round(i * 0.05, 3)
#
#     # GN 类接收的是方差 (Variance)，所以需要平方
#     var = round(std ** 2, 5)
#
#     # 构造字符串: GN(0.0004)
#     config_str = f"GN({var})" if var > 0 else "Identity()"
#
#     noise_schedule.append({
#         "type": "GN",  # 噪声类型
#         "intensity": std,  # X轴: 这里记录标准差 std 更符合直觉
#         "config_list": [config_str]
#     })


# ------------------------------------------------------
# 8. 遮挡攻击：Dropout (随机丢弃像素)
# 测试点数：8 个 (p 0.0 -> 0.7)
# ------------------------------------------------------
# 覆盖 0.0, 0.10, ... 0.70
for i in range(0, 8):
    p = round(i * 0.1, 2)
    config_str = f"Dropout({p})" if p > 0.0 else "Identity()"
    noise_schedule.append({
        "type": "Dropout",
        "intensity": p,  # intensity: 0.0 -> 0.7
        "config_list": [config_str]
    })


# # ------------------------------------------------------
# # 9 (+1). 几何攻击：分辨率缩放 (Resizing) [新增]
# # 测试点数：9 个 (1.0 到 0.2，模拟分辨率逐渐降低)
# # ------------------------------------------------------
# for i in range(0, 9):
#     sc = round(1.0 - i * 0.1, 1)  # 1.0, 0.9, 0.8 ... 0.2
#     config_str = f"Resizing({sc})" if sc < 1.0 else "Identity()"
#     noise_schedule.append({
#         "type": "Resizing",
#         "intensity": sc,         # X轴: 当前保留的分辨率比例
#         "config_list": [config_str]
#     })
#
# # ------------------------------------------------------
# # 10 (+2). 颜色失真：亮度/对比度调整 (ColorJitter) [新增]
# # 测试点数：9 个 (扰动幅度从 0.0 到 0.4)
# # ------------------------------------------------------
# for i in range(0, 9):
#     strength = round(i * 0.05, 2)  # 0.0, 0.05, 0.10 ... 0.40
#     config_str = f"ColorJitter({strength})" if strength > 0 else "Identity()"
#     noise_schedule.append({
#         "type": "ColorJitter",
#         "intensity": strength,   # X轴: 颜色扰动幅度
#         "config_list": [config_str]
#     })
#
# # ------------------------------------------------------
# # 11 (+3).. 滤波攻击：锐化 (Sharpening) [新增]
# # 测试点数：9 个 (锐化程度 alpha 从 0.0 到 3.2)
# # ------------------------------------------------------
# for i in range(0, 9):
#     alpha = round(i * 0.4, 1)  # 0.0, 0.4, 0.8 ... 3.2
#     config_str = f"Sharpening({alpha})" if alpha > 0 else "Identity()"
#     noise_schedule.append({
#         "type": "Sharpening",
#         "intensity": alpha,      # X轴: 锐化强度 alpha
#         "config_list": [config_str]
#     })
#
# # ------------------------------------------------------
# # 12 (+4).. 压缩攻击：JPEG2000 [新增]
# # 测试点数：10 个 (Q=100 到 Q=10，与普通JPEG保持一致)
# # ------------------------------------------------------
# for q in range(100, 9, -10):
#     config_str = f"Jpeg2000Sim({q})" if q < 100 else "Identity()"
#     noise_schedule.append({
#         "type": "JPEG2000",
#         "intensity": 100 - q,    # X轴: intensity 0(无损) -> 90(强压)
#         "config_list": [config_str]
#     })

combined_noise_schedule = build_combined_noise_schedule(COMBINATION_REPEATS)
test_for_combined_noises = False
# combined_noise_schedule = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载测试集
test_dataset = MBRSDataset(os.path.join(dataset_path, "test"), H, W)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

print("Initializing Loss Functions (MSE, LPIPS)...")
criterion_MSE = nn.MSELoss().to(device)

criterion_LPIPS = lpips.LPIPS(net='alex').to(device)
criterion_LPIPS.eval()

print(f"Total Test Images: {len(test_dataset)}")
print(f"Models to evaluate: {len(models_to_benchmark)}")
print(f"Noise steps: {len(noise_schedule)}")

detailed_radar_data = {}

data_filepath = "results/detailed_radar_data.json"
detailed_radar_data = load_radar_data(data_filepath)

for model_config in models_to_benchmark:
    if "ablation_mask" in model_config:
        model_id = f"{model_config['name']}_{model_config['ablation_mask']}"
    else:
        model_id = model_config['name']

    already_tested = False
    if detailed_radar_data:
        # 随便取现存的第一个噪声类别查一下
        first_noise = next(iter(detailed_radar_data.values()))
        if model_id in first_noise.get("models", {}):
            already_tested = True

    if already_tested and not FORCE_RETEST:
        print(f"\n[Skip] 模型 '{model_id}' 已有测试数据，直接跳过...")
        continue

    print(f"\n[Run] 正在测试模型 '{model_id}' ...")

    print(f"\n{'=' * 40}")
    print(f"Processing Model: {model_config['name']}")
    print(f"{'=' * 40}")

    if 'ablation_mask' in model_config:
        wandb.init(
            project="Robustness_abl_FullMetrics",
            name=model_config['name'] + "_" + str(model_config['ablation_mask']),
            config=model_config,
            reinit=True
        )
    else:
        wandb.init(
            project="Robustness_FullMetrics",
            name=model_config['name'],
            config=model_config,
            reinit=True
        )

    current_network = Network(H, W, message_length, noise_layers=[],
                              device=device, batch_size=batch_size, lr=0,
                              model_name=model_config['model_name'])

    is_training_free = model_config.get("training_free", False)

    if is_training_free:
        print(f"  -> [Info] {model_config['name']} ")
    else:
        if os.path.exists(model_config['checkpoint']):
            print(f"Loading weights from: {model_config['checkpoint']}")
            current_network.load_model_ed(model_config['checkpoint'])
        else:
            print(f"[Error] Checkpoint not found: {model_config['checkpoint']}")
            continue

    current_network.encoder_decoder.eval()

    if isinstance(current_network.encoder_decoder, torch.nn.DataParallel):
        core_model = current_network.encoder_decoder.module
    else:
        core_model = current_network.encoder_decoder

    if RUN_COMPLEXITY:
        try:
            benchmark_model_complexity(
                core_model=core_model,
                model_config=model_config,
                model_id=model_id,
                test_dataloader=test_dataloader,
                device=device,
                message_length=message_length,
                warmup=TIMING_WARMUP,
                repeats=TIMING_REPEATS,
            )
        except Exception as exc:
            print(f"  -> [Warning] Complexity profiling failed: {exc}")

    current_model_performance = {}

    for step_idx, noise_item in enumerate(noise_schedule):

        noise_type = noise_item['type']
        intensity = noise_item['intensity']
        raw_config_list = noise_item['config_list']

        print(f"  -> Testing {raw_config_list[0]} (Strength: {intensity}) ...", end="")

        try:
            test_noise_module = Noise(copy.deepcopy(raw_config_list)).to(device)
            test_noise_module.eval()
        except Exception as e:
            print(f"\n[Error] Failed to initialize Noise with {raw_config_list}: {e}")
            continue

        metrics_sum = {
            "error_rate": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "lpips": 0.0,
            "encoder_loss": 0.0,
            "decoder_loss": 0.0
        }
        batch_count = 0

        with torch.no_grad():
            for i, images in enumerate(test_dataloader):
                images = images.to(device)
                messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], message_length))).to(device)

                if "ablation_mask" in model_config:
                    return_weights = False
                    if i % 100 == 0:
                        return_weights = True
                    if hasattr(core_model, 'encode'):
                        encoded_images = core_model.encode(images, messages, model_config["ablation_mask"], return_weights=return_weights)
                    else:
                        encoded_images = core_model.encoder(images, messages, model_config["ablation_mask"], return_weights=return_weights)

                    if return_weights:
                        encoded_images, attention_weights = encoded_images
                        save_attention_maps(
                            encoded_images,
                            save_dir="results/visual_weights",
                            batch_idx=i * batch_size,
                            prefix=f"{model_config['name']}_{str(model_config['ablation_mask'])}",
                            original_images=images
                        )

                else:
                    if hasattr(core_model, 'encode'):
                        encoded_images = core_model.encode(images, messages)
                    else:
                        encoded_images = core_model.encoder(images, messages)

                noised_images = test_noise_module([encoded_images, images])

                if hasattr(core_model, 'decode'):
                    decoded_messages = core_model.decode(noised_images)
                else:
                    decoded_messages = core_model.decoder(noised_images)

                if "MaskWM" in model_config['model_name']:
                    decoded_messages, mask_pd = decoded_messages

                if "Muti_HiDDeN" in model_config['model_name']:
                    fused_result, res_result = decoded_messages
                    spa_decoded_messages, dwt_decoded_messages, dct_decoded_messages, svd_decoded_messages = res_result
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

        if batch_count > 0:
            metrics_avg = {k: v / batch_count for k, v in metrics_sum.items()}

            print(f" Done. BER: {metrics_avg['error_rate']:.4f} | LPIPS: {metrics_avg['lpips']:.4f}")

            log_data = {
                "Noise_Intensity": intensity,
                "epoch": intensity
            }
            for key, val in metrics_avg.items():
                log_data[f"{noise_type}/{key}"] = val

            wandb.log(log_data)

            if noise_type not in current_model_performance:
                current_model_performance[noise_type] = {}
            current_model_performance[noise_type][intensity] = metrics_avg['error_rate']

    if test_for_combined_noises:
        benchmark_combined_noises(
            core_model=core_model,
            current_network=current_network,
            model_config=model_config,
            model_id=model_id,
            test_dataloader=test_dataloader,
            device=device,
            message_length=message_length,
            noise_class=Noise,
            criterion_mse=criterion_MSE,
            criterion_lpips=criterion_LPIPS,
            schedule=combined_noise_schedule,
            wandb_module=wandb,
        )

    for n_type, int_dict in current_model_performance.items():
        if n_type not in detailed_radar_data:
            detailed_radar_data[n_type] = {
                "intensities": [],
                "models": {}
            }

        if not detailed_radar_data[n_type]["intensities"]:
            sorted_ints = sorted(int_dict.keys())
            detailed_radar_data[n_type]["intensities"] = sorted_ints

        sorted_ints = detailed_radar_data[n_type]["intensities"]

        data_list = [1.0 - int_dict.get(i, 0.5) for i in sorted_ints]

        if "ablation_mask" in model_config:

            detailed_radar_data[n_type]["models"][model_config['name'] + "_" + str(model_config['ablation_mask'])] = data_list
        else:
            detailed_radar_data[n_type]["models"][model_config['name']] = data_list

    wandb.finish()

save_radar_data(detailed_radar_data, data_filepath)

print("\nGenerating Separate Radar Charts for each Noise Type...")

save_dir = "results/radar_charts"
os.makedirs(save_dir, exist_ok=True)

for noise_type, data_block in detailed_radar_data.items():
    sorted_intensities = data_block["intensities"]
    categories = [str(i) for i in sorted_intensities]

    model_results = data_block["models"]

    filename = os.path.join(save_dir, f"abs_radar_{noise_type}.svg")
    line_filename = os.path.join(save_dir, f"abs_line_{noise_type}.svg")
    title = f"{noise_type}"

    print(f"  -> Plotting {noise_type} ... saved to {filename}")

    plot_radar_chart(
        model_results,
        categories,
        filename=filename,
        title=title
    )

    plot_line_chart(
        model_results,
        categories,
        filename=line_filename,
        title=title
    )

print(f"\nAll charts saved in '{save_dir}'. Benchmark Completed!")

observations_file = "results/combined_noise_observations.json"
if os.path.exists(observations_file):
    print("\nGenerating Cross-Method Combined Noise Boxplots...")
    plot_cross_method_combined_boxplots(observations_file, "results/combined_boxplots")
else:
    print("\n[Warning] Observations file not found, skipping cross-method boxplots.")

print("\nBenchmark Completed!")

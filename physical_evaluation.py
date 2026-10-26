import torch
from torch.utils.data import DataLoader
import os
import json

# 引入你的网络模型和数据加载器
from network.ModelTrainer import Network
from real_datasets_generator.Physical_Dataloader import PhysicalWatermarkDataset

# ==========================================
# 1. 实验配置区域
# ==========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 4
H, W, message_length = 256, 256, 64

# [核心修改] 在字典中配置每个模型对应的 checkpoint 和 物理数据集路径
# models_to_benchmark = [
#     {
#         "name": "SP_AEGIS_256x256_m64",
#         "model_name": "SP_AEGIS",
#         "checkpoint": "results/SP_AEGIS_256x256_m64/models/EC_368.pth",
#         "dataset_path": "real_datasets_generator/dataset_final/AEGIS(Ours)"  # 该模型对应的裁切数据集目录
#     },
#     {
#         "name": "HiDDeN",
#         "model_name": "HiDDeN",
#         "checkpoint": "results/HiDDeN_256x256_m64/models/EC_100.pth",
#         "dataset_path": "real_datasets_generator/dataset_final/HiDDeN"  # 该模型对应的裁切数据集目录
#     },
#     {
#         "name": "MBRS",
#         "model_name": "MBRS",
#         "checkpoint": "results/MBRS_256x256_m64/models/EC_100.pth",
#         "dataset_path": "real_datasets_generator/dataset_final/MBRS_128x128_m64"  # 该模型对应的裁切数据集目录
#     },
#     {
#         "name": "StegaStamp",
#         "model_name": "StegaStamp",
#         "checkpoint": "results/StegaStamp_256x256_m64/models/EC_100.pth",
#         "dataset_path": "real_datasets_generator/dataset_final/StegaStamp_128x128_m64"  # 该模型对应的裁切数据集目录
#     },
#     {
#         "name": "EditGuard",
#         "model_name": "EditGuard",
#         "checkpoint": "results/EditGuard_256x256_m64/models/EC_100.pth",
#         "dataset_path": "real_datasets_generator/dataset_final/EditGuard_128x128_m64"  # 该模型对应的裁切数据集目录
#     },
#     # 按照上面的格式添加你的其他模型...
# ]

models_to_benchmark = [
    {
        "name": "MaskWM",
        "model_name": "MaskWM",
        "checkpoint": "results/MaskWM_256x256_m64/coco_s1/models/EC_100.pth",
        "dataset_path": "real_datasets_generator/dataset_final/MaskWM"
    },
    {
        "name": "SCIA-GAN",
        "model_name": "SCIA_GAN",
        "checkpoint": "results/SCIA_GAN_256x256_m64/coco_s1/models/EC_100.pth",
        "dataset_path": "real_datasets_generator/dataset_final/SCIA-GAN"
    },
    {
        "name": "SP_AEGIS",
        "model_name": "SP_AEGIS",
        "checkpoint": "results/SP_AEGIS_256x256_m64/models_pro_noise_SS_cocoS1_11/models/EC_500.pth",
        "dataset_path": "real_datasets_generator/dataset_final/AEGIS(Ours)"
    },
    {
        "name": "HiDDeN",
        "model_name": "HiDDeN",
        "checkpoint": "results/HiDDeN_256x256_m64/coco_s1/models/EC_200.pth",
        "dataset_path": "real_datasets_generator/dataset_final/HiDDeN"
    },
    {
        "name": "MBRS",
        "model_name": "MBRS",
        "checkpoint": "results/MBRS_256x256_m64/coco_s1/models/EC_100.pth",
        "dataset_path": "real_datasets_generator/dataset_final/MBRS"
    },
    {
        "name": "StegaStamp",
        "model_name": "StegaStamp",
        "checkpoint": "results/StegaStamp_256x256_m64/coco_s1/models/EC_200.pth",
        "dataset_path": "real_datasets_generator/dataset_final/StegaStamp"
    },
    {
        "name": "EditGuard",
        "model_name": "EditGuard",
        "checkpoint": "results/EditGuard_256x256_m64/coco_s1/models/EC_100.pth",
        "dataset_path": "real_datasets_generator/dataset_final/EditGuard"
    },
    # {
    #     "name": "DWT_DCT_SVD",
    #     "model_name": "DWT_DCT_SVD",
    #     "training_free": True,
    #     "dataset_path": "real_datasets_generator/dataset_final/DWT_DCT_SVD"
    # },
    {
        "name": "TrustMark",
        "model_name": "TrustMark",
        "training_free": True,
        "dataset_path": "real_datasets_generator/dataset_final/TrustMark"
    },
    {
        "name": "EDBFormer",
        "model_name": "EDBFormer",
        "checkpoint": "results/EDBFormer_256x256_m64/coco_s1/models/EC_300.pth",
        "dataset_path": "real_datasets_generator/dataset_final/EDBFormer"
    },
    # {
    #     "name": "Universal-DCT",
    #     "model_name": "Universal_DCT",
    #     "checkpoint": "results/Universal_DCT_256x256_m64/coco_s1/models/EC_100.pth",
    #     "dataset_path": "real_datasets_generator/dataset_final/Universal-DCT"
    # },
    # ... 添加更多
]

# ==========================================
# 2. 评估函数定义
# ==========================================

def evaluate_physical_dataset():
    print(f"Start Physical Evaluation on Device: {device}")
    print("-" * 50)

    final_results = {}

    for config in models_to_benchmark:
        model_name = config['name']
        # ckpt_path = config['checkpoint']
        dataset_path = config['dataset_path']

        print(f"\n>> Evaluating Model: {model_name}")

        if not os.path.exists(dataset_path):
            print(f"   [警告] 找不到数据集路径: {dataset_path}，跳过该模型。")
            continue

        current_network = Network(H, W, message_length, noise_layers=[],
                device=device, batch_size=batch_size, lr=0,
                model_name=config['model_name'])

        is_training_free = config.get("training_free", False)

        if is_training_free:
            print(f"  -> [Info] {config['name']} 是无需训练的方法，跳过权重加载。")
        else:
            # 加载权重
            if os.path.exists(config['checkpoint']):
                print(f"Loading weights from: {config['checkpoint']}")
                current_network.load_model_ed(config['checkpoint'])
            else:
                print(f"[Error] Checkpoint not found: {config['checkpoint']}")
                continue

        current_network.encoder_decoder.eval()

        if isinstance(current_network.encoder_decoder, torch.nn.DataParallel):
            core_model = current_network.encoder_decoder.module
        else:
            core_model = current_network.encoder_decoder

        dataset = PhysicalWatermarkDataset(root_dir=dataset_path, H=H, W=W)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"   Loaded {len(dataset)} physical samples.")

        total_ber = 0.0
        exact_match_count = 0  # 记录完全解码正确的图片数量（可选）
        total_samples = 0

        with torch.no_grad():
            for images, messages, need_log in dataloader:
                images = images.to(device)
                messages = messages.to(device)

                if hasattr(core_model, 'decode'):
                    decoded_messages = core_model.decode(images)
                else:
                    decoded_messages = core_model.decoder(images)

                if "MaskWM" in config['model_name']:
                    decoded_messages, mask_pd = decoded_messages

                if "Muti_HiDDeN" in config['model_name']:
                    fused_result, res_result = decoded_messages
                    spa_decoded_messages, dwt_decoded_messages, dct_decoded_messages, svd_decoded_messages = res_result
                    decoded_messages = fused_result

                preds = (decoded_messages > 0.5).float()

                if sum(need_log) > 0:
                    print(need_log)
                    # print(preds)
                    for nl in range(len(need_log)):
                        if need_log[nl] == 1:
                            preds_str = ''
                            for preds_item in preds[nl]:
                                if preds_item > 0.5:
                                    preds_str += '1'
                                else:
                                    preds_str += '0'
                            print(preds_str)

                errors = (preds != messages).float()
                batch_ber = errors.mean().item()

                batch_exact_matches = (errors.sum(dim=1) == 0).sum().item()

                total_ber += batch_ber * images.size(0)
                exact_match_count += batch_exact_matches
                total_samples += images.size(0)

        avg_ber = total_ber / total_samples
        exact_match_rate = exact_match_count / total_samples

        final_results[model_name] = {
            "BER": avg_ber,
            "Exact Match Rate": exact_match_rate,
            "Total Samples": total_samples
        }

        print(f"   [Result] Average BER: {avg_ber:.4f} ({avg_ber * 100:.2f}%)")
        print(f"   [Result] Exact Match Rate: {exact_match_rate:.4f} ({exact_match_rate * 100:.2f}%)")


    print("\n" + "=" * 50)
    print("Physical Robustness Benchmark Summary")
    print("=" * 50)
    print(f"{'Model Name':<30} | {'Avg BER':<10} | {'Avg Acc':<10} | {'Exact Match':<12}")
    print("-" * 55)
    for name, metrics in final_results.items():
        ber_str = f"{metrics['BER'] * 100:.2f}%"
        acc_str = f"{(1 - metrics['BER']) * 100:.2f}%"
        match_str = f"{metrics['Exact Match Rate'] * 100:.5f}%"
        print(f"{name:<30} | {ber_str:<10} | {acc_str:<10} | {match_str:<12}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_physical_dataset()
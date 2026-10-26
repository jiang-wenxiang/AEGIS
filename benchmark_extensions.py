from __future__ import annotations

import copy
import csv
import json
import os
import time
from typing import Any, Dict, Iterable, List



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch import nn

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None


# def build_combined_noise_schedule(repeats: int = 5) -> List[Dict[str, Any]]:
#     combinations = [
#         ("Combo_Rotation_JPEG", "R10_J50", ["Rotation(10)", "Jpeg(50)"]),
#         ("Combo_Scaling_JPEG", "S0.8_J50", ["Scaling(0.8)", "Jpeg(50)"]),
#         ("Combo_Crop_JPEG", "C0.8_J50", ["Crop(0.8, 0.8)", "Jpeg(50)"]),
#         ("Combo_Warp_JPEG", "W0.25_J50", ["Warp(0.25)", "Jpeg(50)"]),
#         ("Combo_Blur_JPEG", "GF1.0_J50", ["GF(1.0)", "Jpeg(50)"]),
#         ("Combo_Noise_JPEG", "GN0.0004_J50", ["GN(0.0004)", "Jpeg(50)"]),
#         ("Combo_Color_JPEG", "CJ0.2_J50", ["ColorJitter(0.2)", "Jpeg(50)"]),
#         (
#             "Combo_Rotation_Scaling_JPEG",
#             "R10_S0.8_J50",
#             ["Rotation(10)", "Scaling(0.8)", "Jpeg(50)"],
#         ),
#         (
#             "Combo_Noise_Blur_JPEG",
#             "GN0.0004_GF1.0_J50",
#             ["GN(0.0004)", "GF(1.0)", "Jpeg(50)"],
#         ),
#         (
#             "Combo_Cropout_JPEG",
#             "CO0.2_J50",
#             ["Cropout(0.2, 0.2)", "Jpeg(50)"],
#         ),
#     ]
#     return [
#         {
#             "type": noise_type,
#             "intensity": intensity,
#             "config_list": config_list,
#             "repeats": int(repeats),
#         }
#         for noise_type, intensity, config_list in combinations
#     ]

def build_combined_noise_schedule(repeats: int = 5) -> List[Dict[str, Any]]:
    combinations = [
        # --- 组别 1: 几何 + 压缩 (Geometric + Compression) ---
        ("Combo_Rotation_JPEG", "R10_J50", ["Rotation(10)", "Jpeg(50)"]),
        ("Combo_Scaling_JPEG", "S0.8_J50", ["Scaling(0.8)", "Jpeg(50)"]),
        ("Combo_Crop_JPEG", "C0.8_J50", ["Crop(0.8, 0.8)", "Jpeg(50)"]),
        ("Combo_Warp_JPEG", "W0.25_J50", ["Warp(0.25)", "Jpeg(50)"]),
        ("Combo_Rotation_Scaling_JPEG", "R10_S0.8_J50", ["Rotation(10)", "Scaling(0.8)", "Jpeg(50)"]),
        ("Combo_Rotation_Crop_JPEG", "R10_C0.8_J50", ["Rotation(10)", "Crop(0.8, 0.8)", "Jpeg(50)"]), # [新增]

        # --- 组别 2: 像素失真/信号 + 压缩 (Pixel & Signal + Compression) ---
        ("Combo_Blur_JPEG", "GF1.0_J50", ["GF(1.0)", "Jpeg(50)"]),
        ("Combo_Noise_JPEG", "GN0.0004_J50", ["GN(0.0004)", "Jpeg(50)"]),
        ("Combo_Color_JPEG", "CJ0.2_J50", ["ColorJitter(0.2)", "Jpeg(50)"]),
        ("Combo_Sharpening_JPEG", "SH1.2_J50", ["Sharpening(1.2)", "Jpeg(50)"]), # [新增]
        ("Combo_Noise_Blur_JPEG", "GN0.0004_GF1.0_J50", ["GN(0.0004)", "GF(1.0)", "Jpeg(50)"]),

        # --- 组别 3: 遮挡/擦除 + 压缩 (Erasure/Occlusion + Compression) ---
        ("Combo_Cropout_JPEG", "CO0.2_J50", ["Cropout(0.2, 0.2)", "Jpeg(50)"]),
        ("Combo_Dropout_JPEG", "DO0.7_J50", ["Dropout(0.7)", "Jpeg(50)"]), # [新增]
        ("Combo_Cropout_Blur_JPEG", "CO0.2_GF1.0_J50", ["Cropout(0.2, 0.2)", "GF(1.0)", "Jpeg(50)"]), # [新增]
    ]
    return [
        {
            "type": noise_type,
            "intensity": intensity,
            "config_list": config_list,
            "repeats": int(repeats),
        }
        for noise_type, intensity, config_list in combinations
    ]

def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(data: Dict[str, Any], path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def _encode(core_model, images, messages, model_config):
    if "ablation_mask" in model_config:
        if hasattr(core_model, "encode"):
            return core_model.encode(
                images,
                messages,
                model_config["ablation_mask"],
                return_weights=False,
            )
        return core_model.encoder(
            images,
            messages,
            model_config["ablation_mask"],
            return_weights=False,
        )
    if hasattr(core_model, "encode"):
        return core_model.encode(images, messages)
    return core_model.encoder(images, messages)


def _decode(core_model, images, model_config):
    decoded = (
        core_model.decode(images)
        if hasattr(core_model, "decode")
        else core_model.decoder(images)
    )
    if "MaskWM" in model_config['model_name']:
        decoded, _ = decoded
    if "Muti_HiDDeN" in model_config["model_name"]:
        decoded, _ = decoded
    return decoded


def _parameter_count(module):
    if module is None:
        return 0, 0
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return int(total), int(trainable)


def _storage_bytes(module):
    if module is None:
        return 0
    parameters = sum(
        tensor.numel() * tensor.element_size()
        for tensor in module.parameters()
    )
    buffers = sum(
        tensor.numel() * tensor.element_size()
        for tensor in module.buffers()
    )
    return int(parameters + buffers)


class _EncoderProfile(nn.Module):
    def __init__(self, core_model, model_config):
        super().__init__()
        self.core_model = core_model
        self.model_config = model_config

    def forward(self, image, message):
        return _encode(self.core_model, image, message, self.model_config)


class _DecoderProfile(nn.Module):
    def __init__(self, core_model, model_config):
        super().__init__()
        self.core_model = core_model
        self.model_config = model_config

    def forward(self, image):
        return _decode(self.core_model, image, self.model_config)


def _elapsed_ms(function, use_cuda: bool, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    if use_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            function()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / repeats)
    start_time = time.perf_counter()
    for _ in range(repeats):
        function()
    return float((time.perf_counter() - start_time) * 1000.0 / repeats)


def _save_complexity_csv(results, path):
    fields = [
        "model", "encoder_params", "decoder_params", "total_params",
        "trainable_params", "encoder_macs", "decoder_macs", "total_macs",
        "total_flops", "encoder_latency_ms", "decoder_latency_ms",
        "single_image_latency_ms", "throughput_images_per_second",
        "model_storage_mb", "peak_gpu_memory_mb",
        "process_peak_gpu_memory_mb", "incremental_gpu_memory_mb",
        "device", "pytorch_version",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({
                field: name if field == "model" else result.get(field)
                for field in fields
            })


def benchmark_model_complexity(
    core_model,
    model_config,
    model_id: str,
    test_dataloader,
    device,
    message_length: int,
    output_dir: str = "results",
    warmup: int = 20,
    repeats: int = 100,
):
    sample_image = next(iter(test_dataloader))[:1].to(device)
    sample_message = torch.randint(
        0,
        2,
        (1, message_length),
        device=device,
        dtype=torch.float32,
    )
    encoder_module = getattr(core_model, "encoder", None)
    decoder_module = getattr(core_model, "decoder", None)
    encoder_params, encoder_trainable = _parameter_count(encoder_module)
    decoder_params, decoder_trainable = _parameter_count(decoder_module)
    storage_mb = (
        _storage_bytes(encoder_module) + _storage_bytes(decoder_module)
    ) / (1024 ** 2)

    core_model.eval()
    with torch.inference_mode():
        encoded = _encode(core_model, sample_image, sample_message, model_config)
        if not torch.is_tensor(encoded):
            raise TypeError("Encoder must return an image tensor for profiling.")
        encoder_call = lambda: _encode(
            core_model, sample_image, sample_message, model_config
        )
        decoder_call = lambda: _decode(core_model, encoded, model_config)
        full_call = lambda: _decode(
            core_model,
            _encode(core_model, sample_image, sample_message, model_config),
            model_config,
        )
        use_cuda = sample_image.is_cuda
        encoder_ms = _elapsed_ms(encoder_call, use_cuda, warmup, repeats)
        decoder_ms = _elapsed_ms(decoder_call, use_cuda, warmup, repeats)
        total_ms = _elapsed_ms(full_call, use_cuda, warmup, repeats)

        process_peak_mb = incremental_mb = model_peak_mb = None
        if use_cuda:
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated(sample_image.device)
            torch.cuda.reset_peak_memory_stats(sample_image.device)
            full_call()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(sample_image.device)
            process_peak_mb = float(peak / (1024 ** 2))
            incremental_mb = float(max(0, peak - baseline) / (1024 ** 2))
            model_peak_mb = float(storage_mb + incremental_mb)

        encoder_macs = decoder_macs = None
        flop_note = "Install THOP with `pip install thop` to enable FLOPs."
        if thop_profile is not None:
            try:
                encoder_macs, _ = thop_profile(
                    _EncoderProfile(core_model, model_config).eval(),
                    inputs=(sample_image, sample_message),
                    verbose=False,
                )
                decoder_macs, _ = thop_profile(
                    _DecoderProfile(core_model, model_config).eval(),
                    inputs=(encoded,),
                    verbose=False,
                )
                encoder_macs = float(encoder_macs)
                decoder_macs = float(decoder_macs)
                flop_note = (
                    "THOP MACs; FLOPs=2*MACs. Unsupported custom operations "
                    "may be omitted."
                )
            except Exception as error:
                flop_note = f"THOP failed: {error}"

    total_macs = (
        None
        if encoder_macs is None
        else encoder_macs + decoder_macs
    )
    result = {
        "input_resolution": list(sample_image.shape),
        "message_length": int(message_length),
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "total_params": encoder_params + decoder_params,
        "trainable_params": encoder_trainable + decoder_trainable,
        "encoder_macs": encoder_macs,
        "decoder_macs": decoder_macs,
        "total_macs": total_macs,
        "total_flops": None if total_macs is None else 2.0 * total_macs,
        "encoder_latency_ms": encoder_ms,
        "decoder_latency_ms": decoder_ms,
        "single_image_latency_ms": total_ms,
        "throughput_images_per_second": float(1000.0 / total_ms),
        "model_storage_mb": float(storage_mb),
        "peak_gpu_memory_mb": model_peak_mb,
        "process_peak_gpu_memory_mb": process_peak_mb,
        "incremental_gpu_memory_mb": incremental_mb,
        "device": (
            torch.cuda.get_device_name(sample_image.device)
            if use_cuda
            else "CPU"
        ),
        "pytorch_version": torch.__version__,
        "timing_warmup": int(warmup),
        "timing_repeats": int(repeats),
        "flops_note": flop_note,
    }

    json_path = os.path.join(output_dir, "model_complexity.json")
    results = _load_json(json_path)
    results[model_id] = result
    _save_json(results, json_path)
    _save_complexity_csv(results, os.path.join(output_dir, "model_complexity.csv"))
    flops_text = (
        "N/A"
        if result["total_flops"] is None
        else f"{result['total_flops'] / 1e9:.3f}"
    )
    print(
        f"  -> Complexity: {result['total_params'] / 1e6:.3f} M params | "
        f"{flops_text} GFLOPs | "
        f"{result['single_image_latency_ms']:.3f} ms/image | "
        f"peak VRAM={result['peak_gpu_memory_mb']} MB"
    )
    return result


def _summary(values: Iterable[float], lower_is_better: bool):
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    q1, median, q3 = np.percentile(array, [25, 50, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = array[(array < lower_fence) | (array > upper_fence)]
    minimum, maximum = float(array.min()), float(array.max())
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "best": minimum if lower_is_better else maximum,
        "worst": maximum if lower_is_better else minimum,
        "min": minimum,
        "max": maximum,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_fence": float(lower_fence),
        "upper_fence": float(upper_fence),
        "outlier_count": int(outliers.size),
        "outlier_rate": float(outliers.size / array.size),
        "outliers": outliers.tolist(),
    }


def _save_statistics_csv(results, path):
    fields = [
        "model", "noise_type", "intensity", "config", "metric", "count",
        "mean", "std", "best", "worst", "min", "max", "median", "q1",
        "q3", "iqr", "outlier_count", "outlier_rate",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for model_name, model_results in results.items():
            for noise_type, noise_results in model_results.items():
                for intensity, result in noise_results.items():
                    config = " -> ".join(result.get("config_list", []))
                    for metric, stats in result.get("metrics", {}).items():
                        row = {
                            "model": model_name,
                            "noise_type": noise_type,
                            "intensity": intensity,
                            "config": config,
                            "metric": metric,
                        }
                        row.update({field: stats.get(field) for field in fields if field not in row})
                        writer.writerow(row)


# def _plot_boxplot(model_id, distributions, output_dir):
#     if not distributions:
#         return
#     os.makedirs(output_dir, exist_ok=True)
#     labels = list(distributions)
#     values = [distributions[label] for label in labels]
#     figure, axis = plt.subplots(figsize=(max(12, 1.6 * len(labels)), 7))
#     axis.boxplot(values, labels=labels, showfliers=True, whis=1.5)
#     axis.set_ylabel("Bit error rate")
#     axis.set_title(f"{model_id}: combined-noise BER distributions")
#     axis.grid(axis="y", linestyle="--", alpha=0.35)
#     axis.tick_params(axis="x", rotation=35)
#     figure.tight_layout()
#     safe_name = "".join(
#         character if character.isalnum() or character in "-_" else "_"
#         for character in model_id
#     )
#     figure.savefig(
#         os.path.join(output_dir, f"{safe_name}_combined_noise_boxplot.png"),
#         dpi=300,
#     )
#     figure.savefig(
#         os.path.join(output_dir, f"{safe_name}_combined_noise_boxplot.svg")
#     )
#     plt.close(figure)
#
#
# def benchmark_combined_noises(
#     core_model,
#     current_network,
#     model_config,
#     model_id: str,
#     test_dataloader,
#     device,
#     message_length: int,
#     noise_class,
#     criterion_mse,
#     criterion_lpips,
#     schedule=None,
#     output_dir: str = "results",
#     wandb_module=None,
# ):
#     """Evaluate sequential combined attacks and persist repeated statistics."""
#     schedule = schedule or build_combined_noise_schedule()
#     statistics_path = os.path.join(output_dir, "robustness_statistics.json")
#     observations_path = os.path.join(
#         output_dir, "combined_noise_observations.json"
#     )
#     statistics = _load_json(statistics_path)
#     observations = _load_json(observations_path)
#     statistics.setdefault(model_id, {})
#     observations.setdefault(model_id, {})
#     distributions = {}
#
#     for item in schedule:
#         noise_type = item["type"]
#         intensity = item["intensity"]
#         configs = item["config_list"]
#         repeats = int(item.get("repeats", 5))
#         label = " + ".join(configs)
#         print(f"  -> Testing combined attack {label} x {repeats} ...", end="")
#         try:
#             noise = noise_class(copy.deepcopy(configs)).to(device)
#             noise.eval()
#         except Exception as error:
#             print(f" initialization failed: {error}")
#             continue
#
#         values = {
#             "error_rate": [],
#             "psnr": [],
#             "ssim": [],
#             "lpips": [],
#             "encoder_loss": [],
#             "decoder_loss": [],
#         }
#         # Imported lazily so the extension module remains reusable for complexity-only runs.
#         import kornia
#         with torch.inference_mode():
#             for images in test_dataloader:
#                 images = images.to(device)
#                 messages = torch.randint(
#                     0,
#                     2,
#                     (images.shape[0], message_length),
#                     device=device,
#                     dtype=torch.float32,
#                 )
#                 encoded = _encode(core_model, images, messages, model_config)
#                 if not torch.is_tensor(encoded):
#                     raise TypeError(f"Encoder of {model_id} must return a tensor.")
#                 values["psnr"].append(
#                     kornia.metrics.psnr(encoded, images, 2).mean().item()
#                 )
#                 values["ssim"].append(
#                     kornia.metrics.ssim(encoded, images, 5).mean().item()
#                 )
#                 values["lpips"].append(
#                     criterion_lpips(encoded, images).mean().item()
#                 )
#                 values["encoder_loss"].append(
#                     criterion_mse(encoded, images).item()
#                 )
#                 for _ in range(repeats):
#                     noised = noise([encoded, images])
#                     decoded = _decode(core_model, noised, model_config)
#                     values["decoder_loss"].append(
#                         criterion_mse(decoded, messages).item()
#                     )
#                     for index in range(messages.shape[0]):
#                         ber = current_network.decoded_message_error_rate_batch(
#                             messages[index:index + 1],
#                             decoded[index:index + 1],
#                         )
#                         values["error_rate"].append(float(ber))
#
#         metric_stats = {
#             metric: _summary(samples, metric not in {"psnr", "ssim"})
#             for metric, samples in values.items()
#         }
#         statistics[model_id].setdefault(noise_type, {})[str(intensity)] = {
#             "config_list": configs,
#             "is_combination": True,
#             "repeats": repeats,
#             "metrics": metric_stats,
#         }
#         distributions[label] = values["error_rate"]
#         observations[model_id][label] = values["error_rate"]
#         ber = metric_stats["error_rate"]
#         print(
#             f" Done. BER={ber['mean']:.4f} +/- {ber['std']:.4f}, "
#             f"best={ber['best']:.4f}, worst={ber['worst']:.4f}, "
#             f"outliers={ber['outlier_count']}"
#         )
#         if wandb_module is not None:
#             wandb_module.log({
#                 f"{noise_type}/error_rate": ber["mean"],
#                 f"{noise_type}/error_rate_std": ber["std"],
#                 f"{noise_type}/error_rate_best": ber["best"],
#                 f"{noise_type}/error_rate_worst": ber["worst"],
#                 f"{noise_type}/outlier_rate": ber["outlier_rate"],
#             })
#         _save_json(statistics, statistics_path)
#         _save_json(observations, observations_path)
#         _save_statistics_csv(
#             statistics,
#             os.path.join(output_dir, "robustness_statistics.csv"),
#         )
#
#     _plot_boxplot(
#         model_id,
#         distributions,
#         os.path.join(output_dir, "outlier_boxplots"),
#     )
#     return statistics.get(model_id, {})

def benchmark_combined_noises(
        core_model,
        current_network,
        model_config,
        model_id: str,
        test_dataloader,
        device,
        message_length: int,
        noise_class,
        criterion_mse,
        criterion_lpips,
        schedule=None,
        output_dir: str = "results",
        wandb_module=None,
):
    schedule = schedule or build_combined_noise_schedule()
    statistics_path = os.path.join(output_dir, "robustness_statistics.json")
    observations_path = os.path.join(
        output_dir, "combined_noise_observations.json"
    )
    statistics = _load_json(statistics_path)
    observations = _load_json(observations_path)
    statistics.setdefault(model_id, {})
    observations.setdefault(model_id, {})
    distributions = {}

    for item in schedule:
        noise_type = item["type"]
        intensity = item["intensity"]
        configs = item["config_list"]
        repeats = int(item.get("repeats", 5))
        label = " + ".join(configs)
        print(f"  -> Testing combined attack {label} x {repeats} ...", end="")
        try:
            noise = noise_class(copy.deepcopy(configs)).to(device)
            noise.eval()
        except Exception as error:
            print(f" initialization failed: {error}")
            continue

        values = {
            "error_rate": [],
            "psnr": [],
            "ssim": [],
            "lpips": [],
            "encoder_loss": [],
            "decoder_loss": [],
        }
        import kornia
        with torch.inference_mode():
            for images in test_dataloader:
                images = images.to(device)
                messages = torch.randint(
                    0,
                    2,
                    (images.shape[0], message_length),
                    device=device,
                    dtype=torch.float32,
                )
                encoded = _encode(core_model, images, messages, model_config)
                if not torch.is_tensor(encoded):
                    raise TypeError(f"Encoder of {model_id} must return a tensor.")
                values["psnr"].append(
                    kornia.metrics.psnr(encoded, images, 2).mean().item()
                )
                values["ssim"].append(
                    kornia.metrics.ssim(encoded, images, 5).mean().item()
                )
                values["lpips"].append(
                    criterion_lpips(encoded, images).mean().item()
                )
                values["encoder_loss"].append(
                    criterion_mse(encoded, images).item()
                )
                for _ in range(repeats):
                    noised = noise([encoded, images])
                    decoded = _decode(core_model, noised, model_config)
                    values["decoder_loss"].append(
                        criterion_mse(decoded, messages).item()
                    )
                    for index in range(messages.shape[0]):
                        ber = current_network.decoded_message_error_rate_batch(
                            messages[index:index + 1],
                            decoded[index:index + 1],
                        )
                        values["error_rate"].append(float(ber))

        metric_stats = {
            metric: _summary(samples, metric not in {"psnr", "ssim"})
            for metric, samples in values.items()
        }
        statistics[model_id].setdefault(noise_type, {})[str(intensity)] = {
            "config_list": configs,
            "is_combination": True,
            "repeats": repeats,
            "metrics": metric_stats,
        }
        distributions[label] = values["error_rate"]
        observations[model_id][label] = values["error_rate"]
        ber = metric_stats["error_rate"]
        print(
            f" Done. BER={ber['mean']:.4f} +/- {ber['std']:.4f}, "
            f"best={ber['best']:.4f}, worst={ber['worst']:.4f}, "
            f"outliers={ber['outlier_count']}"
        )
        if wandb_module is not None:
            wandb_module.log({
                f"{noise_type}/error_rate": ber["mean"],
                f"{noise_type}/error_rate_std": ber["std"],
                f"{noise_type}/error_rate_best": ber["best"],
                f"{noise_type}/error_rate_worst": ber["worst"],
                f"{noise_type}/outlier_rate": ber["outlier_rate"],
            })
        _save_json(statistics, statistics_path)
        _save_json(observations, observations_path)
        _save_statistics_csv(
            statistics,
            os.path.join(output_dir, "robustness_statistics.csv"),
        )

    return statistics.get(model_id, {})


def get_model_value_name(model_name, cat_name):
    res_models = ["AEGIS(Ours)"]
    to_models = ["AEGIS"]
    value_cat_name = []
    if model_name in res_models and cat_name in value_cat_name:
        return to_models[res_models.index(model_name)]
    else:
        return model_name

def plot_cross_method_combined_boxplots(observations_path: str, output_dir: str):
    observations = _load_json(observations_path)
    if not observations:
        return

    models = list(observations.keys())

    if "AEGIS" in models:
        models.remove("AEGIS")

    if not models:
        return

    categories = {
        "Geometric_and_Compression": [
            "Rotation(10) + Jpeg(50)",
            "Scaling(0.8) + Jpeg(50)",
            "Crop(0.8, 0.8) + Jpeg(50)",
            "Warp(0.25) + Jpeg(50)",
            "Rotation(10) + Scaling(0.8) + Jpeg(50)",
            "Rotation(10) + Crop(0.8, 0.8) + Jpeg(50)"
        ],
        "Pixel_Signal_and_Compression": [
            "GF(1.0) + Jpeg(50)",
            "GN(0.0004) + Jpeg(50)",
            "ColorJitter(0.2) + Jpeg(50)",
            "Sharpening(1.2) + Jpeg(50)",
            "GN(0.0004) + GF(1.0) + Jpeg(50)"
        ],
        "Erasure_and_Compression": [
            "Cropout(0.2, 0.2) + Jpeg(50)",
            "Dropout(0.7) + Jpeg(50)",
            "Cropout(0.2, 0.2) + GF(1.0) + Jpeg(50)"
        ]
    }

    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.get_cmap('tab10').colors

    for cat_name, combos in categories.items():
        pooled_values = []
        plotted_models = []
        for model in models:
            model_values = []
            model_observations = observations.get(get_model_value_name(model, cat_name), {})
            for combo in combos:
                samples = model_observations.get(combo, [])
                if isinstance(samples, (list, tuple)):
                    model_values.extend(
                        1.0 - float(value) for value in samples if value is not None
                    )

            if model_values:
                pooled_values.append(model_values)
                plotted_models.append(model)

        if not pooled_values:
            print(f"[Warning] No observations found for category: {cat_name}")
            continue

        fig_height = max(5, len(plotted_models) * 0.65 + 2)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        positions = list(range(1, len(plotted_models) + 1))
        MIN_VISUAL_BOX_WIDTH = 0.008
        BOX_HEIGHT = 0.28
        bp = ax.boxplot(
            pooled_values,
            positions=positions,
            vert=False,
            widths=0.28,
            patch_artist=True,
            flierprops=dict(
                marker='o',
                markerfacecolor='#0b0b0b',
                markeredgecolor='#0b0b0b',
                markeredgewidth=0.3,
                markersize=1.4,
                alpha=0.9,
                linestyle='none',
            ),
            whis=1.5,
        )

        for index, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[index % len(colors)])

        for median in bp['medians']:
            median.set_color('white')
            median.set_linewidth(1.2)

        for index, values in enumerate(pooled_values):
            array = np.asarray(values, dtype=float)
            array = array[np.isfinite(array)]

            if array.size == 0:
                continue

            q1, q3 = np.percentile(array, [25, 75])
            actual_width = q3 - q1

            if actual_width < MIN_VISUAL_BOX_WIDTH:
                mean_value = float(np.mean(array))
                half_width = MIN_VISUAL_BOX_WIDTH / 2.0

                # 在 0 或 1 附近裁剪，因此端点处只显示半个箱体
                left = max(0.0, mean_value - half_width)
                right = min(1.0, mean_value + half_width)

                rectangle = Rectangle(
                    (left, positions[index] - BOX_HEIGHT / 2.0),
                    right - left,
                    BOX_HEIGHT,
                    facecolor=colors[index % len(colors)],
                    edgecolor='black',
                    linewidth=0.8,
                    zorder=3,
                )
                ax.add_patch(rectangle)

                ax.vlines(
                    mean_value,
                    positions[index] - BOX_HEIGHT / 2.0,
                    positions[index] + BOX_HEIGHT / 2.0,
                    colors='white',
                    linewidth=1.2,
                    zorder=4,
                )

        ax.set_yticks(positions)
        ax.set_yticklabels(plotted_models)
        ax.set_xlabel("Bit Accuracy")
        ax.set_xlim(0.0, 1.0)

        display_title = cat_name.replace("_", " ")
        ax.set_title(
            f"Cross-Method Robustness: {display_title}\n"
            f"Pooled distributions from {len(combos)} combined attacks",
            pad=15,
        )
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{cat_name}_boxplot.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(output_dir, f"{cat_name}_boxplot.svg"), bbox_inches='tight')
        plt.close(fig)

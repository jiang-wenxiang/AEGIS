import torch
import numpy as np
import cv2
import os


def save_attention_maps(weights, save_dir="results/attention_maps", batch_idx=0, prefix="", original_images=None):

    os.makedirs(save_dir, exist_ok=True)

    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    orig_imgs_np = None
    if original_images is not None:
        imgs = (original_images.detach().cpu().permute(0, 2, 3, 1).numpy() + 1) / 2.0 * 255
        orig_imgs_np = imgs.astype(np.uint8)
        orig_imgs_np = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in orig_imgs_np]

    # 判断维度
    if len(weights.shape) == 4:
        B, N, H, W = weights.shape
        is_spatial = True
    elif len(weights.shape) == 2:
        B, N = weights.shape
        is_spatial = False
        H, W = 128, 128
    else:
        return

    domain_names = ["SVD", "DCT", "DWT", "Spatial"]

    for b in range(B):
        base_img = orig_imgs_np[b] if (orig_imgs_np is not None and is_spatial) else None
        if base_img is not None:
            base_img = cv2.resize(base_img, (W, H))

        for d in range(N):
            domain_name = domain_names[d] if d < len(domain_names) else f"Domain_{d}"

            if is_spatial:
                w_map = weights[b, d, :, :]
            else:
                val = weights[b, d]
                w_map = np.full((H, W), val)

            norm_map = (w_map * 255).astype(np.uint8)

            heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

            filename_heat = f"{prefix}img{batch_idx + b}_{domain_name}_heatmap.png"
            cv2.imwrite(os.path.join(save_dir, filename_heat), heatmap)

            if base_img is not None and is_spatial:
                overlay = cv2.addWeighted(heatmap, 0.5, base_img, 0.5, 0)

                filename_overlay = f"{prefix}img{batch_idx + b}_{domain_name}_overlay.png"
                cv2.imwrite(os.path.join(save_dir, filename_overlay), overlay)
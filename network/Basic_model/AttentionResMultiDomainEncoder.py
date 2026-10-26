from .options import HiDDenConfiguration
from .HaarDWT import HaarIDWT, HaarDWT
from .encoder import ResFlexibleEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct  # pip install torch-dct
from .Attention import SpatialAttentionFusion, GlobalAttentionFusion


class AttentionResMultiDomainEncoder(nn.Module):
    def __init__(self, config: HiDDenConfiguration, sp_attention=True, dl_res_attention=False):
        super(AttentionResMultiDomainEncoder, self).__init__()

        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        # --- 1. SVD 编码器 (最深层, 处理 32x32) ---
        # 注意：我们复用 FlexibleEncoder。
        # SVD 的 S 矩阵通常是对角阵。我们将其视为一个通道数为 3 的特征图进行微调。
        self.svd_layer = ResFlexibleEncoder(config, in_channels=3, out_channels=3, res_output=True)

        # --- 2. DCT 编码器 (处理 LL1, 64x64) ---
        # 输入是频域系数，也是 3 通道
        self.dct_layer = ResFlexibleEncoder(config, in_channels=3, out_channels=3, res_output=True)

        # --- 3. DWT 编码器 (处理 LL1, 64x64) ---
        # 输入是 LL1 子带 (空间近似)
        self.dwt_layer = ResFlexibleEncoder(config, in_channels=3, out_channels=3, res_output=True)

        # --- 4. Spatial 编码器 (HiDDeN 原版, 处理 128x128) ---
        self.spatial_layer = ResFlexibleEncoder(config, in_channels=3, out_channels=3, res_output=True)

        self.sp_attention = sp_attention

        if sp_attention:
            self.attention = SpatialAttentionFusion()
        else:
            self.attention = GlobalAttentionFusion()

    def forward(self, image, message, ablation_mask=None, return_weights=False):
        # image: [B, 3, 128, 128]

        # === 1. 下采样阶段 ===
        # Level 1 DWT (128 -> 64)
        LL1, coeffs1 = self.dwt(image)

        # Level 2 DWT (64 -> 32)
        LL2, coeffs2 = self.dwt(LL1)

        # === 2. SVD 嵌入阶段 (在 32x32 上) ===
        # U: [B, 3, 32, 32], S: [B, 3, 32], V: [B, 3, 32, 32]
        U, S, Vh = torch.linalg.svd(LL2)

        # 技巧：将 S 向量扩展成对角矩阵图，作为 Encoder 的输入
        S_diag = torch.zeros_like(LL2)
        for i in range(S.shape[-1]):
            S_diag[:, :, i, i] = S[:, :, i]
        # 使用 Encoder 修改奇异值矩阵
        # 这里 S_diag 是 [B, 3, 32, 32]，完全符合 Encoder 的输入要求

        S_embedded = self.svd_layer(S_diag, S_diag, message)
        # 逆 SVD 合成修改后的 LL2
        LL2_svd = U @ S_embedded @ Vh
        # === 3. 恢复与 DCT 嵌入阶段 ===
        # 逆 DWT 回到 64x64
        LL1_svd = self.idwt(LL2_svd, coeffs2)
        img_svd = self.idwt(LL1_svd, coeffs1)
        # DCT 变换
        # Delta_LL1_dct = dct.dct_2d(LL1_recon)

        Ori_LL1_dct = dct.dct_2d(LL1)
        # 残差链接
        # LL1_dct = torch.cat((Ori_LL1_dct, Delta_LL1_dct), dim=1)
        # 使用 Encoder 修改 DCT 系数
        LL1_dct_mod = self.dct_layer(Ori_LL1_dct, Ori_LL1_dct, message)

        # 逆 DCT
        LL1_dct = dct.idct_2d(LL1_dct_mod)
        img_dct = self.idwt(LL1_dct, coeffs1)

        # === 4. DWT 域嵌入阶段 ===
        # 残差链接
        # LL1_mod = torch.cat((LL1, LL1_mod_from_dct), dim=1)
        # 直接在 LL1 (64x64) 上再次嵌入
        LL1_dwt = self.dwt_layer(LL1, LL1, message)

        # === 5. 恢复与空间域嵌入阶段 ===
        # 逆 DWT 回到 128x128
        img_dwt = self.idwt(LL1_dwt, coeffs1)

        # 残差链接
        # image_mod = torch.cat((image, image_recon), dim=1)
        # 最后一次使用原版逻辑嵌入
        img_img = self.spatial_layer(image, image, message)

        output_image, weights = self.attention([img_svd, img_dct, img_dwt, img_img], ablation_mask=ablation_mask)

        if return_weights:
            return output_image, weights

        return output_image

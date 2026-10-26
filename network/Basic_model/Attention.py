import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionFusion(nn.Module):
    def __init__(self, in_channels=3, num_images=4):
        """
        Args:
            in_channels: 图像的通道数，通常为3 (RGB)
            num_images: 输入图像的数量，这里是4
        """
        super(SpatialAttentionFusion, self).__init__()

        # 1. 特征提取层
        # 输入通道数 = 图像通道数 * 图像数量 (例如 3 * 4 = 12)
        # 我们把4张图在通道维度拼接，让卷积层同时看到4张图的信息
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels * num_images, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 2. 权重生成层
            # 输出通道数 = 图像数量 (4)，代表每张图在每个像素点的权重
            nn.Conv2d(16, num_images, kernel_size=1)
        )

        self.softmax = nn.Softmax(dim=1)

        # === 【在这里添加初始化】 ===
        # 只对生成权重的最后一层卷积进行零初始化
        # 这样初始状态下，所有图片的权重都是 1/num_images (即平均融合)
        nn.init.constant_(self.conv_block[-1].weight, 0)
        nn.init.constant_(self.conv_block[-1].bias, 0)

    def forward(self, images_list, ablation_mask=None):
        """
        Args:
            images_list: 一个包含4个Tensor的列表，每个Tensor形状为 (B, C, H, W)
                         或者一个堆叠后的Tensor，形状为 (B, 4, C, H, W)
        """
        # 如果输入是列表，先转换为 (B, 4, C, H, W)
        if isinstance(images_list, list):
            # stack: (B, 4, C, H, W)
            stack_imgs = torch.stack(images_list, dim=1)
        else:
            stack_imgs = images_list

        B, N, C, H, W = stack_imgs.shape

        # 1. 准备输入数据：将4张图在通道维度拼接
        # reshape: (B, 4*C, H, W) -> 例如 (B, 12, 256, 256)
        concat_input = stack_imgs.view(B, N * C, H, W)

        # 2. 生成注意力分数 (Logits)
        # output: (B, 4, H, W)
        attention_logits = self.conv_block(concat_input)

        # 3. 计算归一化权重
        # Softmax 保证了对于每个像素点，4张图的权重之和为1
        # attention_weights: (B, 4, H, W)
        attention_weights = self.softmax(attention_logits)

        # === 【修改点 2：应用消融掩码】 ===
        if ablation_mask is not None:
            # 确保 mask 是 tensor 并移动到正确的设备
            if not isinstance(ablation_mask, torch.Tensor):
                mask = torch.tensor(ablation_mask, device=stack_imgs.device, dtype=stack_imgs.dtype)
            else:
                mask = ablation_mask.to(stack_imgs.device)

            # mask shape: (4) -> (1, 4, 1, 1) 以便广播
            mask = mask.view(1, N, 1, 1)

            # 1. 强制置零
            attention_weights = attention_weights * mask

            # 2. 重归一化 (防止图像变暗)，加上 epsilon 防止除零
            sum_weights = torch.sum(attention_weights, dim=1, keepdim=True) + 1e-8
            attention_weights = attention_weights / sum_weights

        # 4. 扩展权重维度以便进行广播乘法
        # (B, 4, H, W) -> (B, 4, 1, H, W) 以便匹配图像的 (B, 4, 3, H, W)
        attention_weights_expanded = attention_weights.unsqueeze(2)

        # 5. 加权融合
        # (B, 4, 3, H, W) * (B, 4, 1, H, W) -> (B, 4, 3, H, W)
        weighted_imgs = stack_imgs * attention_weights_expanded

        # 6. 在“图像数量”维度求和，得到最终图像
        # (B, 4, 3, H, W) -> (B, 3, H, W)
        fused_image = torch.sum(weighted_imgs, dim=1)

        return fused_image, attention_weights


class GlobalAttentionFusion(nn.Module):
    def __init__(self, in_channels=3, num_images=4):
        super(GlobalAttentionFusion, self).__init__()

        # 全局平均池化 + MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * num_images, 128),
            nn.ReLU(),
            nn.Linear(128, num_images),  # 输出4个数值
            nn.Softmax(dim=1)
        )

        # === 【在这里添加初始化】 ===
        # 注意：self.mlp 是一个 Sequential 容器
        # 我们需要索引到第 2 层 (即最后一个 Linear 层)
        # 索引 0 是 Linear, 1 是 ReLU, 2 是 Linear
        last_linear_layer = self.mlp[2]

        # 零初始化：确保初始 Logits 全为 0
        nn.init.constant_(last_linear_layer.weight, 0)
        nn.init.constant_(last_linear_layer.bias, 0)

    def forward(self, images_list, ablation_mask=None):
        if isinstance(images_list, list):
            stack_imgs = torch.stack(images_list, dim=1)
        else:
            stack_imgs = images_list

        B, N, C, H, W = stack_imgs.shape

        # (B, N*C, H, W)
        concat_input = stack_imgs.view(B, N * C, H, W)

        # 1. 全局池化 -> (B, N*C, 1, 1) -> (B, N*C)
        global_features = torch.mean(concat_input, dim=[2, 3])

        # 2. 计算权重 -> (B, N)
        weights = self.mlp(global_features)

        # === 【修改点 4：应用消融掩码】 ===
        if ablation_mask is not None:
            if not isinstance(ablation_mask, torch.Tensor):
                mask = torch.tensor(ablation_mask, device=stack_imgs.device, dtype=stack_imgs.dtype)
            else:
                mask = ablation_mask.to(stack_imgs.device)

            # mask shape: (4) -> (1, 4)
            mask = mask.view(1, N)

            # 强制置零并重归一化
            weights = weights * mask
            sum_weights = torch.sum(weights, dim=1, keepdim=True) + 1e-8
            weights = weights / sum_weights

        # 3. 维度调整以便乘法 -> (B, N, 1, 1, 1)
        weights_expanded = weights.view(B, N, 1, 1, 1)

        # 4. 加权求和
        fused_image = torch.sum(stack_imgs * weights_expanded, dim=1)

        return fused_image, weights

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PhysicalWatermarkDataset(Dataset):
    def __init__(self, root_dir, H=128, W=128, log_list=['_p3_patch_24.png', '_p2_patch_44.png']):
        """
        Args:
            root_dir: 包含 images/ 和 dataset.json 的根目录
            H, W: 目标尺寸 (默认 128)
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.json_path = os.path.join(root_dir, "dataset.json")

        # 1. 内置与 MBRSDataset 一致的 Transform
        # 物理拍摄图片通常已经是裁切好的，直接 Resize 即可
        self.transform = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            # [关键] 保持 [-1, 1] 归一化，与训练一致
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        with open(self.json_path, 'r') as f:
            self.metadata = json.load(f)

        self.log_list = log_list

        self.image_filenames = list(self.metadata.keys())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)

        need_log = 0

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            # 读取 Message 字符串 "0101..." 转为 Tensor
            msg_str = self.metadata[img_name]['message']
            message = torch.tensor([int(bit) for bit in msg_str], dtype=torch.float32)

            # print(img_name[len(img_name) - 16:])
            if img_name[len(img_name) - 16:] in self.log_list:
                print(img_name)
                print(msg_str)
                need_log = 1

            return image, message, need_log

        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            # 简单的容错处理：递归返回下一个
            return self.__getitem__((idx + 1) % len(self))

# # 测试 MBRS
# dataset = PhysicalWatermarkDataset(root_dir="dataset_final/MBRS_ResNet")
#
# # 测试 HiDDeN
# dataset = PhysicalWatermarkDataset(root_dir="dataset_final/HiDDeN")

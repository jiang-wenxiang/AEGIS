import torch
import cv2
import numpy as np
import os
import argparse

# 复用您修改后的 ModelTrainer，它包含了模型工厂模式
from network.ModelTrainer import Network

# [修改] 引入 Noise 类，以便支持动态构建噪声层
# 假设您的文件结构是 network/Noise.py，如果在 utils 下请修改为 from utils.Noise import Noise
from network.Noise import Noise

class WatermarkEngine:
    def __init__(self, model_path, settings, device, noise_layers=None):

        self.device = device
        self.settings = settings
        self.H = settings['H']
        self.W = settings['W']
        self.message_length = settings['message_length']
        self.model_name = settings['model_name']

        self.default_noise_layers = noise_layers if noise_layers else []

        print(f"Initializing model: {self.model_name}...")
        self.net = Network(
            H=self.H,
            W=self.W,
            message_length=self.message_length,
            noise_layers=noise_layers if noise_layers else [],
            device=device,
            batch_size=1,
            lr=0,
            model_name=self.model_name,
            with_discriminator=False
        )

        if os.path.exists(model_path):
            print(f"Loading weights from {model_path}...")
            self.net.load_model_ed(model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.net.encoder_decoder.eval()
        for p in self.net.encoder_decoder.parameters():
            p.requires_grad = False

    def _preprocess_image(self, image_source):
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
            if img is None: raise ValueError(f"Cannot read image: {image_source}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_source

        # Resize
        img = cv2.resize(img, (self.W, self.H))
        # Normalize & Tensor
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device), img  # 返回 Tensor 和 原始 resized numpy

    def _postprocess_image(self, tensor):
        """内部工具：将 Tensor 转换为可保存的 uint8 图像"""
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def encode(self, image_source, message_bits, strength_factor=1.0):

        if len(message_bits) != self.message_length:
            raise ValueError(f"Message length mismatch! Expected {self.message_length}, got {len(message_bits)}")

        image_tensor, original_img_np = self._preprocess_image(image_source)
        message_tensor = torch.tensor(message_bits, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():

            encoder = self.net.encoder_decoder.module.encoder

            encoded_tensor = encoder(image_tensor, message_tensor)

            residual = encoded_tensor - image_tensor
            final_tensor = image_tensor + residual * strength_factor

        return self._postprocess_image(final_tensor)

    def attack(self, image_source, specific_layers=["Warp(0.3)"], attack_factor=1.0):

        image_tensor, _ = self._preprocess_image(image_source)

        current_noise_module = Noise(specific_layers).to(self.device)

        current_noise_module.eval()

        with torch.no_grad():

            noised_tensor = current_noise_module([image_tensor, image_tensor])

            residual = noised_tensor - image_tensor
            final_noised_tensor = image_tensor + residual * attack_factor

        return self._postprocess_image(final_noised_tensor)

    def decode(self, image_source):

        image_tensor, _ = self._preprocess_image(image_source)

        with torch.no_grad():
            decoder = self.net.encoder_decoder.module.decoder

            decoded_logits = decoder(image_tensor)

            if "Muti_HiDDeN" in self.model_name:
                fused_result, res_result = decoded_logits
                spa_decoded_messages, dwt_decoded_messages, dct_decoded_messages, svd_decoded_messages = res_result
                decoded_messages = fused_result

            decoded_probs = torch.sigmoid(decoded_messages)
            decoded_bits = (decoded_probs > 0.5).int().cpu().numpy().flatten().tolist()

        return decoded_bits


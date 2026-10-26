import torch
import timm

# 1. 创建模型（这步会自动下载 safetensors 并加载到内存中）
print("Downloading and loading model...")
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# 2. 将加载好的权重“另存为”标准的 .pth 文件
save_path = "vit_base_patch16_224.pth"
print(f"Saving to {save_path}...")

# 只保存 state_dict (推荐做法，体积小，兼容性好)
torch.save(model.state_dict(), save_path)

print("Done! Please upload 'vit_base_patch16_224.pth' to your server.")
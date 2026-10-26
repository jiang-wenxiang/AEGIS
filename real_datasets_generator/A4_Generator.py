import cv2
import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageFont
import os
import torch
import random
from torchvision import transforms
from network.ModelTrainer import Network


class A4ModelGenerator:
    def __init__(self, source_images_dir, save_dir="print_materials"):
        # ... (初始化代码保持不变) ...
        self.A4_W, self.A4_H = 2480, 3508
        self.save_dir = save_dir
        self.source_dir = source_images_dir
        os.makedirs(save_dir, exist_ok=True)

        self.rows, self.cols = 10, 5
        self.total_imgs_per_sheet = self.rows * self.cols
        self.margin_x, self.margin_y = 200, 250

        self.cell_w = (self.A4_W - 2 * self.margin_x) // self.cols
        self.cell_h = (self.A4_H - 2 * self.margin_y) // self.rows

        # 0.8倍率
        safe_size = int(min(self.cell_w, self.cell_h) * 0.8)
        self.print_img_size = safe_size
        self.img_w = safe_size
        self.img_h = safe_size

        self.gap_x = (self.cell_w - self.print_img_size) // 2
        self.gap_y = (self.cell_h - self.print_img_size) // 2

        self.model_input_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.message_length = 64

        self.to_tensor = transforms.Compose([
            transforms.Resize((self.model_input_size, self.model_input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def load_source_images(self):
        exts = ('.jpg', '.png', '.jpeg')
        files = [os.path.join(self.source_dir, f) for f in os.listdir(self.source_dir) if f.lower().endswith(exts)]
        return files

    def generate_random_message(self):
        bits = [random.randint(0, 1) for _ in range(self.message_length)]
        msg_str = "".join(map(str, bits))
        return torch.tensor(bits, dtype=torch.float32).unsqueeze(0).to(self.device), msg_str

    # === [新增辅助函数] 绘制文件名标签 ===
    def draw_filename_label(self, img, text):
        draw = ImageDraw.Draw(img)

        # 尝试加载一个大字体，否则字太小看不清
        try:
            # Linux 常见字体 / Windows Arial
            font_path = "arial.ttf"
            if not os.path.exists(font_path):
                # 尝试一些常见的系统字体路径 (针对 Linux 服务器)
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

            # 字号设为 80，保证肉眼清晰可见
            font = ImageFont.truetype(font_path, 200)
        except:
            # 如果加载失败，使用默认字体（比较小，但总比没有好）
            font = ImageFont.load_default()
            print("[Warning] 无法加载大字体，标签可能会比较小。")

        # 1. 在左上角绘制 (Header)
        draw.text((100, 100), text, fill="black", font_size=100)

        # 2. 在右下角绘制 (Footer) - 方便不同握持姿势确认
        # 计算文字宽度以便右对齐 (简单估算或使用 getbbox)
        try:
            bbox = draw.textbbox((0, 0), text, font_size=100)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except:
            text_w, text_h = 500, 100  # 估算

        x = self.A4_W - text_w - 100
        y = self.A4_H - text_h - 100
        draw.text((x, y), text, fill="black", font_size=100)

    def run(self, models_config, limit=100):
        all_source_files = self.load_source_images()
        if len(all_source_files) < limit:
            files_to_process = (all_source_files * (limit // len(all_source_files) + 1))[:limit]
        else:
            files_to_process = all_source_files[:limit]

        total_pages = (len(files_to_process) + self.total_imgs_per_sheet - 1) // self.total_imgs_per_sheet

        for cfg in models_config:
            model_name = cfg['name']
            print(f"\n正在处理模型: {model_name} ...")

            # 3.1 初始化模型
            net = Network(self.model_input_size, self.model_input_size, self.message_length, noise_layers=[],
                          device=self.device, batch_size=1, lr=0,
                          model_name=cfg['model_name'])

            # 判断是否是无需训练的传统方法
            is_training_free = cfg.get("training_free", False)

            # 加载权重
            if is_training_free:
                print(f"  -> [Info] {cfg['name']} 是无需训练的方法，跳过权重加载。")
            else:
                if os.path.exists(cfg['checkpoint']):
                    print(f"Loading weights from: {cfg['checkpoint']}")
                    net.load_model_ed(cfg['checkpoint'])
                else:
                    print(f"[Error] Checkpoint not found: {cfg['checkpoint']}")
                    continue

            net.encoder_decoder.eval()

            if isinstance(net.encoder_decoder, torch.nn.DataParallel):
                core_model = net.encoder_decoder.module
            else:
                core_model = net.encoder_decoder

            for page_idx in range(total_pages):
                canvas_img = Image.new('RGB', (self.A4_W, self.A4_H), (255, 255, 255))
                canvas_qr = Image.new('RGB', (self.A4_W, self.A4_H), (255, 255, 255))

                # 定义本页的文件名
                base_filename = f"{model_name}_sheet_p{page_idx}"

                # === [调用] 在生成内容前，先把大标题写上去 ===
                self.draw_filename_label(canvas_img, f"File: {base_filename}")
                self.draw_filename_label(canvas_qr, f"QR Map: {base_filename}")

                start_idx = page_idx * self.total_imgs_per_sheet
                end_idx = min((page_idx + 1) * self.total_imgs_per_sheet, len(files_to_process))

                print(f"  -> 生成第 {page_idx + 1}/{total_pages} 页...")

                for i in range(start_idx, end_idx):
                    local_idx = i - start_idx
                    r, c = divmod(local_idx, self.cols)

                    x_pos = self.margin_x + c * self.cell_w + self.gap_x
                    y_pos = self.margin_y + r * self.cell_h + self.gap_y

                    # 1. 生成水印图
                    src_path = files_to_process[i]
                    img_pil = Image.open(src_path).convert("RGB")
                    img_tensor = self.to_tensor(img_pil).unsqueeze(0).to(self.device)

                    msg_tensor, msg_str = self.generate_random_message()

                    with torch.no_grad():
                        if hasattr(core_model, 'encode'):
                            encoded_tensor = core_model.encode(img_tensor, msg_tensor)
                        else:
                            encoded_tensor = core_model.encoder(img_tensor, msg_tensor)

                    encoded_tensor = (encoded_tensor.squeeze(0).permute(1, 2, 0) + 1) / 2.0 * 255
                    encoded_np = encoded_tensor.clamp(0, 255).byte().cpu().numpy()
                    encoded_pil = Image.fromarray(encoded_np)
                    encoded_pil = encoded_pil.resize((self.print_img_size, self.print_img_size), Image.BICUBIC)

                    # 2. 生成二维码图
                    qr = qrcode.QRCode(box_size=5, border=0)
                    qr.add_data(msg_str)
                    qr.make(fit=True)
                    qr_pil = qr.make_image(fill_color="black", back_color="white").convert('RGB')
                    qr_pil = qr_pil.resize((self.print_img_size, self.print_img_size), Image.NEAREST)

                    # 粘贴
                    canvas_img.paste(encoded_pil, (x_pos, y_pos))
                    canvas_qr.paste(qr_pil, (x_pos, y_pos))

                    # 辅助标记 (小字)
                    draw = ImageDraw.Draw(canvas_qr)
                    draw.text((x_pos, y_pos - 40), f"ID:{i}", fill="black", font_size=30)

                # 保存
                canvas_img.save(os.path.join(self.save_dir, f"{base_filename}.png"))
                canvas_qr.save(os.path.join(self.save_dir, f"{base_filename}_msg.png"))
                print(f"     已保存: {base_filename}")


if __name__ == "__main__":
    # 配置列表 (和 benchmark 一样)
    my_models = [
        {
            "name": "MaskWM",
            "model_name": "MaskWM",
            "checkpoint": "../results/MaskWM_256x256_m64/coco_s1/models/EC_100.pth"
        },
        {
            "name": "SCIA-GAN",
            "model_name": "SCIA_GAN",
            "checkpoint": "../results/SCIA_GAN_256x256_m64/coco_s1/models/EC_100.pth"
        },
        {
            "name": "Universal-DCT",
            "model_name": "Universal_DCT",
            "checkpoint": "../results/Universal_DCT_256x256_m64/coco_s1/models/EC_100.pth"
        },
        {
            "name": "AEGIS (Ours)",
            "model_name": "SP_Res_Muti_HiDDeN",
            "checkpoint": "../results/SP_Res_Muti_HiDDeN_256x256_m64/models_pro_noise_SS_cocoS1_11/models/EC_500.pth"
        },
        {
            "name": "HiDDeN",
            "model_name": "HiDDeN",
            "checkpoint": "../results/HiDDeN_256x256_m64/coco_s1/models/EC_200.pth"
        },
        {
            "name": "MBRS",
            "model_name": "MBRS",
            "checkpoint": "../results/MBRS_256x256_m64/coco_s1/models/EC_100.pth"
        },
        {
            "name": "StegaStamp",
            "model_name": "StegaStamp",
            "checkpoint": "../results/StegaStamp_256x256_m64/coco_s1/models/EC_200.pth"
        },
        {
            "name": "EditGuard",
            "model_name": "EditGuard",
            "checkpoint": "../results/EditGuard_256x256_m64/coco_s1/models/EC_100.pth"
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
            "checkpoint": "../results/EDBFormer_256x256_m64/coco_s1/models/EC_300.pth"
        },
        # 添加更多...
    ]

    # 确保 datasets/test 里面有图
    generator = A4ModelGenerator(source_images_dir="../datasets/test")
    # 这里指定生成 100 张图
    # 程序会自动计算出需要 2 页 (100 / 50 = 2)
    generator.run(my_models, limit=500)

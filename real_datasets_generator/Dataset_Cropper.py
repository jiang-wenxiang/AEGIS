import cv2
import numpy as np
import os
import json
import os
import json
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

# 注册 HEIF 开启器，让 PIL 可以直接 open() HEIC 格式文件
register_heif_opener()

def read_image_cv2(img_path):
    """兼容读取普通图像和HEIC图像，并返回OpenCV格式(BGR)"""
    ext = os.path.splitext(img_path)[1].lower()
    if ext in ['.heic', '.heif']:
        try:
            # 用 PIL 读取 HEIC，转换为 RGB
            pil_img = Image.open(img_path).convert('RGB')
            # 转换为 numpy 数组，并从 RGB 转换为 OpenCV 的 BGR
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[错误] 解析 HEIC 失败 {img_path}: {e}")
            return None
    else:
        # 普通图像直接用 cv2 读取
        return cv2.imread(img_path)


class DatasetCropperPhysical:

    def __init__(self, raw_dir="raw_captures", output_dir="dataset_final"):
        self.raw_dir = raw_dir
        self.output_dir = output_dir

        # A4 布局参数 (保持一致)
        self.rows, self.cols = 10, 5
        self.A4_W, self.A4_H = 2480, 3508
        self.margin_x, self.margin_y = 200, 250
        self.cell_w = (self.A4_W - 2 * self.margin_x) // self.cols
        self.cell_h = (self.A4_H - 2 * self.margin_y) // self.rows

        safe_size = int(min(self.cell_w, self.cell_h) * 0.8)
        self.img_w = safe_size
        self.img_h = safe_size
        self.gap_x = (self.cell_w - self.img_w) // 2
        self.gap_y = (self.cell_h - self.img_h) // 2

        self.pts = []
        self.qr_detector = cv2.QRCodeDetector()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pts) < 4:
                self.pts.append((x, y))
                cv2.circle(param, (x, y), 20, (0, 0, 255), -1)
                cv2.putText(param, str(len(self.pts)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.imshow("Select Corners", param)

    def get_grid_coordinates(self, H_matrix):
        grid_polygons = []
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = self.margin_x + c * self.cell_w + self.gap_x
                y1 = self.margin_y + r * self.cell_h + self.gap_y
                x2 = x1 + self.img_w
                y2 = y1 + self.img_h
                src_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='float32').reshape(-1, 1, 2)
                dst_pts = cv2.perspectiveTransform(src_pts, H_matrix)
                grid_polygons.append(dst_pts)
        return grid_polygons

    def decode_qr(self, img):
        """增强型二维码解码"""
        # 1. 直接解
        msg, _, _ = self.qr_detector.detectAndDecode(img)
        if msg: return msg
        # 2. 二值化后解
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        msg, _, _ = self.qr_detector.detectAndDecode(binary)
        return msg if msg else None

    def process_folder(self):
        # 排除 _msg 图，只遍历内容主图
        # files = [f for f in os.listdir(self.raw_dir)
        #          if not f.endswith("_msg.jpg") and not f.endswith("_msg.png")
        #          and f.lower().endswith(('.jpg', '.png'))]
        # files.sort()
        valid_exts = ('.jpg', '.jpeg', '.png', '.heic', '.heif')

        # 排除 _msg 图，只遍历内容主图
        files = [f for f in os.listdir(self.raw_dir)
                 if not f.lower().endswith(('_msg.jpg', '_msg.jpeg', '_msg.png', '_msg.heic', '_msg.heif'))
                 and f.lower().endswith(valid_exts)]
        files.sort()

        print(f"找到 {len(files)} 组拍摄数据。")

        for f in files:
            base_name, ext = os.path.splitext(f)

            # === 1. 解析模型名称 ===
            # 文件名格式: "MBRS_ResNet_sheet_p0" -> 提取 "MBRS_ResNet"
            if "_sheet_" in base_name:
                model_name = base_name.split("_sheet_")[0]
            else:
                model_name = "Default_Model"

            # === 2. 建立模型目录 ===
            model_root = os.path.join(self.output_dir, model_name)
            model_img_dir = os.path.join(model_root, "images")
            os.makedirs(model_img_dir, exist_ok=True)

            # 读取/创建该模型的 json
            model_json_path = os.path.join(model_root, "dataset.json")
            current_model_info = {}
            if os.path.exists(model_json_path):
                with open(model_json_path, 'r') as jf:
                    current_model_info = json.load(jf)

            # === 3. 寻找对应的二维码图 ===
            # === 3. 寻找对应的二维码图 ===
            qr_filename = f"{base_name}_msg{ext}"
            qr_path = os.path.join(self.raw_dir, qr_filename)

            # 容错：允许主图和二维码图后缀不一致，遍历常见的图片后缀寻找
            if not os.path.exists(qr_path):
                valid_exts = ['.jpg', '.jpeg', '.png', '.heic', '.heif']
                found_qr = False
                for alt_ext in valid_exts:
                    test_path = os.path.join(self.raw_dir, f"{base_name}_msg{alt_ext}")
                    if os.path.exists(test_path):
                        qr_path = test_path
                        found_qr = True
                        break

                if not found_qr:
                    print(f"[错误] 找不到对应的二维码图: {base_name}_msg.*，跳过。")
                    continue

            print(f"\n>> 处理: {base_name} (模型: {model_name})")

            img_real = read_image_cv2(os.path.join(self.raw_dir, f))
            img_qr = read_image_cv2(qr_path)

            # === 4. 交互选点（两次选点） ===
            display_h = 800
            win_name = "Select Corners"

            # --- 4.1 选取内容主图 ---
            scale_real = display_h / img_real.shape[0]
            display_w_real = int(img_real.shape[1] * scale_real)
            img_display_real = cv2.resize(img_real, (display_w_real, display_h))

            self.pts = []  # 清空选点记录
            cv2.namedWindow(win_name)
            cv2.setMouseCallback(win_name, self.mouse_callback, img_display_real)
            cv2.imshow(win_name, img_display_real)
            print("   (1/2) 请在【内容图像】上按顺时针点击四个角... (ESC跳过)")
            key1 = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key1 == 27 or key1 == ord('q') or len(self.pts) != 4:
                print("   跳过内容图像选点。")
                continue
            real_pts = np.array(self.pts, dtype='float32') / scale_real

            # --- 4.2 选取二维码图 ---
            scale_qr = display_h / img_qr.shape[0]
            display_w_qr = int(img_qr.shape[1] * scale_qr)
            img_display_qr = cv2.resize(img_qr, (display_w_qr, display_h))

            self.pts = []  # 再次清空，用于接收二维码图的新选点
            cv2.namedWindow(win_name)
            cv2.setMouseCallback(win_name, self.mouse_callback, img_display_qr)
            cv2.imshow(win_name, img_display_qr)
            print("   (2/2) 请在【二维码图像】上按顺时针点击四个角... (ESC跳过)")
            key2 = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key2 == 27 or key2 == ord('q') or len(self.pts) != 4:
                print("   跳过二维码图像选点。")
                continue
            qr_pts = np.array(self.pts, dtype='float32') / scale_qr

            # === 5. 双重裁切 (独立透视) ===
            # 标准 A4 矩形坐标
            src_a4 = np.array([[0, 0], [self.A4_W, 0], [self.A4_W, self.A4_H], [0, self.A4_H]], dtype='float32')

            # 分别计算两个图像的透视变换矩阵
            H_matrix_real = cv2.getPerspectiveTransform(src_a4, real_pts)
            H_matrix_qr = cv2.getPerspectiveTransform(src_a4, qr_pts)

            # 分别计算两个图像的内部网格坐标
            polygons_real = self.get_grid_coordinates(H_matrix_real)
            polygons_qr = self.get_grid_coordinates(H_matrix_qr)

            count = 0
            # 遍历网格，注意现在要同时获取 real 和 qr 的对应多边形
            for idx in range(len(polygons_real)):
                poly_real = polygons_real[idx].reshape(4, 2)
                poly_qr = polygons_qr[idx].reshape(4, 2)

                # 计算内容主图的 Bounding Box
                min_x = max(0, int(np.min(poly_real[:, 0])))
                max_x = min(img_real.shape[1], int(np.max(poly_real[:, 0])))
                min_y = max(0, int(np.min(poly_real[:, 1])))
                max_y = min(img_real.shape[0], int(np.max(poly_real[:, 1])))
                if max_x <= min_x or max_y <= min_y: continue

                # A. 裁切内容图 (使用 poly_real)
                crop_real = img_real[min_y:max_y, min_x:max_x].copy()

                # B. 裁切并矫正二维码 (使用 poly_qr)
                qr_size = 300
                dst_qr_pts = np.array([[0, 0], [qr_size, 0], [qr_size, qr_size], [0, qr_size]], dtype='float32')
                # 注意：这里传入的是 poly_qr，完美解决位姿不对应的问题
                M_local = cv2.getPerspectiveTransform(poly_qr, dst_qr_pts)
                crop_qr_flat = cv2.warpPerspective(img_qr, M_local, (qr_size, qr_size))

                # C. 解码
                message = self.decode_qr(crop_qr_flat)

                if message:
                    save_name = f"{base_name}_patch_{idx}.png"
                    cv2.imwrite(os.path.join(model_img_dir, save_name), crop_real)

                    current_model_info[save_name] = {
                        "message": message,
                        "origin": base_name,
                        "grid_index": idx
                    }
                    count += 1
                else:
                    # 解码失败则不保存该样本
                    pass

            print(f"   -> 成功提取 {count} 张样本")

            # === 6. 更新 JSON ===
            with open(model_json_path, "w") as jf:
                json.dump(current_model_info, jf, indent=4)

        print("\n全部完成！")


if __name__ == "__main__":
    # 请确保 raw_captures 里有成对的 .jpg/.png
    cropper = DatasetCropperPhysical(raw_dir="raw_captures")
    cropper.process_folder()
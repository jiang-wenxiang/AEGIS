import cv2
import numpy as np
import os
import json
from pyzbar.pyzbar import decode as pyzbar_decode
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

        # A4 布局参数 (严格保持一致)
        self.rows, self.cols = 10, 5
        self.A4_W, self.A4_H = 2480, 3508
        self.margin_x, self.margin_y = 600, 750
        self.cell_w = (self.A4_W - 2 * self.margin_x) // self.cols
        self.cell_h = (self.A4_H - 2 * self.margin_y) // self.rows

        safe_size = int(min(self.cell_w, self.cell_h) * 0.8)
        self.img_w = safe_size
        self.img_h = safe_size
        self.gap_x = (self.cell_w - self.img_w) // 2
        self.gap_y = (self.cell_h - self.img_h) // 2

        self.pts = []
        self.current_win_name = ""
        self.qr_detector = cv2.QRCodeDetector()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pts) < 4:
                self.pts.append((x, y))
                cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
                cv2.drawMarker(param, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

                labels = ["1 (TL)", "2 (TR)", "3 (BR)", "4 (BL)"]
                cv2.putText(param, labels[len(self.pts) - 1], (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if len(self.pts) == 4:
                    pts_arr = np.array(self.pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(param, [pts_arr], True, (255, 0, 0), 2)

                cv2.imshow(self.current_win_name, param)

    def get_user_clicks(self, img_display, win_name, prompt_text):
        """通用交互函数：弹窗并获取用户的 4 次点击"""
        self.pts = []
        self.current_win_name = win_name
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self.mouse_callback, img_display)
        cv2.imshow(win_name, img_display)

        print(f"   >>> {prompt_text}")
        print("       (顺序: 1.左上 2.右上 3.右下 4.左下 | 按 ESC 跳过)")

        while len(self.pts) < 4:
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord('q'):
                break

        cv2.destroyWindow(win_name)
        return list(self.pts)

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

    # def decode_qr(self, img):
    #     msg, _, _ = self.qr_detector.detectAndDecode(img)
    #     if msg: return msg
    #
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #     msg, _, _ = self.qr_detector.detectAndDecode(binary)
    #     return msg if msg else None

    def decode_qr(self, img):
        """使用高精度 PyZBar 的增强型二维码解码"""
        # 1. 尝试直接在原图上解码
        decoded_objects = pyzbar_decode(img)
        if decoded_objects:
            return decoded_objects[0].data.decode('utf-8')

        # 2. 如果失败，转为灰度图再次尝试
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        decoded_objects = pyzbar_decode(gray)
        if decoded_objects:
            return decoded_objects[0].data.decode('utf-8')

        # 3. 如果还是失败，使用大津法(Otsu)进行自适应二值化后最后搏一次
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        decoded_objects = pyzbar_decode(binary)
        if decoded_objects:
            return decoded_objects[0].data.decode('utf-8')

        # 全部失败
        return None

    def process_folder(self):
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

        # 计算理想生成图中的四个基准内角点 (物理坐标锚点)
        tl_x = self.margin_x + self.gap_x
        tl_y = self.margin_y + self.gap_y
        tr_x = self.margin_x + (self.cols - 1) * self.cell_w + self.gap_x + self.img_w
        tr_y = self.margin_y + self.gap_y
        br_x = self.margin_x + (self.cols - 1) * self.cell_w + self.gap_x + self.img_w
        br_y = self.margin_y + (self.rows - 1) * self.cell_h + self.gap_y + self.img_h
        bl_x = self.margin_x + self.gap_x
        bl_y = self.margin_y + (self.rows - 1) * self.cell_h + self.gap_y + self.img_h

        src_img_corners = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype='float32')

        for f in files:
            base_name, ext = os.path.splitext(f)

            if "_sheet_" in base_name:
                model_name = base_name.split("_sheet_")[0]
            else:
                model_name = "Default_Model"

            model_root = os.path.join(self.output_dir, model_name)
            model_img_dir = os.path.join(model_root, "images")
            os.makedirs(model_img_dir, exist_ok=True)

            model_json_path = os.path.join(model_root, "dataset.json")
            current_model_info = {}
            if os.path.exists(model_json_path):
                with open(model_json_path, 'r') as jf:
                    current_model_info = json.load(jf)

            qr_filename = f"{base_name}_msg{ext}"
            qr_path = os.path.join(self.raw_dir, qr_filename)

            # if not os.path.exists(qr_path):
            #     alt_ext = '.png' if ext.lower() == '.jpg' else '.jpg'
            #     if os.path.exists(os.path.join(self.raw_dir, f"{base_name}_msg{alt_ext}")):
            #         qr_path = os.path.join(self.raw_dir, f"{base_name}_msg{alt_ext}")
            #     else:
            #         print(f"[错误] 找不到对应的二维码图: {qr_filename}，跳过。")
            #         continue

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

            print(f"\n>> 正在处理配对: {base_name} (模型: {model_name})")

            # 读取图像
            img_real = read_image_cv2(os.path.join(self.raw_dir, f))
            img_qr = read_image_cv2(qr_path)

            display_h = 900

            # --- 步骤 1：处理主图像 (Content Image) ---
            scale_real = display_h / img_real.shape[0]
            display_w_real = int(img_real.shape[1] * scale_real)
            img_real_disp = cv2.resize(img_real, (display_w_real, display_h))

            pts_real = self.get_user_clicks(img_real_disp, "Step 1: CONTENT Image",
                                            "请在【主图像】上点击最外围的 4 个图片角点")
            if len(pts_real) != 4:
                print("   已跳过当前配对。")
                continue

            # --- 步骤 2：处理二维码图像 (QR Image) ---
            scale_qr = display_h / img_qr.shape[0]
            display_w_qr = int(img_qr.shape[1] * scale_qr)
            img_qr_disp = cv2.resize(img_qr, (display_w_qr, display_h))

            pts_qr = self.get_user_clicks(img_qr_disp, "Step 2: QR CODE Image",
                                          "请在【二维码图像】上点击最外围的 4 个二维码角点")
            if len(pts_qr) != 4:
                print("   已跳过当前配对。")
                continue

            # ================= 计算两套独立的透视矩阵 =================
            real_pts_real = np.array(pts_real, dtype='float32') / scale_real
            H_matrix_real = cv2.getPerspectiveTransform(src_img_corners, real_pts_real)
            polygons_real = self.get_grid_coordinates(H_matrix_real)

            real_pts_qr = np.array(pts_qr, dtype='float32') / scale_qr
            H_matrix_qr = cv2.getPerspectiveTransform(src_img_corners, real_pts_qr)
            polygons_qr = self.get_grid_coordinates(H_matrix_qr)

            # ================= 分离裁切，同步配对 =================
            count = 0
            # 使用 zip 同时遍历两个多边形列表
            for idx, (poly_real, poly_qr) in enumerate(zip(polygons_real, polygons_qr)):

                # ================= 分离裁切，同步配对 =================
                poly_real = poly_real.reshape(4, 2)
                poly_qr = poly_qr.reshape(4, 2)

                # A. 裁切内容图 (向外扩展像素)
                expand_px = 20  # [参数] 内容图向外扩展的像素值，可根据需要调大
                min_x = max(0, int(np.min(poly_real[:, 0])) - expand_px)
                max_x = min(img_real.shape[1], int(np.max(poly_real[:, 0])) + expand_px)
                min_y = max(0, int(np.min(poly_real[:, 1])) - expand_px)
                max_y = min(img_real.shape[0], int(np.max(poly_real[:, 1])) + expand_px)
                if max_x <= min_x or max_y <= min_y: continue
                crop_real = img_real[min_y:max_y, min_x:max_x].copy()

                # B. 裁切并矫正二维码 (基于多边形中心按比例向外放大)
                expand_ratio = 1.2  # [参数] 放缩比例。1.20 表示从中心向外扩展 20% 的边距

                # 计算这块二维码多边形的几何中心
                center_qr = np.mean(poly_qr, axis=0)
                # 将 4 个角点基于中心向外延展
                poly_qr_expanded = center_qr + (poly_qr - center_qr) * expand_ratio
                poly_qr_expanded = poly_qr_expanded.astype('float32')

                qr_size = 300
                dst_qr_pts = np.array([[0, 0], [qr_size, 0], [qr_size, qr_size], [0, qr_size]], dtype='float32')

                # 使用外扩后的角点计算透视矩阵
                M_local = cv2.getPerspectiveTransform(poly_qr_expanded, dst_qr_pts)

                # 执行裁切。增加 borderValue=(255, 255, 255) 的原因是：
                # 如果外扩后超出了原图边界，OpenCV 默认填黑色，这会破坏二维码。填纯白色则完美契合。
                crop_qr_flat = cv2.warpPerspective(img_qr, M_local, (qr_size, qr_size), borderValue=(255, 255, 255))

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
                    # 解码失败时的可视化调试
                    print(f"   [警告] 网格索引 {idx} 解码失败！展示裁剪结果...")
                    cv2.imshow("Debug - Failed QR (Any key to next, ESC to skip)", crop_qr_flat)
                    cv2.imshow("Debug - Failed Content", crop_real)

                    debug_key = cv2.waitKey(0) & 0xFF
                    cv2.destroyWindow("Debug - Failed QR (Any key to next, ESC to skip)")
                    cv2.destroyWindow("Debug - Failed Content")

                    if debug_key == 27:
                        print("   手动终止当前图片的排查。")
                        break

            print(f"   -> 成功提取 {count} 张样本")

            with open(model_json_path, "w") as jf:
                json.dump(current_model_info, jf, indent=4)

        print("\n全部处理完成！")


if __name__ == "__main__":
    cropper = DatasetCropperPhysical(raw_dir="raw_captures")
    cropper.process_folder()
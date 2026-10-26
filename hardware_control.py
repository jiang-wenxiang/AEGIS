import time
import socket
import json
import math
import os
import ctypes
import threading
from ctypes import wintypes
import cv2
import numpy as np
import torch


class ScreenCameraHardware:
    HWND_TOPMOST = wintypes.HWND(-1)
    SW_RESTORE = 9
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_NOACTIVATE = 0x0010
    SWP_SHOWWINDOW = 0x0040
    WS_EX_TOPMOST = 0x00000008
    GWL_EXSTYLE = -20

    def __init__(self, camera_index=0, waypoints_file="watermark_waypoints.json", gateway_ip="127.0.0.1",
                 gateway_port=8085):
        print("[Hardware] 正在初始化机械臂与摄像机...")
        self.gateway_ip = gateway_ip
        self.gateway_port = gateway_port

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.window_name = "Screen_Target"
        self._topmost_stop_event = threading.Event()
        self._topmost_thread = None
        self._user32 = None
        self._window_hwnd = None
        self._initialize_topmost_backend()
        self._create_display_window()

        self.waypoints = self._load_and_sort_waypoints(waypoints_file)
        self.num_poses = len(self.waypoints)

        self.current_pose_idx = -1
        self.current_roi = None
        self.current_win_pos = None

        self.base_save_dir = "results_screencam"
        os.makedirs(self.base_save_dir, exist_ok=True)

        self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_client.connect((self.gateway_ip, self.gateway_port))
        self._start_topmost_guard()
        print(f"[Hardware] ✅ {self.num_poses}")

    def _initialize_topmost_backend(self):
        if os.name != "nt":
            return
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        self._user32.FindWindowW.restype = wintypes.HWND
        self._user32.IsWindow.argtypes = [wintypes.HWND]
        self._user32.IsWindow.restype = wintypes.BOOL
        self._user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self._user32.IsWindowVisible.restype = wintypes.BOOL
        self._user32.IsIconic.argtypes = [wintypes.HWND]
        self._user32.IsIconic.restype = wintypes.BOOL
        self._user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self._user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        self._user32.SetWindowPos.restype = wintypes.BOOL
        self._user32.BringWindowToTop.argtypes = [wintypes.HWND]
        self._user32.BringWindowToTop.restype = wintypes.BOOL
        self._user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        self._user32.GetWindowLongW.restype = ctypes.c_long

    def _create_display_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
        if hasattr(cv2, "WND_PROP_TOPMOST"):
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)

        cv2.waitKey(1)
        self._force_native_topmost()

    def _window_is_open(self):
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def _force_native_topmost(self):
        if self._user32 is None:
            return True

        hwnd = self._window_hwnd
        if not hwnd or not self._user32.IsWindow(hwnd):
            hwnd = self._user32.FindWindowW(None, self.window_name)
            self._window_hwnd = hwnd
        if not hwnd:
            return False

        if self._user32.IsIconic(hwnd):
            self._user32.ShowWindow(hwnd, self.SW_RESTORE)

        flags = (
            self.SWP_NOMOVE
            | self.SWP_NOSIZE
            | self.SWP_NOACTIVATE
            | self.SWP_SHOWWINDOW
        )
        self._user32.SetWindowPos(
            hwnd,
            self.HWND_TOPMOST,
            0,
            0,
            0,
            0,
            flags,
        )
        self._user32.BringWindowToTop(hwnd)

        is_visible = bool(self._user32.IsWindowVisible(hwnd))
        ex_style = self._user32.GetWindowLongW(hwnd, self.GWL_EXSTYLE)
        is_topmost = bool(ex_style & self.WS_EX_TOPMOST)
        return is_visible and is_topmost

    def _ensure_display_window_topmost(self):
        if not self._window_is_open():
            self._window_hwnd = None
            self._create_display_window()

        if hasattr(cv2, "WND_PROP_TOPMOST"):
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)

        if not self._force_native_topmost():
            cv2.waitKey(1)
            self._force_native_topmost()

    def _topmost_watch_loop(self):
        while not self._topmost_stop_event.wait(0.05):
            try:
                self._force_native_topmost()
            except Exception:
                pass

    def _start_topmost_guard(self):
        if self._user32 is None or (
            self._topmost_thread is not None
            and self._topmost_thread.is_alive()
        ):
            return

        self._topmost_stop_event.clear()
        self._topmost_thread = threading.Thread(
            target=self._topmost_watch_loop,
            name="screen-target-topmost-guard",
            daemon=True,
        )
        self._topmost_thread.start()

    def _stop_topmost_guard(self):
        self._topmost_stop_event.set()
        if self._topmost_thread is not None and self._topmost_thread.is_alive():
            self._topmost_thread.join(timeout=1.0)

    def _load_and_sort_waypoints(self, filepath):
        with open(filepath, 'r') as f:
            raw_waypoints = json.load(f)

        if not raw_waypoints:
            raise ValueError(f"Poses file {filepath} NULL！")

        sorted_wp = [raw_waypoints.pop(0)]
        while raw_waypoints:
            last_pose = sorted_wp[-1]["pose"]
            best_idx = 0
            min_dist = float('inf')
            for i, wp in enumerate(raw_waypoints):
                curr_pose = wp["pose"]
                dist = sum((curr_pose[k] - last_pose[k]) ** 2 for k in range(5))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            sorted_wp.append(raw_waypoints.pop(best_idx))
        return sorted_wp

    def _generate_checkerboard(self, size=256, squares=6):
        sq_size = size // squares
        board = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(squares):
            for j in range(squares):
                if (i + j) % 2 == 0:
                    board[i * sq_size:(i + 1) * sq_size, j * sq_size:(j + 1) * sq_size] = (255, 255, 255)
        return board

    def _clear_camera_buffer(self):
        for _ in range(5):
            self.cap.grab()
        ret, frame = self.cap.read()
        return ret, frame

    def _show_on_fullscreen(self, target_img_256, win_pos):
        self._ensure_display_window_topmost()

        screen_h, screen_w = 1080, 1920

        canvas = np.full((screen_h, screen_w, 3), 128, dtype=np.uint8)

        wx, wy = win_pos

        target_h, target_w = target_img_256.shape[:2]
        end_y = min(wy + target_h, screen_h)
        end_x = min(wx + target_w, screen_w)

        canvas[wy:end_y, wx:end_x] = target_img_256[:(end_y - wy), :(end_x - wx)]

        cv2.imshow(self.window_name, canvas)
        self._ensure_display_window_topmost()
        cv2.waitKey(200)
        self._force_native_topmost()

    def move_to_pose_and_verify(self, pose_idx):
        wp = self.waypoints[pose_idx]
        pose = wp["pose"]
        self.current_win_pos = wp["win_pos"]
        self.current_roi = wp["roi"]

        print(f"\n[Hardware] 🦾 goto... {pose_idx + 1}/{self.num_poses} ...")
        cmd = f"#1P{pose[0]}#2P{pose[1]}#3P{pose[2]}#4P{pose[3]}#5P{pose[4]}T1500\r\n"
        self.tcp_client.sendall(cmd.encode('utf-8'))

        checkerboard = self._generate_checkerboard()
        self._show_on_fullscreen(checkerboard, self.current_win_pos)

        print("[Hardware] ⏳ 机械臂已就位，防抖震定与屏幕刷新中 (等待 5 秒)...")
        time.sleep(5.0)

        self._ensure_display_window_topmost()
        ret, frame = self._clear_camera_buffer()
        if not ret:
            raise Exception("相机掉线！")

        rx, ry, rw, rh = self.current_roi
        img_h, img_w = frame.shape[:2]

        def clamp(x, y, w, h):
            return max(0, min(x, img_w - w)), max(0, min(y, img_h - h))

        rx, ry = clamp(rx, ry, rw, rh)
        roi_img = frame[ry:ry + rh, rx:rx + rw]
        found, _ = cv2.findChessboardCorners(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY), (5, 5), None)

        if not found:
            print("[Hardware] ⚠️ 原设定区域内未检测到靶区！触发全屏扫描...")
            found_full, corners_full = cv2.findChessboardCorners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), None)

            if found_full:
                new_cx = int(np.mean(corners_full[:, 0, 0]))
                new_cy = int(np.mean(corners_full[:, 0, 1]))
                rx, ry = clamp(new_cx - (rw // 2), new_cy - (rh // 2), rw, rh)
                self.current_roi = [rx, ry, rw, rh]
                print(f"[Hardware] 🔧 动态补偿成功！新 ROI 坐标: [{rx}, {ry}, {rw}, {rh}]")
            else:
                # [修改] 统一将报错现场存放到 results_screencam/errors 文件夹中
                error_dir = os.path.join(self.base_save_dir, "errors")
                os.makedirs(error_dir, exist_ok=True)
                error_path = os.path.join(error_dir, f"fatal_error_pose_{pose_idx + 1}.jpg")
                cv2.imwrite(error_path, frame)

                raise Exception(f"❌ 严重错误！镜头已彻底脱离屏幕区域，测试安全熔断。错误现场已保存至: {error_path}")
        else:
            print("[Hardware] 👁️ 视觉校验。")

    def display_and_capture_batch(self, tensor_batch, global_start_idx, model_name):

        B, C, H, W = tensor_batch.shape
        processed_tensors = []
        rx, ry, rw, rh = self.current_roi

        for i in range(B):
            global_img_idx = global_start_idx + i

            # Tensor [-1, 1] RGB 转 CV2 [0, 255] BGR
            img_tensor = tensor_batch[i].cpu().permute(1, 2, 0).numpy()
            img_cv2 = ((img_tensor + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

            self._show_on_fullscreen(img_cv2, self.current_win_pos)

            time.sleep(1.5)

            self._ensure_display_window_topmost()
            _, frame = self._clear_camera_buffer()

            cropped_roi = frame[ry:ry + rh, rx:rx + rw]
            resized_roi = cv2.resize(cropped_roi, (W, H), interpolation=cv2.INTER_CUBIC)

            # if global_img_idx % 50 == 0:
            #     model_save_dir = os.path.join(self.base_save_dir, model_name)
            #     os.makedirs(model_save_dir, exist_ok=True)
            #
            #     save_path = os.path.join(model_save_dir, f"img_idx_{global_img_idx:04d}.jpg")
            #     cv2.imwrite(save_path, resized_roi)

            # CV2 BGR 转回 Tensor [-1, 1] RGB
            rgb_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
            out_tensor = torch.from_numpy(rgb_roi).permute(2, 0, 1).float() / 255.0
            out_tensor = (out_tensor * 2.0) - 1.0
            processed_tensors.append(out_tensor)

        return torch.stack(processed_tensors, dim=0)

    def close(self):
        print("[Hardware] 释放硬件资源...")
        self._stop_topmost_guard()
        self.cap.release()
        cv2.destroyAllWindows()
        self.tcp_client.close()

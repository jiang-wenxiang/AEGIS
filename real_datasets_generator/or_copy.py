import cv2
import numpy as np
import os
import shutil
from pathlib import Path

def adjust_gamma(image, gamma=1.0):
    """
    使用伽马校正调整图像亮度。
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_dataset_directory(input_dir, output_dir, gamma_value=1.5):
    """
    深度遍历文件夹，提亮图像，复刻目录结构并拷贝非图像文件。
    """
    # 转换为绝对路径，避免相对路径拼接时的潜在错误
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    # 常见的图像扩展名（可以根据你的数据集自行增减）
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    
    # 统计计数器
    img_count = 0
    other_count = 0
    
    print(f"开始遍历并处理数据集目录: {input_dir}")
    print(f"目标输出目录: {output_dir}\n")

    # 使用 os.walk 递归遍历所有目录和子目录
    for root, dirs, files in os.walk(input_dir):
        
        # 1. 计算当前文件夹相对于输入根目录的相对路径
        # 例如：root 是 "/input/sub1/sub2"，input_dir 是 "/input"
        # 那么 rel_path 就是 "sub1/sub2"
        rel_path = os.path.relpath(root, input_dir)
        
        # 2. 在输出文件夹中构建对应的层级目录
        target_root = os.path.join(output_dir, rel_path)
        os.makedirs(target_root, exist_ok=True)
        
        # 3. 遍历当前文件夹下的所有文件
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_root, file)
            
            # 判断是否为支持的图像格式
            if file.lower().endswith(image_extensions):
                # 读取图像
                img = cv2.imread(source_file)
                
                if img is not None:
                    # 如果成功读取，进行亮度调整并保存
                    brightened_img = adjust_gamma(img, gamma=gamma_value)
                    cv2.imwrite(target_file, brightened_img)
                    img_count += 1
                else:
                    # 如果 OpenCV 读取失败（可能是文件损坏或伪装扩展名），直接拷贝
                    shutil.copy2(source_file, target_file)
                    other_count += 1
            else:
                # 非图像文件（如 .json, .txt, .csv 等），直接拷贝保留元数据
                shutil.copy2(source_file, target_file)
                other_count += 1
                
        # 打印当前子目录的处理进度
        if files:
            print(f"已处理目录 -> {rel_path} (包含 {len(files)} 个文件)")

    print("\n" + "="*40)
    print("数据集处理完成！")
    print(f"成功提亮图像: {img_count} 张")
    print(f"直接拷贝文件: {other_count} 个")
    print("="*40)

if __name__ == "__main__":
    # ================= 配置区域 =================
    # 1. 设置包含多级子目录的数据集输入路径
    INPUT_FOLDER = "./dataset_final" 
    
    # 2. 设置输出路径（程序会自动在此创建与输入端完全一致的内部结构）
    OUTPUT_FOLDER = "./dataset_final_bright" 
    
    # 3. 设置提亮程度 (Gamma 值)
    GAMMA = 2.2 
    # ============================================
    
    process_dataset_directory(INPUT_FOLDER, OUTPUT_FOLDER, gamma_value=GAMMA)
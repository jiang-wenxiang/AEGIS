import matplotlib.pyplot as plt
import numpy as np
import os


def _is_ours(name):
    """Recognize the proposed method before assigning plot styles."""
    normalized = str(name).strip().lower()
    return "(ours)" in normalized


def _ours_first(results):
    """Drop the legacy AEGIS key and put the current Ours entry first."""
    filtered_items = [
        item
        for item in results.items()
        if str(item[0]).strip().lower() != "aegis"
    ]
    return sorted(filtered_items, key=lambda item: 0 if _is_ours(item[0]) else 1)


def plot_radar_chart(results, categories, filename="radar_chart.pdf", title="Robustness Comparison"):
    """
    生成用于对比不同模型抗噪能力的雷达图 (Radar Chart)。

    参数:
        results (dict): 数据字典。
                        格式: {
                            "Model_Name_1": [bit_acc_1, bit_acc_2, ...],
                            "Model_Name_2": [bit_acc_1, bit_acc_2, ...],
                        }
                        注意：列表中的值应该是 Bit Accuracy (0.0 - 1.0)。
        categories (list): 噪声名称列表，长度必须与 bit_acc 列表一致。
                           例如: ["Rotation", "Crop", "JPEG", ...]
        filename (str): 保存路径。支持 .svg, .pdf (矢量图) 或 .png。
        title (str): 图表标题。
    """

    # 【关键】设置 SVG 字体为 'none'，这样生成的 SVG 里的文字就是真正的 Text 标签，可以直接用记事本改
    plt.rcParams['svg.fonttype'] = 'none'

    # 1. 准备角度 (将圆周分为 N 份)
    N = len(categories)
    # 生成角度，并闭合（最后一个点回到起点）
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # 2. 初始化图表
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 设置顺时针方向，且 0 度在顶部
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 3. 设置 X 轴标签 (噪声名称)
    plt.xticks(angles[:-1], categories, size=12)

    # 4. 设置径向轴。绘图区从 0.45 开始，但只标注 0.5 以上的刻度。
    # 这样可以压缩无数据的中心区域，同时给接近 0.5 的曲线留出余量。
    y_ticks_locs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y_ticks_labels = ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

    ax.set_rlabel_position(0)
    ax.set_yticks(y_ticks_locs)
    ax.set_yticklabels(y_ticks_labels, color="grey", size=10)
    ax.set_ylim(0.45, 1.02)
    ax.set_facecolor("white")

    # 5. 绘制数据
    # 预定义一些好看的颜色
    # 12种具有明显区分度的颜色
    colors = [
        '#E63946',  # 红
        '#2A9D8F',  # 青绿
        '#F4A261',  # 橙
        '#264653',  # 深蓝绿
        '#8338EC',  # 紫
        '#FFB703',  # 金黄
        '#3A86FF',  # 蓝
        '#FB5607',  # 橙红
        '#06D6A0',  # 亮绿
        '#FF006E',  # 品红
        '#8C564B',  # 棕色
        '#4D4D4D',  # 深灰
    ]

    # 12种线型，包括Matplotlib自定义虚线形式
    line_styles = [
        '-',  # 实线
        '--',  # 虚线
        '-.',  # 点划线
        ':',  # 点线
        (0, (6, 2)),  # 长虚线
        (0, (3, 1)),  # 短虚线
        (0, (1, 1)),  # 密集点线
        (0, (5, 2, 1, 2)),  # 长划-点
        (0, (3, 1, 1, 1)),  # 短划-点
        (0, (7, 2, 2, 2)),  # 长划-短划
        (0, (2, 2, 6, 2)),  # 短划-长划
        (0, (5, 1, 1, 1, 1, 1)),  # 长划-双点
    ]

    other_style_index = 0
    for name, bit_accuracies in _ours_first(results):
        is_ours = _is_ours(name)

        # 数据闭合
        values = bit_accuracies
        radii = [val for val in values]
        radii += radii[:1]  # 闭合

        if is_ours:
            color = colors[0]
            line_style = '-'
            line_width = 3.2
            zorder = 10
        else:
            # 跳过预留给 Ours 的红色和实线。
            color = colors[1 + other_style_index % (len(colors) - 1)]
            line_style = line_styles[1 + other_style_index % (len(line_styles) - 1)]
            line_width = 2.0
            zorder = 3
            other_style_index += 1

        # 绘制线条
        ax.plot(
            angles,
            radii,
            linewidth=line_width,
            linestyle=line_style,
            label=name,
            color=color,
            zorder=zorder,
        )

    # 6. 添加图例和标题
    # 将图例放在图表外侧，避免遮挡
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1.1),
        frameon=True,
        framealpha=1.0,
        borderpad=1.1,
    )
    plt.title(title, size=16, y=1.1)

    # 7. 保存为矢量图
    # 自动创建目录
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    plt.savefig(filename, format=filename.split('.')[-1], bbox_inches='tight', dpi=300)
    print(f"[Plot] Radar chart saved to {filename}")
    plt.close()


import matplotlib.pyplot as plt
import os


def plot_line_chart(results, categories, filename="line_chart.pdf", title="Robustness Comparison"):
    """
    生成用于对比不同模型抗噪能力的折线图 (Line Chart)。
    使用与雷达图相同的数据输入格式，但优化了颜色对比度和线型区分度。

    参数:
        results (dict): 数据字典。
                        格式: {
                            "Model_Name_1": [bit_acc_1, bit_acc_2, ...],
                            "Model_Name_2": [bit_acc_1, bit_acc_2, ...],
                        }
                        注意：列表中的值应该是 Bit Accuracy (0.0 - 1.0)。
        categories (list): 噪声名称列表，长度必须与 bit_acc 列表一致。
                           例如: ["Rotation", "Crop", "JPEG", ...]
        filename (str): 保存路径。支持 .svg, .pdf (矢量图) 或 .png。
        title (str): 图表标题。
    """

    # 设置 SVG 字体为 'none'，以便后续在矢量软件中编辑文字
    plt.rcParams['svg.fonttype'] = 'none'

    # 1. 初始化图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 2. 高对比度调色板、不同形状的标记点、不同的线型
    # 这确保了即使颜色相近，也能通过形状和线型区分
    # 12种具有明显区分度的颜色
    colors = [
        '#E63946',  # 红
        '#2A9D8F',  # 青绿
        '#F4A261',  # 橙
        '#264653',  # 深蓝绿
        '#8338EC',  # 紫
        '#FFB703',  # 金黄
        '#3A86FF',  # 蓝
        '#FB5607',  # 橙红
        '#06D6A0',  # 亮绿
        '#FF006E',  # 品红
        '#8C564B',  # 棕色
        '#4D4D4D',  # 深灰
    ]

    # 12种线型，包括Matplotlib自定义虚线形式
    line_styles = [
        '-',  # 实线
        '--',  # 虚线
        '-.',  # 点划线
        ':',  # 点线
        (0, (6, 2)),  # 长虚线
        (0, (3, 1)),  # 短虚线
        (0, (1, 1)),  # 密集点线
        (0, (5, 2, 1, 2)),  # 长划-点
        (0, (3, 1, 1, 1)),  # 短划-点
        (0, (7, 2, 2, 2)),  # 长划-短划
        (0, (2, 2, 6, 2)),  # 短划-长划
        (0, (5, 1, 1, 1, 1, 1)),  # 长划-双点
    ]

    markers = [
        'o', 's', '^', 'D', 'v', 'P',
        'X', '*', '<', '>', 'h', 'p'
    ]

    # 3. 绘制数据
    x_positions = range(len(categories))

    other_style_index = 0
    for name, bit_accuracies in _ours_first(results):
        is_ours = _is_ours(name)

        if len(bit_accuracies) != len(categories):
            print(
                f"Warning: Data length for {name} ({len(bit_accuracies)}) does not match categories ({len(categories)}).")
            continue

        if is_ours:
            color = colors[0]
            marker = markers[0]
            line_style = '-'
            line_width = 3.2
            marker_size = 8.5
            zorder = 10
        else:
            color = colors[1 + other_style_index % (len(colors) - 1)]
            marker = markers[1 + other_style_index % (len(markers) - 1)]
            line_style = line_styles[1 + other_style_index % (len(line_styles) - 1)]
            line_width = 2.2
            marker_size = 7
            zorder = 3
            other_style_index += 1

        ax.plot(
            x_positions,
            bit_accuracies,
            linewidth=line_width,
            markersize=marker_size,
            label=name,
            color=color,
            marker=marker,
            linestyle=line_style,
            zorder=zorder,
            # alpha=0.8 # 如果线太多觉得乱，可以稍微调低透明度
        )

    # 4. 配置 X 轴
    ax.set_xticks(x_positions)
    # 稍微倾斜标签，防止名称过长重叠
    ax.set_xticklabels(categories, rotation=0, ha="center", size=11)
    ax.set_xlabel("Noise Intensity", size=13, labelpad=10)

    # 5. 配置 Y 轴
    ax.set_ylabel("Bit Acc", size=13, labelpad=10)
    ax.set_ylim(0.45, 1.02)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))

    # 增加细致的背景网格线，方便读数
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color='lightgray', alpha=0.7)
    ax.xaxis.grid(False)  # 通常折线图只保留横向网格更清晰

    # 隐藏上方和右侧的边框线条
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 6. 添加图例和标题
    # 将图例放在外侧，分栏显示，看起来更整洁
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1.1),
        frameon=True,
        framealpha=1.0,
        borderpad=1.1,
    )
    plt.title(title, size=16, pad=20, fontweight='bold')

    # 7. 保存为矢量图
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    # bbox_inches='tight' 确保保存在外部的图例不会被裁剪掉
    plt.savefig(filename, format=filename.split('.')[-1], bbox_inches='tight', dpi=300)
    print(f"[Plot] Line chart saved to {filename}")
    plt.close()

# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟数据
    mock_data = {
        "MBRS": [0.05, 0.10, 0.02, 0.20, 0.00],  # Error Rates
        "RivaGAN": [0.15, 0.05, 0.10, 0.25, 0.05],
        "HiDDeN": [0.30, 0.40, 0.20, 0.50, 0.10]
    }
    cats = ["Rotation", "Crop", "JPEG", "Noise", "Blur"]

    plot_radar_chart(mock_data, cats, "test_radar.svg")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 检查文件是否存在
file_path = "depth_000387.png"
if not os.path.exists(file_path):
    print(f"文件 {file_path} 不存在!")
    exit(1)

try:
    # 读取深度图像
    depth_img = Image.open(file_path)

    # 转换为numpy数组
    depth_array = np.array(depth_img)

    # 显示通道数量信息
    if len(depth_array.shape) == 2:
        height, width = depth_array.shape
        channels = 1
        print(f"图像大小: {width}x{height}, 通道数: {channels} (灰度图)")
    else:
        height, width, channels = depth_array.shape
        print(f"图像大小: {width}x{height}, 通道数: {channels}")

    # 显示位深度信息
    print(f"数据类型: {depth_array.dtype}")
    print(f"最小值: {depth_array.min()}, 最大值: {depth_array.max()}")

    # 按照要求进行值的变换: 值 / 65535 * 5 * 255
    # 确保转换为浮点数进行计算
    normalized_array = depth_array.astype(np.float32) / 65535.0 * 5 * 255

    # 确保值在有效范围内
    normalized_array = np.clip(normalized_array, 0, 255)

    # 转换为uint8以便显示
    display_array = normalized_array.astype(np.uint8)

    # 创建图像显示
    plt.figure(figsize=(10, 8))

    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.title("原始深度图")
    plt.imshow(depth_array, cmap="plasma")  # plasma是适合深度图的色彩映射
    plt.colorbar(label="深度值")

    # 显示变换后的图像
    plt.subplot(1, 2, 2)
    plt.title("变换后 (值/65535*5*255)")
    plt.imshow(display_array, cmap="plasma")
    plt.colorbar(label="变换后的值")

    plt.tight_layout()
    plt.show()

    print("变换公式: 原始值 / 65535 * 5 * 255")

except Exception as e:
    print(f"处理图像时出错: {e}")

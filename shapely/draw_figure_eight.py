import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

def create_figure_eight_with_lines(size=2):
    """
    创建一个由直线组成的8字形
    
    参数:
    size: 控制8字形的大小
    
    返回:
    一个由直线组成的8字形几何体
    """
    # 定义8字形的点
    # 使用参数方程定义一个顺滑的8字形
    t = np.linspace(0, 2 * np.pi, 100)
    x = size * np.sin(t) / 2
    y = size * np.sin(t) * np.cos(t)
    
    # 创建点列表
    points = [(x[i], y[i]) for i in range(len(t))]
    
    # 使用LineString创建闭合曲线
    line = LineString(points + [points[0]])
    
    # 将线转换为多边形以获得填充区域
    polygon = Polygon(line)
    
    return polygon

def plot_figure_eight(figure_eight):
    """
    使用matplotlib绘制8字形
    
    参数:
    figure_eight: 由shapely创建的8字形几何体
    """
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 获取几何体的外部坐标
    x, y = figure_eight.exterior.xy
    
    # 绘制几何体的轮廓
    ax.plot(x, y, 'b-', linewidth=2)
    
    # 填充几何体
    ax.fill(x, y, alpha=0.3, fc='blue', ec='blue')
    
    # 设置坐标轴等比例
    ax.set_aspect('equal')
    
    # 设置标题
    ax.set_title('Figure Eight (8字形) - 由直线构成')
    
    # 添加坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # 显示网格
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 创建由直线组成的8字形几何体
    figure_eight = create_figure_eight_with_lines(size=2)
    
    # 绘制8字形
    plot_figure_eight(figure_eight)
    
    print("由直线组成的8字形已成功绘制！")

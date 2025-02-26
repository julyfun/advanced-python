from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from typing import List, Union
from typeguard import typechecked

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def erode_polygon(polygon, width):
    """
    对多边形进行向内腐蚀处理
    
    参数:
    polygon: 输入的多边形
    width: 腐蚀的宽度
    
    返回:
    腐蚀后的多边形
    """
    # 使用buffer实现腐蚀效果
    eroded_polygon = polygon.buffer(-width)
    
    return eroded_polygon

@typechecked
def get_eroded_polygons(polygon: Polygon, width: float, max_num_layers: int = 21) -> List[Polygon]:
    """
    生成多层腐蚀的多边形序列
    
    参数:
    polygon: 原始多边形
    width: 每层腐蚀的宽度
    max_num_layers: 最大腐蚀层数
    
    返回:
    腐蚀多边形列表，从外到内排序（第一个是原始多边形）
    """
    eroded_polygons = [polygon]  # 从原始多边形开始
    
    for i in range(max_num_layers):
        # 对最后一个多边形进行腐蚀
        eroded = erode_polygon(eroded_polygons[-1], width)
        
        # 如果腐蚀结果为空或无效，停止腐蚀
        if eroded.is_empty or not eroded.is_valid:
            break
            
        # 如果腐蚀后的面积太小，停止腐蚀
        if eroded.area < polygon.area * 0.01:  # 面积小于原始面积的1%
            break
            
        eroded_polygons.append(eroded)
    
    return eroded_polygons


def quick_plot(geometry: Union[Polygon, MultiPolygon], color='blue', alpha=0.3, title='Polygon Visualization', 
               ax=None, show=True, label=None):
    """
    快速绘制 Shapely 的 Polygon 或 MultiPolygon 对象，并标注顶点编号
    
    参数:
    geometry: Polygon 或 MultiPolygon 对象
    color: 填充颜色，默认为蓝色
    alpha: 透明度，默认为 0.3
    title: 图表标题，默认为 'Polygon Visualization'
    ax: matplotlib axes对象，若为None则创建新图
    show: 是否立即显示图形，默认为True
    label: 图例标签，默认为None
    
    返回:
    ax: matplotlib axes对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    def plot_polygon_with_numbers(poly):
        # 绘制外边界
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=alpha, fc=color, label=label)
        ax.plot(x, y, color=color, linewidth=2)
        
        # 标注外边界顶点编号
        for i, (xi, yi) in enumerate(zip(x[:-1], y[:-1])):  # [:-1]去掉重复的最后一个点
            ax.text(xi, yi, str(i), fontsize=10, 
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # 绘制内部孔洞（如果有的话）
        for interior in poly.interiors:
            x, y = interior.xy
            ax.plot(x, y, color=color, linewidth=2)
            ax.fill(x, y, alpha=alpha, fc='white')
            # 标注孔洞顶点编号
            for i, (xi, yi) in enumerate(zip(x[:-1], y[:-1])):
                ax.text(xi, yi, f'h{i}', fontsize=10, 
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    if isinstance(geometry, Polygon):
        plot_polygon_with_numbers(geometry)
    else:  # MultiPolygon
        for i, poly in enumerate(geometry.geoms):
            plot_polygon_with_numbers(poly)
            # 在每个多边形的中心添加多边形编号
            centroid = poly.centroid
            ax.text(centroid.x, centroid.y, f'Poly{i}', fontsize=12, 
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    ax.set_aspect('equal')
    ax.grid(True)
    if title and ax.get_title() == '':  # 只在没有标题时设置标题
        ax.set_title(title)
    
    if show:
        plt.tight_layout()
        plt.show()
        plt.pause(0)
    
    return ax

def plot_multiple_erosion_layers(polygon, widths, colors=None, alphas=None):
    """
    在同一个图上展示多层腐蚀效果
    
    参数:
    polygon: 原始多边形
    widths: 腐蚀宽度列表，从小到大排序
    colors: 每层使用的颜色列表，若为None则自动生成
    alphas: 每层的透明度，若为None则自动生成
    """
    if colors is None:
        # 使用颜色映射生成渐变色
        cmap = plt.cm.viridis
        colors = [cmap(i) for i in np.linspace(0, 0.8, len(widths) + 1)]
    
    if alphas is None:
        # 透明度从不透明到透明
        alphas = np.linspace(0.8, 0.2, len(widths) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 存储所有腐蚀后的多边形
    eroded_polygons = [polygon]  # 从原始多边形开始
    for width in widths:
        eroded = erode_polygon(eroded_polygons[-1], width)
        if not eroded.is_empty and eroded.is_valid:
            eroded_polygons.append(eroded)
        else:
            break
    
    # 从内到外绘制（先绘制最内层）
    for i, poly in enumerate(reversed(eroded_polygons)):
        label = f'腐蚀 {sum(widths[:len(eroded_polygons)-i-1]):.1f}' if i < len(eroded_polygons)-1 else '原始'
        quick_plot(poly, color=colors[i], alpha=alphas[i], ax=ax, show=False, label=label)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('多层腐蚀效果叠加展示')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 创建8字形多边形
    
    # 使用原来的定义的多边形也可以测试
    points = [
        (0, 0), (2, 1), (0, 2), (-2, 3), (-1, 4),
        (0, 3), (2, 4), (3, 2), (2, 0), (0, -1),
        (-2, 0), (0, 0)
    ]
    custom_polygon = Polygon(points)
    
    # 对自定义多边形进行腐蚀演示
    plot_multiple_erosion_layers(custom_polygon, [0.2, 0.2, 0.2, 0.2])

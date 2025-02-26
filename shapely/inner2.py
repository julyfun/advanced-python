from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon
from shapely.ops import unary_union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

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

def quick_plot(geometry, color='blue', alpha=0.3, title='Polygon Visualization', 
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


@dataclass
class PathNode:
    """表示盘旋路径中的一个节点"""
    x: float
    y: float
    next: Optional['PathNode'] = None  # 主路径的下一个节点
    branch: Optional['PathNode'] = None  # 分支路径的起始节点
    layer: int = 0  # 所在腐蚀层级，0表示最外层
    
    def __repr__(self):
        return f"PathNode(({self.x:.2f}, {self.y:.2f}), layer={self.layer})"

def find_nearest_point(point: Tuple[float, float], points: List[Tuple[float, float]]) -> int:
    """找到距离给定点最近的点的索引"""
    distances = [(i, np.sqrt((p[0] - point[0])**2 + (p[1] - point[1])**2)) 
                for i, p in enumerate(points)]
    return min(distances, key=lambda x: x[1])[0]

def extract_polygon_points(polygon) -> List[Tuple[float, float]]:
    """从多边形提取所有顶点坐标"""
    if isinstance(polygon, MultiPolygon):
        points = []
        for poly in polygon.geoms:
            points.extend(list(poly.exterior.coords)[:-1])  # 去掉重复的最后一个点
        return points
    else:
        return list(polygon.exterior.coords)[:-1]  # 去掉重复的最后一个点

def build_spiral_path(polygons: List[Polygon], start_point: Optional[Tuple[float, float]] = None) -> PathNode:
    """
    构建盘旋路径树
    
    参数:
    polygons: 按腐蚀顺序排列的多边形列表（从外到内）
    start_point: 起始点坐标，如果为None则自动选择
    
    返回:
    PathNode: 路径树的根节点
    """
    if not polygons:
        return None
    
    # 如果没有指定起始点，选择最外层多边形最左边的点
    if start_point is None:
        outer_points = extract_polygon_points(polygons[0])
        start_point = min(outer_points, key=lambda p: p[0])
    
    def build_layer_path(current_poly, layer: int, start_idx: int) -> PathNode:
        """构建单层的路径"""
        points = extract_polygon_points(current_poly)
        if not points:
            return None
        
        # 创建当前层的所有节点
        nodes = [PathNode(x=p[0], y=p[1], layer=layer) for p in points]
        if not nodes:
            return None
        
        # 从起始点开始，按照最近邻原则连接节点
        used = set()
        current_idx = start_idx
        first_node = nodes[current_idx]
        current_node = first_node
        used.add(current_idx)
        
        while len(used) < len(nodes):
            # 找到最近的未使用节点
            remaining_points = [(i, p) for i, p in enumerate(points) if i not in used]
            current_point = (current_node.x, current_node.y)
            distances = [(i, np.sqrt((p[0] - current_point[0])**2 + (p[1] - current_point[1])**2))
                        for i, p in remaining_points]
            next_idx = min(distances, key=lambda x: x[1])[0]
            
            current_node.next = nodes[next_idx]
            current_node = current_node.next
            used.add(next_idx)
        
        # 闭合路径
        current_node.next = first_node
        return first_node
    
    # 构建主路径（从外到内）
    root = None
    prev_node = None
    
    for layer, polygon in enumerate(polygons):
        if isinstance(polygon, MultiPolygon):
            # 处理多个多边形的情况
            for poly in polygon.geoms:
                points = extract_polygon_points(poly)
                if not points:
                    continue
                    
                # 找到距离前一个节点最近的起始点
                if prev_node:
                    start_idx = find_nearest_point(
                        (prev_node.x, prev_node.y), 
                        points
                    )
                else:
                    start_idx = 0
                
                path = build_layer_path(poly, layer, start_idx)
                if not root:
                    root = path
                elif prev_node:
                    prev_node.branch = path
                
                # 更新前一个节点
                current = path
                while current.next != path:
                    current = current.next
                prev_node = current
        else:
            # 处理单个多边形的情况
            points = extract_polygon_points(polygon)
            if not points:
                continue
                
            # 找到距离前一个节点最近的起始点
            if prev_node:
                start_idx = find_nearest_point(
                    (prev_node.x, prev_node.y), 
                    points
                )
            else:
                start_idx = 0
            
            path = build_layer_path(polygon, layer, start_idx)
            if not root:
                root = path
            elif prev_node:
                prev_node.next = path
            
            # 更新前一个节点
            current = path
            while current.next != path:
                current = current.next
            prev_node = current
    
    return root

def visualize_spiral_path(root: PathNode, ax=None, show=True):
    """可视化盘旋路径"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    def plot_path(node: PathNode, color=None, linestyle='-'):
        if node is None:
            return
        
        # 如果没有指定颜色，根据层级生成颜色
        if color is None:
            color = plt.cm.viridis(node.layer / 10)
        
        visited = set()
        current = node
        while current and id(current) not in visited:
            visited.add(id(current))
            
            # 绘制到下一个节点的连线
            if current.next:
                ax.plot([current.x, current.next.x], 
                       [current.y, current.next.y], 
                       color=color, linestyle=linestyle)
                
                # 添加箭头指示方向
                mid_x = (current.x + current.next.x) / 2
                mid_y = (current.y + current.next.y) / 2
                dx = current.next.x - current.x
                dy = current.next.y - current.y
                ax.arrow(mid_x - dx/4, mid_y - dy/4, dx/8, dy/8,
                        head_width=0.1, head_length=0.2, fc=color, ec=color)
            
            # 递归处理分支
            if current.branch:
                plot_path(current.branch, plt.cm.Reds(current.layer / 10), '--')
            
            current = current.next
            if current == node:  # 如果回到起点就停止
                break
    
    plot_path(root)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Spiral Path Visualization')
    
    if show:
        plt.tight_layout()
        plt.show()
        plt.pause(0)
    
    return ax

def get_eroded_polygons(polygon: Polygon, width: float, num_layers: int = 5) -> List[Polygon]:
    """
    生成多层腐蚀的多边形序列
    
    参数:
    polygon: 原始多边形
    width: 每层腐蚀的宽度
    num_layers: 腐蚀层数，默认为5
    
    返回:
    腐蚀多边形列表，从外到内排序（第一个是原始多边形）
    """
    eroded_polygons = [polygon]  # 从原始多边形开始
    
    for i in range(num_layers):
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

def find_point_at_distance(polygon: Polygon, width: float) -> Tuple[Tuple[float, float], int]:
    """
    在多边形边界上查找距离起始点指定距离的点
    
    参数:
    polygon: 输入的多边形
    width: 要查找的距离
    
    返回:
    Tuple[Tuple[float, float], int]: (找到的点坐标, 所在边的起始点索引)
    如果找不到符合条件的点，返回 (None, -1)
    """
    coords = list(polygon.exterior.coords)
    points = coords[:-1]  # 去掉重复的最后一个点
    n = len(points)
    
    def distance(p1, p2):
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def find_point_on_segment(p1, p2, target_dist):
        """在线段上查找距离p1指定距离的点"""
        seg_length = distance(p1, p2)
        if seg_length == 0:
            return None
            
        # 如果目标距离大于线段长度，返回None
        if target_dist > seg_length:
            return None
            
        # 计算插值比例
        t = target_dist / seg_length
        # 线性插值计算点的坐标
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        return (x, y)
    
    # 遍历所有边
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        
        # 在当前边上查找目标点
        point = find_point_on_segment(p1, p2, width)
        if point is not None:
            return point, i
    
    return None, -1

if __name__ == "__main__":
    # 创建8字形多边形
    
    # 使用原来的定义的多边形也可以测试
    points = [
        (0, 0), (2, 1), (0, 2), (-2, 3), (-1, 4),
        (0, 3), (2, 4), (3, 2), (2, 0), (0, -1),
        (-2, 0), (0, 0)
    ]
    custom_polygon = Polygon(points)

    # 生成腐蚀序列
    eroded_polygons = get_eroded_polygons(custom_polygon, width=0.3, num_layers=5)
    
    # 可视化腐蚀结果
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, poly in enumerate(eroded_polygons):
        color = plt.cm.viridis(i / len(eroded_polygons))
        quick_plot(poly, color=color, alpha=0.3, ax=ax, show=False, 
                  label=f'Layer {i}')
    ax.legend()
    plt.show()

    # 生成并可视化盘旋路径
    root = build_spiral_path(eroded_polygons)
    visualize_spiral_path(root)

    # 遍历路径示例
    current = root
    while current:
        print(f"Point: ({current.x}, {current.y}), Layer: {current.layer}")
        if current.branch:
            print(f"Branch at: ({current.x}, {current.y})")
        current = current.next
        if current == root:
            break

    # 创建测试多边形
    points = [
        (0, 0), (2, 1), (0, 2), (-2, 3), (-1, 4),
        (0, 3), (2, 4), (3, 2), (2, 0), (0, -1),
        (-2, 0), (0, 0)
    ]
    test_polygon = Polygon(points)
    
    # 测试函数
    width = 1.0
    found_point, edge_index = find_point_at_distance(test_polygon, width)
    
    # 可视化结果
    fig, ax = plt.subplots(figsize=(8, 8))
    quick_plot(test_polygon, ax=ax, show=False)
    print(f"found_point: {found_point}, edge_index: {edge_index}")
    
    if found_point is not None:
        # 标记找到的点
        ax.plot(found_point[0], found_point[1], 'ro', markersize=10, label=f'Found point (edge {edge_index})')
        # 标记起始点
        start_point = points[edge_index]
        ax.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start point')
        
        # 绘制距离标记
        ax.plot([start_point[0], found_point[0]], 
                [start_point[1], found_point[1]], 
                'r--', label=f'Distance = {width}')
    
    ax.legend()
    plt.show()

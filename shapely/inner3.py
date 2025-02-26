from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union, nearest_points
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from typing import List, Union, NamedTuple, Optional
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
def get_eroded_polygons(polygon, width, max_num_layers: Union[int, None] = 21) -> List[Polygon]:
    """
    生成多层腐蚀的多边形序列
    
    参数:
    polygon: 原始多边形
    width: 每层腐蚀的宽度
    max_num_layers: 最大腐蚀层数，None表示不限制
    
    返回:
    腐蚀多边形列表，从外到内排序（第一个是原始多边形）
    """
    eroded_polygons = [polygon]  # 从原始多边形开始
    
    i = 0
    while max_num_layers is None or i < max_num_layers:
        # 对最后一个多边形进行腐蚀
        eroded = erode_polygon(eroded_polygons[-1], width)
        
        # 如果腐蚀结果为空或无效，停止腐蚀
        if eroded.is_empty or not eroded.is_valid:
            break
            
        # 如果腐蚀后的面积太小，停止腐蚀
        if eroded.area < polygon.area * 0.01:  # 面积小于原始面积的1%
            break
            
        eroded_polygons.append(eroded)
        i += 1
    
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


def find_points_at_distance_from_line(line: LineString, point: Point, width: float) -> list[Point]:
    """
    找出 LineString 上所有距离给定 Point 为 width 的点
    
    参数:
    line: 输入的线段
    point: 参考点
    width: 目标距离
    
    返回:
    list[Point]: 所有满足条件的点的列表
    """
    # 创建一个以给定点为中心、半径为width的圆
    circle = point.buffer(width)
    circle_boundary = circle.boundary
    
    # 计算圆边界与线段的交点
    if line.intersects(circle_boundary):
        intersection = line.intersection(circle_boundary)
        
        # 交点可能是Point或MultiPoint
        if intersection.geom_type == 'Point':
            return [intersection]
        elif intersection.geom_type == 'MultiPoint':
            return list(intersection.geoms)
        
    return []

@typechecked
def find_point_at_distance(polygon: Polygon, width: float) -> tuple[Point, int]:
    """
    在多边形边界上查找距离0号顶点指定距离的点
    
    参数:
    polygon: 输入的多边形
    width: 指定的距离
    
    返回:
    tuple: ((x, y), edge_index)
        - (x, y): 找到的点的坐标
        - edge_index: 点所在的边的索引（从0开始）
    """
    # 排除重复的最后一个点
    coords = list(polygon.exterior.coords)[0:-1]
    point0 = Point(coords[0])  # 0号点
    
    # 从0..-1边开始,依次检查每条边
    # 0 .. -1; -n + 1.. -n
    for i in range(0, -len(coords), -1):
        # 获取当前边的起点和终点
        start = Point(coords[i])
        end = Point(coords[i-1])
        
        # 创建当前边的LineString
        edge = LineString([start, end])
        
        # 找到所有距离0号点为width的点
        points_at_width = find_points_at_distance_from_line(edge, point0, width)
                
        if points_at_width:
            point = min(points_at_width, key=lambda x: x.distance(start))
            return point, i
            
    raise ValueError(f"找不到距离为 {width} 的点")

@typechecked
def find_nearest_point_on_exterior(point: Point, ext_polygon: Polygon) -> tuple[Point, int, float]:
    """
    找到点在外部多边形上的最近点及其所在边的末端点编号
    
    参数:
    point: 待检查的点
    ext_polygon: 外部多边形
    
    返回:
    tuple: (nearest_point, edge_end_idx, distance)
        - nearest_point: 最近点
        - edge_end_idx: 最近点所在边的末端点编号
        - distance: 最近点到边末端点的距离
    """
    coords = list(ext_polygon.exterior.coords)[:-1]
    min_dist = float('inf')
    nearest_point = None
    edge_end_idx = -1
    dist_to_end = 0
    
    for i in range(len(coords)):
        start = Point(coords[i])
        end = Point(coords[(i+1) % len(coords)])
        edge = LineString([start, end])
        
        curr_nearest = nearest_points(point, edge)[1]
        dist = point.distance(curr_nearest)
        
        if dist < min_dist:
            min_dist = dist
            nearest_point = curr_nearest
            edge_end_idx = (i+1) % len(coords)
            dist_to_end = curr_nearest.distance(end)
            
    return nearest_point, edge_end_idx, dist_to_end

class FromEnd(NamedTuple):
    """从内部多边形端点创建的新多边形"""
    polygon: Polygon

class FromMid(NamedTuple):
    """从内部多边形中间点创建的新多边形"""
    polygon: Polygon
    edge_end_idx: int
    nearest_point: Point
    dist_to_end: float

PolygonInfo = Union[FromEnd, FromMid]

@typechecked
def check_intersection_excluding_endpoints(line: LineString, polygon_boundary: LineString) -> bool:
    """
    检查线段与多边形边界是否相交，不考虑端点处的相交
    
    参数:
    line: 待检查的线段
    polygon_boundary: 多边形边界
    
    返回:
    bool: 如果存在非端点的相交则返回True
    """
    if not line.intersects(polygon_boundary):
        return False
    
    intersection = line.intersection(polygon_boundary)
    if intersection.is_empty:
        return False
    
    # 获取线段的端点
    line_endpoints = [Point(p) for p in line.coords]
    
    # 如果交点是Point
    if intersection.geom_type == 'Point':
        # 检查交点是否是线段的端点
        return not any(intersection.equals(ep) for ep in line_endpoints)
    
    # 如果交点是MultiPoint，检查每个交点
    if intersection.geom_type == 'MultiPoint':
        for point in intersection.geoms:
            # 如果有任何一个交点不是端点，则认为相交
            if not any(point.equals(ep) for ep in line_endpoints):
                return True
    
    return False

@typechecked
def find_nearest_point_and_create_polygon(ext: Polygon, interior: Union[Polygon, MultiPolygon], width: float, 
                                        tolerance: float = 1e-10) -> tuple[int, Point, List[PolygonInfo]]:
    """
    在外部多边形上找到距离0号顶点width距离的点pt1，然后根据条件创建新的多边形
    
    参数:
    ext: 外部多边形
    interior: 内部多边形或多个多边形
    width: 指定的距离
    tolerance: 距离比较时的容差
    
    返回:
    tuple: (end_idx, pt1, polygon_infos)
        - end_idx: 外部裁剪后的end_idx
        - pt1: 外部裁剪后的最后一个点
        - polygon_infos: 新创建的多边形信息列表，每个元素可能是:
            - FromEnd: 从端点创建的新多边形
            - FromMid: 从中间点创建的新多边形，包含额外的外部多边形信息
    """
    # 找到外部多边形上距离0号顶点width距离的点
    pt1, ed_idx = find_point_at_distance(ext, width)
    
    # 将interior转换为多边形列表
    interior_polygons = list(interior.geoms) if isinstance(interior, MultiPolygon) else [interior]
    
    # 存储所有创建的多边形信息
    polygon_infos: List[PolygonInfo] = []
    
    for curr_interior in interior_polygons:
        int_coords = list(curr_interior.exterior.coords)[:-1]
        
        # 找到距离pt1最近的边上的点
        min_dist = float('inf')
        nearest_edge_start = 0
        nearest_point = None
        
        for i in range(len(int_coords)):
            start = Point(int_coords[i])
            end = Point(int_coords[(i + 1) % len(int_coords)])
            edge = LineString([start, end])
            curr_nearest = nearest_points(pt1, edge)[1]
            dist = pt1.distance(curr_nearest)
            
            if dist < min_dist:
                min_dist = dist
                nearest_edge_start = i
                nearest_point = curr_nearest
        
        # 检查最近点是否 FromEnd 满足条件
        connection_line = LineString([pt1, nearest_point])
        if (min_dist > width + tolerance or 
            check_intersection_excluding_endpoints(connection_line, ext.exterior)):
            # 需要检查所有顶点
            vertex_nearest_info = []
            
            for i, coord in enumerate(int_coords):
                pt_i = Point(coord)
                pt_i_nearest, edge_end_idx, dist_to_end = find_nearest_point_on_exterior(pt_i, ext)
                vertex_nearest_info.append((pt_i, pt_i_nearest, edge_end_idx, dist_to_end, i))
            
            # 按照edge_end_idx降序和dist_to_end升序排序
            vertex_nearest_info.sort(key=lambda x: (-x[2], x[3]))
            best_info = vertex_nearest_info[0]
            
            # 创建从端点的新多边形
            start_point = best_info[0]
            vertex_idx = best_info[4]
            
            # 创建新的顶点序列
            new_coords = [start_point]
            for i in range(vertex_idx + 1, len(int_coords)):
                new_coords.append(Point(int_coords[i]))
            for i in range(0, vertex_idx):
                new_coords.append(Point(int_coords[i]))
            new_coords.append(start_point)
            
            new_polygon = Polygon([(p.x, p.y) for p in new_coords])
            polygon_infos.append(FromEnd(polygon=new_polygon))
        else:
            # 创建从中间点的新多边形
            new_coords = [nearest_point]
            for i in range(nearest_edge_start + 1, len(int_coords)):
                new_coords.append(Point(int_coords[i]))
            for i in range(0, nearest_edge_start + 1):
                new_coords.append(Point(int_coords[i]))
            new_coords.append(nearest_point)
            
            new_polygon = Polygon([(p.x, p.y) for p in new_coords])
            polygon_infos.append(FromMid(polygon=new_polygon, edge_end_idx=ed_idx, nearest_point=pt1, dist_to_end=dist_to_end))
    
    return ed_idx, pt1, polygon_infos

class SpiralNode(NamedTuple):
    """表示盘旋图中的一个节点"""
    points: List[Point]  # 从0号点到最后一个点的链
    children: List['SpiralNode']  # 子节点列表

@typechecked
def create_spiral_tree(polygon: Polygon, width: float) -> tuple[SpiralNode, int]:
    """
    生成多边形的盘旋树形图
    
    参数:
    polygon: 输入的多边形
    width: 腐蚀宽度
    
    返回:
    tuple: (root_node, end_idx)
        - root_node: 树的根节点
        - end_idx: 最外层多边形的end_idx
    """
    def _create_node(curr_polygon: Polygon) -> tuple[Optional[SpiralNode], int]:
        # 获取腐蚀后的多边形
        eroded = get_eroded_polygons(curr_polygon, width, 1)
        if len(eroded) < 2:  # 如果无法继续腐蚀
            return None, -1
        
        interior_polygon = eroded[1]
        
        # 获取当前层的信息
        end_idx, pt1, polygon_infos = find_nearest_point_and_create_polygon(
            curr_polygon, interior_polygon, width
        )
        
        # 创建当前多边形的点链（从0号点到pt1）
        curr_points = []
        coords = list(curr_polygon.exterior.coords)[:-1]  # 去掉重复的最后一个点
        
        # 添加从0到end_idx-1的点
        assert end_idx <= 0, f"end_idx: {end_idx}"
        for i in range(len(coords) + end_idx):
            curr_points.append(Point(coords[i]))
        curr_points.append(pt1)  # 添加pt1
        
        # 处理所有子节点
        children = []
        next_child = None  # 用于存储FromMid情况下的后续子节点
        
        for info in polygon_infos:
            if isinstance(info, FromEnd):
                # 直接连接到内部多边形的0号点
                child_node, _ = _create_node(info.polygon)
                if child_node is not None:
                    children.append(child_node)
            else:  # FromMid
                # 创建内部多边形的节点
                child_node, _ = _create_node(info.polygon)
                if child_node is not None:
                    # 保存当前点链的长度（插入点的位置）
                    insert_idx = len(curr_points) - 1
                    
                    # 添加中间点
                    curr_points.append(info.nearest_point)
                    
                    # 如果有后续子节点，将其添加到children中
                    if next_child is not None:
                        children.append(next_child)
                    
                    # 创建一个新的节点，包含插入点和两个子节点
                    mid_children = []
                    if child_node is not None:
                        mid_children.append(child_node)  # 连向内部多边形
                    if insert_idx < len(curr_points) - 1:
                        # 创建一个包含剩余点的子节点
                        remaining_points = curr_points[insert_idx+1:]
                        if remaining_points:
                            mid_children.append(SpiralNode(points=remaining_points, children=[]))
                    
                    # 更新当前点链，只保留到插入点
                    curr_points = curr_points[:insert_idx+1]
                    
                    # 添加包含插入点的节点
                    children.append(SpiralNode(
                        points=[info.nearest_point],
                        children=mid_children
                    ))
        
        return SpiralNode(points=curr_points, children=children), end_idx
    
    # 创建整个树
    root_node, final_end_idx = _create_node(polygon)
    if root_node is None:
        raise ValueError("无法创建盘旋图")
    
    return root_node, final_end_idx

def plot_spiral_tree(root: SpiralNode, ax=None, color='blue', alpha=0.3, show=True, show_numbers=False):
    """
    绘制盘旋树形图
    
    参数:
    root: 树的根节点
    ax: matplotlib axes对象
    color: 线条颜色
    alpha: 透明度
    show: 是否显示图形
    show_numbers: 是否显示点的编号
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    def plot_node(node: SpiralNode, color_idx: int):
        # 绘制当前节点的点链
        points = node.points
        for i in range(len(points) - 1):
            x = [points[i].x, points[i+1].x]
            y = [points[i].y, points[i+1].y]
            ax.plot(x, y, color=plt.cm.viridis(color_idx/4), linewidth=2)
            
            # 在每个点上标注编号
            if show_numbers:
                ax.text(points[i].x, points[i].y, f'L{color_idx}_{i}', 
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # 标注最后一个点
        if points and show_numbers:
            i = len(points) - 1
            ax.text(points[i].x, points[i].y, f'L{color_idx}_{i}', 
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # 绘制到子节点的连接线
        for child in node.children:
            if child.points:  # 确保子节点有点
                # 绘制从当前节点最后一个点到子节点第一个点的连线
                x = [points[-1].x, child.points[0].x]
                y = [points[-1].y, child.points[0].y]
                ax.plot(x, y, '--', color=plt.cm.viridis((color_idx+1)/4), linewidth=1.5)
                
                # 递归绘制子节点
                plot_node(child, color_idx + 1)
    
    plot_node(root, 0)
    ax.set_aspect('equal')
    ax.grid(True)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax

if __name__ == "__main__":
    # 创建8字形多边形
    
    # 使用原来的定义的多边形也可以测试
    width = 0.2
    points = [
        (0, 0), (2, 1), (0, 2), (-2, 3), (-1, 4),
        (0, 3), (2, 4), (3, 2), (2, 0), (0, -1),
        (-2, 0), (0, 0)
    ]
    custom_polygon = Polygon(points)
    
    # 对自定义多边形进行腐蚀演示
    # plot_multiple_erosion_layers(custom_polygon, [width] * 5)
    # point, edge_index = find_point_at_distance(custom_polygon, width)
    # print(f"找到的点坐标: {point}")
    # print(f"所在边的索引: {edge_index}")

    # # plot it
    # fig, ax = plt.subplots(figsize=(8, 8))
    # quick_plot(custom_polygon, ax=ax, show=False)
    # ax.plot(point.x, point.y, 'ro', markersize=10, label=f'Found point (edge {edge_index})')
    # ax.legend()
    # plt.show()

    # 创建一个内部多边形用于测试
    # 改为腐蚀 0.2 得到的第一个多边形
    eroded_polygons = get_eroded_polygons(custom_polygon, width, 1)
    interior_polygon = eroded_polygons[1]
    
    # 绘制结果
    # fig, ax = plt.subplots(figsize=(10, 10))
    # quick_plot(custom_polygon, color='blue', alpha=0.3, ax=ax, show=False, label='External')
    
    # # 绘制所有新创建的多边形
    # colors = ['red', 'green', 'purple', 'orange']  # 为不同类型准备不同颜色
    
    # ax.legend()
    # plt.show()

    # # 创建一个MultiPolygon用于测试
    # interior_polygon2 = Polygon([(1.5, 1.5), (2.5, 2), (2, 2.5), (1.5, 2), (1.5, 1.5)])
    # interior_multi = MultiPolygon([interior_polygon, interior_polygon2])
    
    # # 测试新函数
    # end_idx, pt1, polygon_infos = find_nearest_point_and_create_polygon(custom_polygon, interior_multi, width)
    
    # 测试盘旋树形图
    spiral_root, final_end_idx = create_spiral_tree(custom_polygon, width)
    
    # 不显示编号的版本
    plot_spiral_tree(spiral_root, show_numbers=False)
    
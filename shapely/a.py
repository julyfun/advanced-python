from shapely.geometry import Point, LineString, Polygon, MultiLineString
from shapely.ops import unary_union
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def generate_spiral_skeleton(polygon, width):
    """
    生成多边形的盘旋骨架结构
    Args:
        polygon: 输入的多边形
        width: 每次腐蚀的宽度
    Returns:
        list of LineStrings: 骨架线段列表
    """
    skeleton_lines = []
    current_polygon = polygon
    
    while True:
        # 进行腐蚀
        next_polygon = current_polygon.buffer(-width)
        
        # 检查腐蚀后的多边形是否有效
        if next_polygon.is_empty or not next_polygon.is_valid or next_polygon.area < 0.0001:
            break
            
        # 获取当前多边形的边界
        if isinstance(current_polygon, Polygon):
            current_boundaries = [current_polygon.exterior]
        else:  # MultiPolygon
            current_boundaries = [geom.exterior for geom in current_polygon.geoms]
            
        # 获取下一个多边形的边界
        if isinstance(next_polygon, Polygon):
            next_boundaries = [next_polygon.exterior]
        else:  # MultiPolygon
            next_boundaries = [geom.exterior for geom in next_polygon.geoms]
            
        # 为每个下一级边界找到连接线
        for current_boundary in current_boundaries:
            for next_boundary in next_boundaries:
                # 在边界上均匀采样点
                current_length = current_boundary.length
                next_length = next_boundary.length
                
                # 每隔固定距离采样点
                sample_distance = width * 2
                
                # 在当前边界上采样点
                current_distances = np.arange(0, current_length, sample_distance)
                current_points = [current_boundary.interpolate(d) for d in current_distances]
                
                # 在下一个边界上采样点
                next_distances = np.arange(0, next_length, sample_distance)
                next_points = [next_boundary.interpolate(d) for d in next_distances]
                
                # 为每个当前点找到最近的下一层点
                for curr_point in current_points:
                    min_dist = float('inf')
                    nearest_next_point = None
                    
                    for next_point in next_points:
                        dist = curr_point.distance(next_point)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_next_point = next_point
                    
                    if nearest_next_point and min_dist < width * 2:
                        connection_line = LineString([curr_point, nearest_next_point])
                        skeleton_lines.append(connection_line)
        
        current_polygon = next_polygon
    
    return skeleton_lines

# 使用原来的8字形多边形
points = [
    (0, 0), (2, 1), (0, 2), (-2, 3), (-1, 4),
    (0, 3), (2, 4), (3, 2), (2, 0), (0, -1),
    (-2, 0), (0, 0)
]

# 创建原始多边形
polygon1 = Polygon(points)

# 生成盘旋骨架
skeleton_lines = generate_spiral_skeleton(polygon1, 0.2)

# 创建新的图形
fig, ax = plt.subplots(figsize=(8, 10))

# 绘制原始多边形
x, y = polygon1.exterior.xy
ax.plot(x, y, 'b-', label='原始多边形', alpha=0.3)
ax.fill(x, y, alpha=0.1, fc='b')

# 绘制骨架线段
for line in skeleton_lines:
    x, y = line.xy
    ax.plot(x, y, 'r-', linewidth=1)

# 设置图形属性
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title('多边形盘旋骨架示例')

plt.show()


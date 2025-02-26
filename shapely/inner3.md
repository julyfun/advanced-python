新增一个函数，输入一个 Polygon 调用这些函数生成盘旋图。盘旋图由点组成，并且是树形数据结构。大致流程：

1. 对于 Polygon 求


修改 find_nearest_point_and_create_polygon，

1. interior 是 MultiPolygon 时，需要对每个 interior 都进行类似逻辑
2. 找到内部多边形上距离pt1最近的点 pt2，如果距离大于 width (注意精度问题)，或者 pt1-pt2 连线与外部多边形相交，则按照下满逻辑创建多边形 FromMid
    - 遍历内部多边形的端点 pt_i，求其相对于外部多边形的最近点 pt_i_nearest 和对应的在外部多边形上的末端点编号 pt_i_nearest_edge_ed_idx
        - 以及得到 pt_i_nearest 到 pt_i_nearest_edge_ed 的距离 dist_i
    - 所有内部端点会得到多个 pt_i_nearest，则优先取 pt_i_nearest_edge_ed_idx 最大的那个，如果还有多个，则取 dist_i 最小的那个
    - 得到最优 pt_i_nearest 后，返回内部多边形从 pt_i 开始的点和 pt_i_nearest

---

返回类型不是  tuple[Polygon, int]，而是:

- 一个 int 表示外部裁剪后的的 end_idx
- 一个 Point 表示外部裁剪后的最后一个点(pt1)
- 一个联合类型列表（如果内部多边形为 MultiPolygon，则返回多个）：
    1. FromEnd:
        - Polygon: 新创建的内部多边形
    2. FromMid:
        - Polygon: 新创建的内部多边形
        - int: 对应的 pt_i_nearest_edge_ed_idx
        - Point: 对应的 pt_i_nearest
    

---

利用现有函数，生成盘旋图,
- 输入:
    - 一个 Polygon
    - 一个 float 表示宽度
腐蚀一次 width 的多边形，执行 find_nearest_point_and_create_polygon 函数，得到一个 Info 列表
- 外部多边形改为从 0 号点开始到 .. ed_idx - 1，并连向 pt1，形成一个链（用树形图存储）
- 如果内部 find_nearest_point_and_create_polygon 是 FromEnd，则从 pt1 连向 FromEnd.polygon 的 0 号点
- 如果内部 find_nearest_point_and_create_polygon 是 FromMid，
    - 不需要 pt1 连向 FromMid.polygon 的 0 号点
    - 显然它在 edge_end_idx - 1 和 edge_end_idx 这条边上。对所有的外部边 idx, 先缓存 idx - 1 -> idx 上的所有 nearest_points.
- 所有内部 Polygon 缓存结束后：
- 对于 idx 存储的 nearest_points，如果不为空, nearest_points 按照 dist_to_end 从大到小排序
    - 删除 idx - 1 -> idx 这条边:
    - 按照 dist_to_end 从大到小排序，idx - 1 连向 nearest_points[0], nearest_points[0] 连向 nearest_points[1], ...
    - nearest_points[-1] 连向 idx
    - nearest_points[i] 还需要连向内部 Polygon 的 0 号点
- 对于每个内部 Polygon，执行一样的操作

- 最后返回单个树形图(每个节点含有一个 Point)和一个 ed_idx，为最外层的 ed_idx

---



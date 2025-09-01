import cv2
import numpy as np
import argparse
from typing import Tuple, Optional

def get_video_info(video_path: str) -> Tuple[int, int, float, int]:
    """获取视频信息：宽度、高度、帧率、总帧数"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    return width, height, fps, frame_count

def extract_first_frame(video_path: str) -> np.ndarray:
    """提取视频的第一帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"无法读取视频第一帧: {video_path}")
    
    return frame

def detect_and_match_features(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """检测特征点并进行匹配"""
    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 使用SIFT检测器
    sift = cv2.SIFT_create()
    
    # 检测关键点和描述符
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        raise ValueError("无法在图像中检测到足够的特征点")
    
    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # 应用Lowe's ratio test筛选好的匹配
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        raise ValueError("匹配的特征点数量不足，无法计算变换矩阵")
    
    # 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    return src_pts, dst_pts

def compute_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """计算4自由度相似变换矩阵（平移、旋转、缩放）"""
    # 使用RANSAC估计相似变换
    # 注意：这里应该是从dst_pts到src_pts，即从video2到video1
    transform_matrix, mask = cv2.estimateAffinePartial2D(
        dst_pts, src_pts,  # 交换了参数顺序
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99
    )
    
    if transform_matrix is None:
        raise ValueError("无法计算相似变换矩阵")
    
    return transform_matrix

def create_blended_video(video1_path: str, video2_path: str, 
                        transform_matrix: np.ndarray, 
                        frame_offset: int = 0,
                        output_path: str = "output_blended.mp4",
                        alpha: float = 0.5) -> None:
    """创建融合后的视频"""
    # 打开两个视频
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("无法打开视频文件")
    
    # 获取视频信息
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 处理帧偏移
    if frame_offset > 0:
        # video2 延迟 frame_offset 帧
        for _ in range(frame_offset):
            cap2.read()
    elif frame_offset < 0:
        # video1 延迟 abs(frame_offset) 帧
        for _ in range(abs(frame_offset)):
            cap1.read()
    
    frame_idx = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1:
            break
        
        if ret2:
            # 对第二个视频帧应用变换
            transformed_frame2 = cv2.warpAffine(frame2, transform_matrix, (width, height))
            
            # 创建掩码，只在有效像素区域进行融合
            mask = np.any(transformed_frame2 != 0, axis=2)
            
            # 融合两帧
            blended_frame = frame1.copy()
            blended_frame[mask] = (alpha * transformed_frame2[mask] + 
                                 (1 - alpha) * frame1[mask]).astype(np.uint8)
        else:
            # 如果第二个视频已经结束，只使用第一个视频
            blended_frame = frame1
        
        out.write(blended_frame)
        frame_idx += 1
        
        # 显示进度
        if frame_idx % 30 == 0:
            print(f"已处理 {frame_idx} 帧")
    
    # 释放资源
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"融合视频已保存到: {output_path}")

def video_match_and_blend(video1_path: str, video2_path: str, 
                         frame_offset: int = 0,
                         output_path: str = "output_blended.mp4",
                         alpha: float = 0.5) -> None:
    """主函数：视频匹配和融合"""
    print("开始视频匹配和融合处理...")
    
    # 1. 验证视频信息
    print("验证视频信息...")
    w1, h1, fps1, fc1 = get_video_info(video1_path)
    w2, h2, fps2, fc2 = get_video_info(video2_path)
    
    print(f"视频1: {w1}x{h1}, {fps1:.2f}fps, {fc1}帧")
    print(f"视频2: {w2}x{h2}, {fps2:.2f}fps, {fc2}帧")
    
    # 验证分辨率和帧率
    assert w1 == w2 and h1 == h2, f"视频分辨率不匹配: {w1}x{h1} vs {w2}x{h2}"
    assert abs(fps1 - fps2) < 0.1, f"视频帧率不匹配: {fps1:.2f} vs {fps2:.2f}"
    
    # 2. 提取第一帧
    print("提取第一帧进行特征匹配...")
    frame1 = extract_first_frame(video1_path)
    frame2 = extract_first_frame(video2_path)
    
    # 3. 特征点检测和匹配
    print("检测和匹配特征点...")
    src_pts, dst_pts = detect_and_match_features(frame1, frame2)
    print(f"找到 {len(src_pts)} 个匹配的特征点")
    
    # 4. 计算相似变换矩阵
    print("计算相似变换矩阵...")
    transform_matrix = compute_similarity_transform(src_pts, dst_pts)
    print(f"变换矩阵:\n{transform_matrix}")
    
    # 5. 创建融合视频
    print(f"创建融合视频，帧偏移: {frame_offset}...")
    create_blended_video(video1_path, video2_path, transform_matrix, 
                        frame_offset, output_path, alpha)
    
    print("视频匹配和融合完成！")

def main():
    parser = argparse.ArgumentParser(description='视频匹配和融合工具')
    parser.add_argument('video1', help='第一个视频文件路径')
    parser.add_argument('video2', help='第二个视频文件路径')
    parser.add_argument('--frame_offset', type=int, default=0, 
                       help='帧偏移量，正数表示video2延迟，负数表示video1延迟')
    parser.add_argument('--output', default='output_blended.mp4', 
                       help='输出视频文件路径')
    parser.add_argument('--alpha', type=float, default=0.5, 
                       help='第二个视频的透明度 (0.0-1.0)')
    
    args = parser.parse_args()
    
    try:
        video_match_and_blend(
            args.video1, 
            args.video2, 
            args.frame_offset, 
            args.output, 
            args.alpha
        )
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
import cv2
import numpy as np
import rerun as rr
import math
from pathlib import Path
import time

# 初始化 rerun (改用英文)
rr.init("6-Axis Robot Arm Trajectory Visualization", spawn=True)

# 读取视频文件
video_path = Path(__file__).parent / "video.mp4"
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"Cannot open video file: {video_path}")
    exit()

# 获取视频参数
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(
    f"Video FPS: {fps}, Resolution: {frame_width}x{frame_height}, Total frames: {total_frames}"
)

# 模拟 6 轴机械臂的参数 (改用英文)
axes_names = ["Base", "Shoulder", "Elbow", "Wrist Pitch", "Wrist Roll", "Tool"]
axes_range = [
    [-180, 180],
    [-90, 90],
    [-120, 120],
    [-180, 180],
    [-120, 120],
    [-360, 360],
]

# 初始化机械臂位置
arm_links = [0.0, 0.3, 0.25, 0.2, 0.1, 0.05]  # 机械臂各关节长度 (meters)
joint_angles = np.zeros(6)  # 各关节角度 (radians)


# 创建一个变换矩阵函数
def transform_matrix(angle, axis, length):
    """创建一个旋转和平移的变换矩阵"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # 根据不同轴创建不同的旋转矩阵
    if axis == 0:  # 绕Z轴旋转
        rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    elif axis == 1:  # 绕Y轴旋转
        rotation = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
    else:  # 绕X轴旋转
        rotation = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])

    # 创建平移向量
    translation = np.array([0, 0, length])

    # 组合旋转和平移
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


# 计算机械臂各关节位置的函数
def compute_arm_positions(joint_angles, arm_links):
    """计算 6 轴机械臂各关节的位置"""
    positions = [np.array([0, 0, 0])]  # 基座位置

    # 初始化累积变换矩阵为单位矩阵
    T = np.eye(4)

    # 机械臂关节轴方向 (交替的 z y z y z y)
    axes = [0, 1, 0, 1, 0, 1]

    for i in range(6):
        # 应用关节旋转
        T = T @ transform_matrix(joint_angles[i], axes[i], arm_links[i])

        # 提取位置
        position = T[:3, 3]
        positions.append(position)

    return positions


# 生成机械臂的轨迹函数
def generate_trajectory(frame_num, total_frames):
    """生成机械臂轨迹，根据当前帧数"""
    t = frame_num / total_frames

    # 为每个关节生成周期性的运动
    joint_angles = np.zeros(6)
    joint_angles[0] = math.sin(t * 2 * math.pi) * math.pi / 2  # 基座
    joint_angles[1] = math.sin(t * 3 * math.pi + 0.5) * math.pi / 4  # 肩部
    joint_angles[2] = math.sin(t * 4 * math.pi + 1.0) * math.pi / 3  # 肘部
    joint_angles[3] = math.sin(t * 5 * math.pi + 1.5) * math.pi / 2  # 手腕俯仰
    joint_angles[4] = math.sin(t * 6 * math.pi + 2.0) * math.pi / 3  # 手腕旋转
    joint_angles[5] = math.sin(t * 7 * math.pi + 2.5) * math.pi  # 工具

    return joint_angles


# 处理每一帧
frame_num = 0
sleep_time = 1.0 / fps  # 控制播放速度

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            # 视频结束，重新开始
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 记录时间戳
        timestamp = frame_num / fps

        # 显示视频帧
        rr.set_time_sequence("frame", frame_num)
        rr.log("video/frame", rr.Image(frame[:, :, ::-1]))  # BGR 转 RGB

        # 生成机械臂轨迹
        joint_angles = generate_trajectory(frame_num, total_frames)

        # 计算机械臂各关节位置
        positions = compute_arm_positions(joint_angles, arm_links)

        # 记录关节角度 (改用英文路径)
        for i in range(6):
            rr.log(
                f"robot_arm/angles/{axes_names[i]}",
                rr.Scalar(joint_angles[i] * 180 / math.pi),
            )  # 转换为角度

        # 可视化机械臂 (改用英文路径)
        points = np.array(positions)
        rr.log(
            "robot_arm/positions", rr.Points3D(points, colors=[255, 0, 0], radii=0.01)
        )
        rr.log(
            "robot_arm/links", rr.LineStrips3D(points, colors=[0, 255, 0], radii=0.005)
        )

        # 显示末端执行器轨迹 (改用英文路径)
        if frame_num % 3 == 0:  # 每 3 帧记录一次轨迹点，避免点太密集
            rr.log(
                "robot_arm/end_effector_trajectory",
                rr.Points3D(
                    positions[-1].reshape(1, 3), colors=[0, 0, 255], radii=0.005
                ),
            )

        # 增加帧计数
        frame_num += 1

        # 控制播放速度
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("Program stopped")
finally:
    cap.release()
    print(f"Processed {frame_num} frames")

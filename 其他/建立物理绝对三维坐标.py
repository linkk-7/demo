import numpy as np
import cv2


def create_absolute_coordinate_system(left_map1, left_map2, right_map1, right_map2,
                                      Q, pts_pix, disp_left, disp_right):
    """
    基于校正后的立体图像，构建物理绝对坐标系，并转换每个标记点坐标。

    参数:
    - left_map1, left_map2: 左相机的校正映射
    - right_map1, right_map2: 右相机的校正映射
    - Q: 立体重建矩阵
    - pts_pix: 以 (u, v) 格式给出的标记点像素坐标 (四个角点)
    - disp_left, disp_right: 左右图像的视差图

    返回:
    - pts_world: 各标记点在绝对坐标系下的物理坐标
    """
    # 步骤 1: 校正图像
    left_rectified = cv2.remap(disp_left, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(disp_right, right_map1, right_map2, cv2.INTER_LINEAR)

    # 步骤 2: 计算3D点
    disp = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=5,
        P1=8 * 3 * 5 ** 2, P2=32 * 3 * 5 ** 2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    ).compute(left_rectified, right_rectified).astype(np.float32) / 16.0

    points_3d = cv2.reprojectImageTo3D(disp, Q)

    # 步骤 3: 获取每个标记点在相机坐标系下的 3D 坐标
    pts_cam = np.array([points_3d[v, u] for (u, v) in pts_pix])  # (4, 3)

    # 步骤 4: 选择一个标记点为原点
    P1, P2, P4 = pts_cam[0], pts_cam[1], pts_cam[3]

    # 计算相机坐标系到绝对坐标系的转换矩阵
    vx = P2 - P1
    vy = P4 - P1
    vz = np.cross(vx, vy)

    ex = vx / np.linalg.norm(vx)
    ey = vy / np.linalg.norm(vy)
    ez = vz / np.linalg.norm(vz)

    R_cw = np.vstack([ex, ey, ez]).T  # 3x3 旋转矩阵
    t_cw = -R_cw @ P1  # 平移向量

    # 步骤 5: 将所有标记点从相机坐标系转换到绝对坐标系
    def cam2world(X_cam):
        return (R_cw @ X_cam.T + t_cw.reshape(3, 1)).T

    pts_world = cam2world(pts_cam)  # 4×3 大小的物理坐标

    return pts_world


# 示例：输入的参数
# 假设你已经获得了标定结果、内外参矩阵等
# 这里直接给出假的内外参数和视差图，真实场景中你应该从标定文件中加载
left_map1, left_map2 = np.eye(3), np.eye(3)  # 模拟的校正映射
right_map1, right_map2 = np.eye(3), np.eye(3)
Q = np.eye(4)  # 假定的重建矩阵
pts_pix = [(100, 150), (200, 150), (200, 250), (100, 250)]  # 模拟的标记点像素坐标
disp_left = np.zeros((480, 640), dtype=np.float32)  # 假定的视差图
disp_right = np.zeros((480, 640), dtype=np.float32)

# 计算绝对坐标系下的物理坐标
pts_world = create_absolute_coordinate_system(left_map1, left_map2, right_map1, right_map2, Q, pts_pix, disp_left,
                                              disp_right)

# 输出计算得到的物理坐标
print("标记点的物理坐标（绝对坐标系下）：")
print(pts_world)

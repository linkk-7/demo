import matplotlib
# use a standard interactive backend:
matplotlib.use('Qt5Agg')      # 需要安装 PyQt5/6
# 或者用非交互式的 Agg：
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 把左右相机历史像素坐标做时间对齐，再结合双目标定参数，把二维像素轨迹恢复成三维坐标轨迹，最后计算并保存该点的三维位移时间历程。

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import cv2
'''
读取一对像素历史坐标计算结果文件，进行特征点三维变形计算
'''

plt.rcParams['axes.unicode_minus'] = False   # 使用 ASCII 减号代替 Unicode 减号

# 读取左右相机的时间戳和历史坐标
# root_path = r"data/2"
root_path = r"new_data5/cab3"

# t_left = np.load(os.path.join(root_path, 't_left_rec.npy'), allow_pickle=True)
# hisl = np.load(os.path.join(root_path, '左相机历史坐标-604.npy'), allow_pickle=True)
# t_right = np.load(os.path.join(root_path, 't_right_rec.npy'), allow_pickle=True)
# hisr = np.load(os.path.join(root_path, '右相机历史坐标-636.npy'), allow_pickle=True)

# ----------以下为修改内容----------
t_left_rel = np.load(os.path.join(root_path, 't_rel_left_rec.npy'), allow_pickle=True)
t_left_abs = np.load(os.path.join(root_path, 't_abs_left_rec.npy'), allow_pickle=True)

t_right_rel = np.load(os.path.join(root_path, 't_rel_right_rec.npy'), allow_pickle=True)
t_right_abs = np.load(os.path.join(root_path, 't_abs_right_rec.npy'), allow_pickle=True)

hisl = np.load(os.path.join(root_path, '左相机历史坐标-354.npy'), allow_pickle=True)  # 左相机历史坐标
hisr = np.load(os.path.join(root_path, '右相机历史坐标-363.npy'), allow_pickle=True)  # 右相机历史坐标
# --------------------------------

print(f"左相机历史坐标数量: {len(hisl)}")
print(f"右相机历史坐标数量: {len(hisr)}")

if len(hisl) == 0 or len(hisr) == 0:
    raise ValueError("左或右相机历史坐标为空！请检查特征点检测/匹配结果，当前点号可能在某相机中未检测到。")

# 把左右数据长度强制统一
# small_num = min(len(t_left),len(t_right),len(hisl), len(hisr))
# print(small_num)
# #针对hisl和hisr步一样长度，强制统一
# t_left = t_left[:small_num]
# hisl = hisl[:small_num]
# t_right = t_right[:small_num]
# hisr = hisr[:small_num]

# ----------以下为修改内容----------
small_num = min(len(t_left_rel), len(t_left_abs), len(t_right_rel), len(t_right_abs), len(hisl), len(hisr))

t_left_rel = t_left_rel[:small_num]
t_left_abs = t_left_abs[:small_num]
t_right_rel = t_right_rel[:small_num]
t_right_abs = t_right_abs[:small_num]
hisl = hisl[:small_num]
hisr = hisr[:small_num]
# --------------------------------


# right的时间与left对齐，并且插值计算hisr
# 第一组数据的横坐标和纵坐标
# x1 = np.array(t_left)
x1 = np.array(t_left_rel)
y1 = hisl
y1_x = y1[:, 0]
y1_y = y1[:, 1]

# 第二组数据的横坐标和纵坐标
# x2 = np.array(t_right)
x2 = np.array(t_right_rel)
y2 = hisr
print("DEBUG y2 shape:", y2.shape)
print("DEBUG y2:", y2)
y2_x = y2[:, 0]
y2_y = y2[:, 1]

# 创建线性插值函数对象，并允许外插值
interp_func_x = interp1d(x2, y2_x, kind='linear', fill_value='extrapolate')
interp_func_y = interp1d(x2, y2_y, kind='linear', fill_value='extrapolate')

# 用第一组数据的横坐标作为新的插值点
y2_x_interpolated = interp_func_x(x1)
y2_y_interpolated = interp_func_y(x1)

# 计算特征点的三维变形
# 相机内外参数
# calibration_path = r"data/cab"
calibration_path = r"new_data5/cab"
R = np.load(os.path.join(calibration_path, 'R.npy'), allow_pickle=True).item()
R = R['rotation']
T = np.load(os.path.join(calibration_path, 'T.npy'), allow_pickle=True).item()  # 两个相机之间的平移矩阵
T = T['trans']
left_intrinsic = np.load(os.path.join(calibration_path, 'calibration_1.npy'), allow_pickle=True).item()
left_intrinsic = left_intrinsic['cameraMatrix']
right_intrinsic = np.load(os.path.join(calibration_path, 'calibration_2.npy'), allow_pickle=True).item()
right_intrinsic = right_intrinsic['cameraMatrix']


# 根据相似三角原理及视差原理计算真实位移
def cal3d3point(R, T, left_intrinsic, right_intrinsic, ul, ur, vl):
    record = []
    # 像素坐标
    # 左相机靶点的像素坐标
    ul = ul
    # 右相机靶点的像素坐标
    ur = ur

    # 图像坐标
    XL1 = (ul - left_intrinsic[0, 2]) / left_intrinsic[0, 0]
    YL1 = (vl - left_intrinsic[1, 2]) / left_intrinsic[1, 1]

    # 计算世界坐标系下的三维坐标，假设世界坐标系与左相机坐标系重合
    fl = (left_intrinsic[0][0] + left_intrinsic[1][1]) / 2
    fr = (right_intrinsic[0][0] + right_intrinsic[1][1]) / 2

    # b = math.sqrt(T[0] ** 2 + T[1] ** 2 + T[2] ** 2)
    # b = np.linalg.norm(T)
    b = abs(float(T[0]))
    d = ur - ul
    Z = fl * b / d
    X = Z * XL1 / fl * left_intrinsic[0, 0]
    Y = Z * YL1 / fl * left_intrinsic[1, 1]
    record = [X, Y, Z]
    print('3dpoint:X:{}(mm),Y:{}(mm),Z:{}(mm)'.format(X, Y, Z))
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    return record, X, Y, Z




def plot_data(t, z, d):
    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 绘制折线图
    plt.plot(t, z, color='b', marker='o', linestyle='-', markersize=2)

    # 设置标题和轴标签
    if d == 'x':
        plt.title('time vs displacement X-direction')
    elif d == 'y':
        plt.title('time vs displacement Y-direction')
    else:
        plt.title('time vs displacement Z-direction')

    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (mm)')

    # 显示网格
    plt.grid(True)

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()


ul = y1_x
ur = y2_x_interpolated
vl = y1_y

record, X, Y, Z = cal3d3point(R, T, left_intrinsic, right_intrinsic, ul, ur, vl)

# result = {}
# result['X'] = X - X[0]
# result['Y'] = -(Y - Y[0])
# result['Z'] = Z - Z[0]
# result['T'] = t_left

# ----------以下为修改内容----------
result = {}
result['X'] = X - X[0]
result['Y'] = -(Y - Y[0])
result['Z'] = Z - Z[0]

# 两套时间都保存
result['T_rel'] = t_left_rel
result['T_abs'] = t_left_abs

# 为了兼容旧代码，也可以暂时保留 T，但建议明确让 T = T_abs
result['T'] = t_left_abs
# --------------------------------

np.save(os.path.join(root_path, '视觉位移计算结果-cal3d-ALIKED.npy'), result)

# time_data = t_left/1000
time_data = t_left_rel / 1000.0
displacement = result
plt.figure(figsize=(12,10))
plt.rcParams['font.family'] = 'SimSun'
# X方向位移
plt.subplot(3, 1, 1)
plt.xlim(time_data.min(), time_data.max())
plt.plot(time_data, displacement['X'], 'r-', linewidth=1.5)
plt.xlabel('时间 (秒)')
plt.ylabel('X方向位移 (mm)')
plt.title('X方向位移-时间曲线')
plt.grid(True)

# Y方向位移
plt.subplot(3, 1, 2)
plt.xlim(time_data.min(), time_data.max())
plt.plot(time_data, displacement['Y'], 'g-', linewidth=1.5)
plt.xlabel('时间 (秒)')
plt.ylabel('Y方向位移 (mm)')
plt.title('Y方向位移-时间曲线')
plt.grid(True)

# Z方向位移
plt.subplot(3, 1, 3)
plt.xlim(time_data.min(), time_data.max())
plt.plot(time_data, displacement['Z'], 'b-', linewidth=1.5)
plt.xlabel('时间 (秒)')
plt.ylabel('Z方向位移 (mm)')
plt.title('Z方向位移-时间曲线')
plt.grid(True)

plt.tight_layout()
plt.show()
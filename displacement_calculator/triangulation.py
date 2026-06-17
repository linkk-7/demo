"""
三维重建算法模块 - 包含视差法和三角测量法两种三维重建算法
"""
import numpy as np
import cv2
import math

def calculate_3d_points_disparity(R, T, left_intrinsic, right_intrinsic, ul, vl, ur):
    """
    使用视差法计算三维坐标 (向量化版本)
    
    参数:
        R, T: 旋转矩阵和平移向量
        left_intrinsic, right_intrinsic: 相机内参矩阵
        ul, vl: 左相机x和y坐标数组
        ur: 右相机x坐标数组
    
    返回:
        三维坐标数组 (N x 3)
    """
    # 左相机内参提取
    fx_left = left_intrinsic[0, 0]
    fy_left = left_intrinsic[1, 1]
    cx_left = left_intrinsic[0, 2]
    cy_left = left_intrinsic[1, 2]
    
    # 计算基线长度b，取平移向量T的模
    b = np.sqrt(T[0]**2 + T[1]**2 + T[2]**2)
    
    # 初始化结果数组
    X = np.zeros_like(ul)
    Y = np.zeros_like(ul)
    Z = np.zeros_like(ul)
    
    # 计算视差
    d = ul - ur
    
    # 过滤有效视差值
    valid_idx = np.abs(d) > 1e-5
    if not np.any(valid_idx):
        print("错误: 所有视差都接近于零，无法计算深度!")
        return np.zeros((len(ul), 3))
    
    # 计算深度Z（使用左相机的x轴焦距）
    Z[valid_idx] = (fx_left * b) / d[valid_idx]
    
    # 转换为归一化坐标
    XL = (ul - cx_left) / fx_left
    YL = (vl - cy_left) / fy_left
    
    # 计算三维坐标
    X[valid_idx] = XL[valid_idx] * Z[valid_idx]
    Y[valid_idx] = YL[valid_idx] * Z[valid_idx]
    
    # 将结果合并为Nx3数组
    points_3d = np.column_stack((X, Y, Z))
    
    return points_3d

def calculate_3d_points_triangulation(R, T, left_intrinsic, right_intrinsic, ul, vl, ur, vr):
    """
    使用三角测量法计算三维坐标
    
    参数:
        R, T: 旋转矩阵和平移向量
        left_intrinsic, right_intrinsic: 相机内参矩阵
        ul, vl: 左相机x和y坐标数组
        ur, vr: 右相机x和y坐标数组
    
    返回:
        三维坐标数组 (N x 3)
    """
    # 准备投影矩阵
    leftRotation = np.eye(3)
    leftTranslation = np.zeros((3, 1))
    mLeft = np.hstack([leftRotation, leftTranslation])
    mLeftM = np.dot(left_intrinsic, mLeft)  # 左相机投影矩阵
    
    rightRotation = R
    rightTranslation = T.reshape(3, 1)  # 确保T是列向量
    mRight = np.hstack([rightRotation, rightTranslation])
    mRightM = np.dot(right_intrinsic, mRight)  # 右相机投影矩阵
    
    # 逐点进行三角测量
    points_3d = []
    for i in range(len(ul)):
        A = np.zeros(shape=(4, 3))
        
        # 构建系数矩阵A
        A[0, :] = ul[i] * mLeftM[2, 0:3] - mLeftM[0, 0:3]
        A[1, :] = vl[i] * mLeftM[2, 0:3] - mLeftM[1, 0:3]
        A[2, :] = ur[i] * mRightM[2, 0:3] - mRightM[0, 0:3]
        A[3, :] = vr[i] * mRightM[2, 0:3] - mRightM[1, 0:3]
        
        # 构建常数项B
        B = np.zeros(shape=(4, 1))
        B[0, 0] = mLeftM[0, 3] - ul[i] * mLeftM[2, 3]
        B[1, 0] = mLeftM[1, 3] - vl[i] * mLeftM[2, 3]
        B[2, 0] = mRightM[0, 3] - ur[i] * mRightM[2, 3]
        B[3, 0] = mRightM[1, 3] - vr[i] * mRightM[2, 3]
        
        # 求解线性方程组
        try:
            retval, XYZ = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
            if not retval:
                XYZ = np.array([np.nan, np.nan, np.nan]).reshape(3, 1)
                print(f"警告: 点 {i} 的求解失败")
        except cv2.error as e:
            XYZ = np.array([np.nan, np.nan, np.nan]).reshape(3, 1)
            print(f"警告: 点 {i} 求解出错: {e}")
        
        points_3d.append(XYZ.flatten())
    
    return np.array(points_3d)

def calculate_displacement(points_3d):
    """
    计算相对于初始位置的位移
    
    参数:
        points_3d: 三维坐标数组 (N x 3)
    
    返回:
        X, Y, Z方向的位移数组
    """
    # 移除包含NaN的点
    valid_indices = ~np.isnan(points_3d[:, 0])
    if not np.any(valid_indices):
        print("错误: 没有有效的三维点!")
        return None, None, None
    
    # 选择第一个有效点作为参考
    ref_idx = np.where(valid_indices)[0][0]
    ref_point = points_3d[ref_idx]
    
    # 计算位移
    disp_x = -(points_3d[:, 0] - ref_point[0])  # X方向位移
    disp_y = -(points_3d[:, 1] - ref_point[1])  # Y方向位移
    disp_z = -(points_3d[:, 2] - ref_point[2])  # Z方向位移
    
    return disp_x, disp_y, disp_z

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import cv2
'''
读取一对像素历史坐标计算结果文件，进行特征点三维变形计算
'''
# 用一对左右图像中的同名像素点，结合双目标定参数，计算这个点的三维空间坐标，用来验证你的标定结果是否合理。


# 计算特征点的三维变形
# 相机内外参数
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

    #b = math.sqrt(T[0] ** 2 + T[1] ** 2 + T[2] ** 2)
    # b = math.sqrt(float(T[0]) ** 2 + float(T[1]) ** 2 + float(T[2]) ** 2)
    b = np.linalg.norm(T)
    # d = ur - ul
    d = ul - ur
    print('d=',d)
    Z = fl * b / d
    X = Z * XL1 / fl * left_intrinsic[0, 0]
    Y = Z * YL1 / fl * left_intrinsic[1, 1]
    record = [X, Y, Z]
    print('3dpoint:X:{}(mm),Y:{}(mm),Z:{}(mm)'.format(X, Y, Z))
    return record


#另一种位移计算公式
# 函数参数为左右相片同名点的像素坐标，获取方式后面介绍
# lx，ly为左相机某点像素坐标，rx，ry为右相机对应点像素坐标
def uvToXYZ(R,T,left_intrinsic,right_intrinsic,ul, vl, ur, vr):
    leftRotation = np.eye(3)
    leftTranslation = np.zeros((3,1))
    mLeft = np.hstack([leftRotation, leftTranslation])
    mLeftM = np.dot(left_intrinsic, mLeft)
    rightRotation = R
    rightTranslation = T
    mRight = np.hstack([rightRotation, rightTranslation])
    mRightM = np.dot(right_intrinsic, mRight)
    
    A = np.zeros(shape=(4, 3))
    for i in range(0, 3):
        A[0][i] = ul * mLeftM[2, i] - mLeftM[0][i]
    for i in range(0, 3):
        A[1][i] = vl * mLeftM[2, i] - mLeftM[1][i]
    for i in range(0, 3):
        A[2][i] = ur * mRightM[2, i] - mRightM[0][i]
    for i in range(0, 3):
        A[3][i] = vr * mRightM[2, i] - mRightM[1][i]
    
    B = np.zeros(shape=(4, 1))
    for i in range(0, 2):
        B[i][0] = mLeftM[i][3] - ul * mLeftM[2][3]
    for i in range(2, 4):
        B[i][0] = mRightM[i - 2][3] - ur * mRightM[2][3]
    
    XYZ = np.zeros(shape=(3, 1))
    # 根据大佬的方法，采用最小二乘法求其空间坐标
    cv2.solve(A, B, XYZ, cv2.DECOMP_SVD)
    print(XYZ)
    
    return XYZ

def cal3d_paper_formula(T, left_intrinsic, right_intrinsic, ul, ur, vl):
    """
    方法2：论文公式的统一写法
    用有符号坐标统一三种情况，不再人为分“左/中/右”
    公式：
        x_l = u_l - cxl
        x_r = u_r - cxr
        Z = B * f_l * f_r / (f_r * x_l - f_l * x_r)
    然后：
        X = Z * x_l / f_l
        Y = Z * y_l / f_y_l
    注意：
    1）这里的 B 仍取有效水平基线 |Tx|
    2）x_l, x_r 必须是相对于主点的有符号坐标
    """
    ul = np.asarray(ul, dtype=np.float64)
    ur = np.asarray(ur, dtype=np.float64)
    vl = np.asarray(vl, dtype=np.float64)

    fx_l = left_intrinsic[0, 0]
    fy_l = left_intrinsic[1, 1]
    cx_l = left_intrinsic[0, 2]
    cy_l = left_intrinsic[1, 2]

    fx_r = right_intrinsic[0, 0]
    cx_r = right_intrinsic[0, 2]

    # B = float(T[0, 0])
    # B = np.linalg.norm(T)

    x_l = ul - cx_l
    x_r = ur - cx_r
    y_l = vl - cy_l

    denom = fx_r * x_l - fx_l * x_r
    Z = safe_divide(B * fx_l * fx_r, denom)
    X = Z * x_l / fx_l
    Y = Z * y_l / fy_l
    print('3dpoint:X:{}(mm),Y:{}(mm),Z:{}(mm)'.format(X, Y, Z))

    return X, Y, Z

def safe_divide(num, den, eps=1e-9):
    den_safe = np.where(np.abs(den) < eps, np.nan, den)
    return num / den_safe


hisl = np.load(r"new_data5\cab1\左相机历史坐标-604.npy", allow_pickle=True)
hisr = np.load(r"new_data5\cab1\右相机历史坐标-636.npy", allow_pickle=True)

frame_id = 200
ul, vl = hisl[frame_id]
ur, vr = hisr[frame_id]

record = cal3d3point(R, T, left_intrinsic, right_intrinsic, ul, ur, vl)#ul, ur, vl
# record2 = uvToXYZ(R, T, left_intrinsic, right_intrinsic, ul, ur, vl, vr)#ul, ur, vl, vr
# record = cal3d_paper_formula(T, left_intrinsic, right_intrinsic, ul, ur, vl)

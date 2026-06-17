import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def camera_calibration1(left_image_file,right_image_file,pattern_size,size,save_path):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 准备左右相机的图像文件
    left_image_files = os.listdir(left_image_file)  # 左相机图像文件列表
    right_image_files = os.listdir(right_image_file)  # 右相机图像文件列表

    # 创建棋盘格角点的坐标
    object_points = []  # 3D物体点的坐标
    left_image_points = []  # 左相机的2D图像点的坐标
    right_image_points = []  # 右相机的2D图像点的坐标

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)*size
    for left_file, right_file in zip(left_image_files, right_image_files):
        print(left_file, right_file)
        left_file = os.path.join(left_image_file,left_file)
        right_file = os.path.join(right_image_file,right_file)
        left_img = cv2.imread(left_file,0)
        right_img = cv2.imread(right_file,0)

        gray_left = left_img
        gray_right = right_img

        # 尝试在图像中查找棋盘格角点
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)

        if ret_left and ret_right:
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (52, 52), (-1, -1), criteria)
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (52, 52), (-1, -1), criteria)
            object_points.append(objp)
            left_image_points.append(corners2_left)
            right_image_points.append(corners2_right)

        else:
            print('未检测到棋盘格角点！')
    #分别对左右相机进行单独标定得到内参矩阵
    ret1,left_K,dist_l,r1,t1 = cv2.calibrateCamera(object_points, left_image_points, gray_left.shape[::-1], None, None)
    print('左相机的内部参数矩阵(left_K):',left_K)
    ret2, right_K, dist_r, r2, t2 = cv2.calibrateCamera(object_points, right_image_points, gray_right.shape[::-1], None, None)
    print('右相机的内部参数矩阵(right_K):', right_K)

    #优化内参矩阵
    hl,wl = gray_left.shape[:2]
    left_K,roil = cv2.getOptimalNewCameraMatrix(left_K, dist_l, (wl, hl), 1, (wl, hl))

    hr,wr = gray_right.shape[:2]
    right_K,roir = cv2.getOptimalNewCameraMatrix(right_K, dist_r, (wr, hr), 1, (wr, hr))
    # 进行双目相机标定
    # R = None  # 旋转矩阵
    # T = None  # 平移向量
    # E = None  # 本质矩阵
    # F = None  # 基本矩阵

    # 进行相机标定和立体标定
    flags = 0
    # flags |= cv2.CALIB_FIX_INTRINSIC


    ret, left_K, dist_coeff_left, right_K, dist_coeff_right, R, T, E, F = cv2.stereoCalibrate(
        object_points, left_image_points, right_image_points, left_K, dist_l, right_K, dist_r,
        gray_left.shape[::-1], criteria=criteria_stereo, flags=flags
    )

    # 输出旋转矩阵(R)和平移向量(T)
    print("右相机相对于左相机的旋转矩阵 (R_left):")
    print(R)
    print("\n右相机相对于左相机的平移向量 (T_left):")
    print(T)
    print('左相机的内部参数矩阵(left_K):')
    print(left_K)
    print('右相机的内部参数矩阵(right_K):')
    print(right_K)
    print('左相机的畸变系数【径向2+切向2+径向1】(dist_l):')
    print(dist_l)
    print('左相机的畸变系数【径向2+切向2+径向1】(dist_r):')
    print(dist_r)
    print("标定误差 (reprojection error):", ret)
    left = {'cameraMatrix':left_K,'distCoeffs':dist_coeff_left}
    right = {'cameraMatrix': right_K, 'distCoeffs': dist_coeff_right}
    r = {'rotation':R}
    t = {'trans':T}

    np.save(os.path.join(save_path,'calibration_1.npy'),left)
    np.save(os.path.join(save_path,'calibration_2.npy'),right)
    np.save(os.path.join(save_path,'R.npy'),r)
    np.save(os.path.join(save_path,'T.npy'),t)
    return left_K,dist_coeff_left,right_K,dist_coeff_right,R,T


def camera_calibration(left_image_file, right_image_file, pattern_size, size, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    left_image_files = os.listdir(left_image_file)
    right_image_files = os.listdir(right_image_file)

    object_points = []
    left_image_points = []
    right_image_points = []

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * size

    # 使用第一张图获取图像尺寸
    first_left_file = os.path.join(left_image_file, left_image_files[0])
    first_right_file = os.path.join(right_image_file, right_image_files[0])
    left_img = cv2.imread(first_left_file)
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    img_size = gray_left.shape[::-1]

    for left_file, right_file in zip(left_image_files, right_image_files):
        left_file = os.path.join(left_image_file, left_file)
        right_file = os.path.join(right_image_file, right_file)
        left_img = cv2.imread(left_file)
        right_img = cv2.imread(right_file)
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)

        if ret_left and ret_right:
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            object_points.append(objp)
            left_image_points.append(corners2_left)
            right_image_points.append(corners2_right)

    ret1, left_K, dist_l, _, _ = cv2.calibrateCamera(object_points, left_image_points, img_size, None, None)
    ret2, right_K, dist_r, _, _ = cv2.calibrateCamera(object_points, right_image_points, img_size, None, None)

    # 立体标定
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, left_K_optimized, dist_coeff_left, right_K_optimized, dist_coeff_right, R, T, E, F = cv2.stereoCalibrate(
        object_points, left_image_points, right_image_points, left_K, dist_l, right_K, dist_r,
        img_size, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
    )

    if ret:
        print("左相机相对于右相机的旋转矩阵 (R_right):")
        print(R)
        print("\n左相机相对于右相机的平移向量 (T_right):")
        print(T)
        print('左相机的内部参数矩阵(left_K):')
        print(left_K_optimized)
        print('右相机的内部参数矩阵(right_K):')
        print(right_K_optimized)
        print('左相机的畸变系数【径向+切向】(dist_l):')
        print(dist_coeff_left)
        print('右相机的畸变系数【径向+切向】(dist_r):')
        print(dist_coeff_right)
        print("标定误差 (reprojection error):", ret)

        left = {'cameraMatrix': left_K_optimized, 'distCoeffs': dist_coeff_left}
        right = {'cameraMatrix': right_K_optimized, 'distCoeffs': dist_coeff_right}
        r = {'rotation': R}
        t = {'trans': T}

        np.save(os.path.join(save_path, 'calibration_1.npy'), left)
        np.save(os.path.join(save_path, 'calibration_2.npy'), right)
        np.save(os.path.join(save_path, 'R.npy'), r)
        np.save(os.path.join(save_path, 'T.npy'), t)
        return left_K_optimized, dist_coeff_left, right_K_optimized, dist_coeff_right, R, T
    else:
        print("立体标定失败，请检查输入图像或调整标定参数。")
        return None, None, None, None, None, None
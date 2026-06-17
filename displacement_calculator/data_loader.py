"""
数据加载和预处理模块 - 负责加载时间序列和坐标数据，进行时间对齐
"""
import os
import numpy as np
from scipy.interpolate import interp1d

def load_calibration_data(calibration_path):
    """加载相机标定参数"""
    try:
        R = np.load(os.path.join(calibration_path, 'R.npy'), allow_pickle=True).item()['rotation']
        T = np.load(os.path.join(calibration_path, 'T.npy'), allow_pickle=True).item()['trans']
        left_intrinsic = np.load(os.path.join(calibration_path, 'calibration_1.npy'), 
                                allow_pickle=True).item()['cameraMatrix']
        right_intrinsic = np.load(os.path.join(calibration_path, 'calibration_2.npy'), 
                                 allow_pickle=True).item()['cameraMatrix']
        return R, T, left_intrinsic, right_intrinsic
    except FileNotFoundError as e:
        print(f"错误: 标定文件未找到: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"加载标定参数时出错: {e}")
        return None, None, None, None

def load_trajectory_data(t_left_path, hisl_path, t_right_path, hisr_path):
    """加载轨迹数据并进行验证"""

    # 添加异常处理
    try:
        t_left = np.load(t_left_path, allow_pickle=True)
        hisl = np.load(hisl_path, allow_pickle=True)
        t_right = np.load(t_right_path, allow_pickle=True)
        hisr = np.load(hisr_path, allow_pickle=True)
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None, None, None, None
    
    # 检查数据格式
    if not isinstance(hisl, np.ndarray) or hisl.shape[1] != 2:
        print(f"错误: hisl.npy 格式不正确, 应为 N x 2 的数组。"
              f"当前形状: {hisl.shape if isinstance(hisl, np.ndarray) else type(hisl)}")
        return None, None, None, None
    
    if not isinstance(hisr, np.ndarray) or hisr.shape[1] != 2:
        print(f"错误: hisr.npy 格式不正确, 应为 N x 2 的数组。"
              f"当前形状: {hisr.shape if isinstance(hisr, np.ndarray) else type(hisr)}")
        return None, None, None, None
    
    print('左时间点长度:', len(t_left), '右时间点长度:', len(t_right), 
          '左历史坐标长度:', len(hisl), '右历史坐标长度:', len(hisr))
    
    return t_left, hisl, t_right, hisr

def align_time_series(t_left, hisl, t_right, hisr):
    """时间序列对齐和插值处理"""
    # 截断到最短长度
    small_num = min(len(t_left), len(t_right), len(hisl), len(hisr))
    print(f"数据将被截断为最短长度: {small_num}")
    
    t_left = t_left[:small_num]
    hisl = hisl[:small_num]
    t_right = t_right[:small_num]
    hisr = hisr[:small_num]
    
    # 准备数据
    x1 = np.array(t_left)  # 基准时间
    y1 = hisl  # 左相机坐标
    y1_x = y1[:, 0]  # 左相机 x 坐标
    y1_y = y1[:, 1]  # 左相机 y 坐标
    
    x2 = np.array(t_right)  # 右相机时间
    y2 = hisr  # 右相机坐标
    y2_x = y2[:, 0]  # 右相机 x 坐标
    y2_y = y2[:, 1]  # 右相机 y 坐标
    
    # 检查时间序列单调性
    if not np.all(np.diff(x1) >= 0) or not np.all(np.diff(x2) >= 0):
        print("警告: 时间序列不是单调递增的，插值结果可能不可靠。")
    
    try:
        # 创建插值函数
        interp_func_x = interp1d(x2, y2_x, kind='linear', fill_value='extrapolate', bounds_error=False)
        interp_func_y = interp1d(x2, y2_y, kind='linear', fill_value='extrapolate', bounds_error=False)
        
        # 执行插值
        y2_x_interpolated = interp_func_x(x1)
        y2_y_interpolated = interp_func_y(x1)
        y2_interpolated = np.vstack((y2_x_interpolated, y2_y_interpolated)).T
        
        return t_left, y1_x, y1_y, y2_x_interpolated, y2_y_interpolated
    
    except ValueError as e:
        print(f"创建插值函数时出错: {e}. 请检查时间序列 x2 是否包含重复值或少于2个点。")
        return None, None, None, None, None

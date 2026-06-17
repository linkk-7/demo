"""
主程序入口 - 集成数据加载、处理和可视化
"""
import os
import sys
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# 设置路径
try:
    from settings import ROOT_PATH, CALIBRATION_PATH, CALIBRATION_RESULT_PATH
except ImportError:
    print("警告: settings.py 未找到或未定义所需路径，使用默认值")
    ROOT_PATH = '.'
    CALIBRATION_RESULT_PATH = '.'
from displacement_calculator.data_loader import load_calibration_data, load_trajectory_data, align_time_series
from displacement_calculator.triangulation import calculate_3d_points_triangulation, calculate_displacement
from displacement_calculator.visualization import plot_displacement, save_results

def run_displacement_calculation_with_params(t_left_path, hisl_path, t_right_path, hisr_path, callback=None):
    """运行位移计算的主函数（从GUI调用）
    
    参数:
        t_left_path (str): 左相机时间文件路径
        hisl_path (str): 左相机轨迹文件路径
        t_right_path (str): 右相机时间文件路径
        hisr_path (str): 右相机轨迹文件路径
        callback (callable, optional): 进度回调函数
    
    返回:
        dict: 计算结果
    """
    # 步骤2: 加载数据
    if callback:
        callback("正在加载轨迹数据...")
    else:
        print("\n正在加载轨迹数据...")
    
    data = load_trajectory_data(t_left_path, hisl_path, t_right_path, hisr_path)
    print(data)
    
    # 检查数据是否为None
    if data is None or any(x is None for x in data):
        message = "加载数据失败，程序退出"
        if callback:
            callback(message)
        else:
            print(message)
        return None
    
    t_left, hisl, t_right, hisr = data
    
    # 步骤3: 时间对齐
    if callback:
        callback("正在进行时间对齐...")
    else:
        print("\n正在进行时间对齐...")
    
    aligned_data = align_time_series(t_left, hisl, t_right, hisr)
    
    # 检查数据是否为None
    if aligned_data is None or any(x is None for x in aligned_data):
        message = "时间对齐失败，程序退出"
        if callback:
            callback(message)
        else:
            print(message)
        return None
    
    t_left, ul, vl, ur, vr = aligned_data
    
    # 步骤4: 加载标定参数
    if callback:
        callback("正在加载相机标定参数...")
    else:
        print("\n正在加载相机标定参数...")
    
    camera_params = load_calibration_data(CALIBRATION_RESULT_PATH)
    
    # 检查数据是否为None
    if camera_params is None or any(x is None for x in camera_params):
        message = "加载标定参数失败，程序退出"
        if callback:
            callback(message)
        else:
            print(message)
        return None
    
    R, T, left_intrinsic, right_intrinsic = camera_params
    
    # 步骤5: 三维重建
    if callback:
        callback("正在计算三维坐标...")
    else:
        print("\n正在计算三维坐标...")
        print("使用三角测量法进行三维重建")
    
    points_3d = calculate_3d_points_triangulation(
        R, T, left_intrinsic, right_intrinsic, ul, vl, ur, vr)
    
    # 步骤6: 计算位移
    if callback:
        callback("正在计算位移...")
    else:
        print("\n正在计算位移...")
    
    disp_x, disp_y, disp_z = calculate_displacement(points_3d)
    if disp_x is None:
        message = "位移计算失败，程序退出"
        if callback:
            callback(message)
        else:
            print(message)
        return None
    
    #时间转换为正确的秒为单位，当前时间为毫秒
    t_left = t_left
    # 步骤7: 保存结果
    results = {
        'points_3d': points_3d,
        'X': disp_x,
        'Y': disp_y,
        'Z': disp_z,
        'T': t_left
    }
    
    save_results(results, '视觉位移计算结果-uvToXYZ.npy', ROOT_PATH)
    
    # 步骤8: 可视化
    if callback:
        callback("正在生成位移图...")
    else:
        print("\n正在生成位移图...")
    
    plot_displacement(t_left, disp_z, 'z', ROOT_PATH)
    plot_displacement(t_left, disp_y, 'y', ROOT_PATH)
    plot_displacement(t_left, disp_x, 'x', ROOT_PATH)
    
    if callback:
        callback("位移计算完成！")
    else:
        print("\n位移计算完成！")
    
    return results

def run_displacement_calculation():
    """运行位移计算的主函数（命令行调用）"""
    print("======= 特征点三维位移计算 =======")
    
    # 步骤1: 获取文件路径
    t_left_path = input("请输入t_left.npy的路径(直接回车使用默认路径): ") or os.path.join(ROOT_PATH, 't_left_rec.npy')
    hisl_path = input("请输入hisl.npy的路径(直接回车使用默认路径): ") or os.path.join(ROOT_PATH, '左相机历史坐标.npy')
    t_right_path = input("请输入t_right.npy的路径(直接回车使用默认路径): ") or os.path.join(ROOT_PATH, 't_right_rec.npy')
    hisr_path = input("请输入hisr.npy的路径(直接回车使用默认路径): ") or os.path.join(ROOT_PATH, '右相机历史坐标.npy')
    
    return run_displacement_calculation_with_params(t_left_path, hisl_path, t_right_path, hisr_path)


"""
数据可视化模块 - 绘制位移-时间图表
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_displacement(t, data, direction, save_path=None, max_points=1000):
    """
    绘制位移-时间图
    
    参数:
        t: 时间数组（毫秒）
        data: 位移数据数组
        direction: 方向标识 ('x', 'y', 或 'z')
        save_path: 保存路径，如果为None则不保存
        max_points: 最大绘制点数，防止图形过于复杂
    """
    # 过滤无效值
    valid_indices = ~np.isnan(data)
    if not np.any(valid_indices):
        print(f"警告: {direction} 方向没有有效数据可绘制")
        return
    
    t_valid = t[valid_indices]
    data_valid = data[valid_indices]
    
    # 限制绘制点数量
    num_points = min(len(t_valid), max_points)
    t_plot = t_valid[:num_points] / 1000.0  # 转换为秒
    data_plot = data_valid[:num_points]
    
    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, data_plot, color='b', marker='o', linestyle='-', markersize=2)
    
    # 设置标题和标签
    title_map = {
        'x': 'X方向位移随时间变化',
        'y': 'Y方向位移随时间变化',
        'z': 'Z方向位移随时间变化'
    }
    plt.title(title_map.get(direction, f'{direction}方向位移随时间变化'))
    plt.xlabel('时间 (秒)')
    plt.ylabel('位移 (毫米)')
    plt.grid(True)
    
    # 保存图形
    if save_path:
        filename = os.path.join(save_path, f'{direction}_displacement.png')
        plt.savefig(filename, dpi=300)
        print(f"图像已保存到: {filename}")
    
    # 显示图形
    plt.show()

def save_results(displacement_data, filename, save_path):
    """
    保存位移计算结果
    
    参数:
        displacement_data: 位移数据字典
        filename: 文件名
        save_path: 保存路径
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    full_path = os.path.join(save_path, filename)
    np.save(full_path, displacement_data)
    print(f"位移数据已保存到: {full_path}")

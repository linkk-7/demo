#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
三维坐标历史轨迹计算与可视化工具
根据左右相机的历史像素坐标计算三维空间轨迹
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
import math
from scipy.interpolate import interp1d
from tqdm import tqdm
import json

class StereoTracker3D:
    """立体视觉三维轨迹计算与可视化类"""
    
    def __init__(self, calibration_path):
        """
        初始化三维追踪器
        
        参数:
        calibration_path: 相机标定参数文件夹路径
        """
        # 加载相机标定参数
        self.load_calibration(calibration_path)
        
        # 初始化变量
        self.left_tracks = {}   # 左相机轨迹数据
        self.right_tracks = {}  # 右相机轨迹数据
        self.tracks_3d = {}     # 三维轨迹数据
    
    def load_calibration(self, calibration_path):
        """
        加载相机标定参数
        
        参数:
        calibration_path: 相机标定参数文件夹路径
        """
        try:
            # 加载相机旋转矩阵
            R_data = np.load(os.path.join(calibration_path, 'R.npy'), allow_pickle=True).item()
            self.R = R_data['rotation']
            
            # 加载相机平移向量
            T_data = np.load(os.path.join(calibration_path, 'T.npy'), allow_pickle=True).item()
            self.T = T_data['trans']
            
            # 加载左相机内参
            left_data = np.load(os.path.join(calibration_path, 'calibration_1.npy'), allow_pickle=True).item()
            self.left_intrinsic = left_data['cameraMatrix']
            
            # 加载右相机内参
            right_data = np.load(os.path.join(calibration_path, 'calibration_2.npy'), allow_pickle=True).item()
            self.right_intrinsic = right_data['cameraMatrix']
            
            print("相机标定参数加载成功")
        except Exception as e:
            raise ValueError(f"无法加载相机标定参数: {e}")
    
    def load_tracks_csv(self, left_csv, right_csv):
        """
        从CSV文件加载左右相机的轨迹数据
        
        参数:
        left_csv: 左相机轨迹CSV文件路径
        right_csv: 右相机轨迹CSV文件路径
        """
        # 读取左相机CSV文件
        left_df = pd.read_csv(left_csv)
        
        # 读取右相机CSV文件
        right_df = pd.read_csv(right_csv)
        
        # 将左相机数据转换为字典格式: {point_id: {frame: (x, y)}}
        for _, row in left_df.iterrows():
            point_id = row['point_id']
            frame_idx = row['frame_index']
            x, y = row['x'], row['y']
            
            if point_id not in self.left_tracks:
                self.left_tracks[point_id] = {}
            
            self.left_tracks[point_id][frame_idx] = (x, y)
        
        # 将右相机数据转换为字典格式: {point_id: {frame: (x, y)}}
        for _, row in right_df.iterrows():
            point_id = row['point_id']
            frame_idx = row['frame_index']
            x, y = row['x'], row['y']
            
            if point_id not in self.right_tracks:
                self.right_tracks[point_id] = {}
            
            self.right_tracks[point_id][frame_idx] = (x, y)
        
        print(f"已加载左相机轨迹 {len(self.left_tracks)} 个点，右相机轨迹 {len(self.right_tracks)} 个点")
    
    def calculate_3d_tracks(self, point_ids=None):
        """
        计算三维轨迹
        
        参数:
        point_ids: 要计算的点ID列表，None表示计算所有点
        """
        # 如果未指定点ID，则使用左相机和右相机共有的点ID
        if point_ids is None:
            left_ids = set(self.left_tracks.keys())
            right_ids = set(self.right_tracks.keys())
            point_ids = left_ids.intersection(right_ids)
        
        print(f"开始计算 {len(point_ids)} 个点的三维轨迹...")
        
        # 计算每个点的三维轨迹
        for point_id in tqdm(point_ids):
            # 检查点是否在左右相机轨迹中都存在
            if point_id not in self.left_tracks or point_id not in self.right_tracks:
                print(f"警告: 点ID {point_id} 在左右相机轨迹中不完全存在，跳过")
                continue
            
            left_frames = set(self.left_tracks[point_id].keys())
            right_frames = set(self.right_tracks[point_id].keys())
            common_frames = left_frames.intersection(right_frames)
            
            if len(common_frames) == 0:
                print(f"警告: 点ID {point_id} 在左右相机中没有共同帧，跳过")
                continue
            
            # 初始化该点的三维轨迹
            self.tracks_3d[point_id] = {}
            
            # 对每个共同帧计算三维坐标
            for frame in common_frames:
                # 获取左右相机的像素坐标
                ul, vl = self.left_tracks[point_id][frame]
                ur, _ = self.right_tracks[point_id][frame]
                
                # 计算三维坐标
                X, Y, Z = self.calculate_3d_point(ul, ur, vl)
                
                # 存储三维坐标
                self.tracks_3d[point_id][frame] = (X, Y, Z)
            
            print(f"点ID {point_id} 的三维轨迹计算完成，共 {len(self.tracks_3d[point_id])} 帧")
    
    def calculate_3d_point(self, ul, ur, vl):
        """
        计算单个点的三维坐标
        
        参数:
        ul: 左相机x坐标
        ur: 右相机x坐标
        vl: 左相机y坐标
        
        返回:
        (X, Y, Z): 三维坐标，单位为毫米
        """
        # 图像坐标
        XL1 = (ul - self.left_intrinsic[0, 2]) / self.left_intrinsic[0, 0]
        YL1 = (vl - self.left_intrinsic[1, 2]) / self.left_intrinsic[1, 1]
        
        # 计算世界坐标系下的三维坐标，假设世界坐标系与左相机坐标系重合
        fl = (self.left_intrinsic[0][0] + self.left_intrinsic[1][1]) / 2
        fr = (self.right_intrinsic[0][0] + self.right_intrinsic[1][1]) / 2
        
        # 计算基线长度
        b = math.sqrt(self.T[0] ** 2 + self.T[1] ** 2 + self.T[2] ** 2)
        
        # 计算视差
        d = ul - ur  # 注意：这里假设ur小于ul，如果不是，可能需要取绝对值或调整计算方式
        
        # 避免视差为0导致的除法错误
        if abs(d) < 1e-6:
            print(f"警告: 视差接近于0 (d={d})，可能导致不稳定的三维坐标计算")
            d = 1e-6 if d >= 0 else -1e-6
            
        # 计算深度
        Z = fl * b / d
        
        # 计算X和Y
        X = Z * XL1 / fl * self.left_intrinsic[0, 0]
        Y = Z * YL1 / fl * self.left_intrinsic[1, 1]
        
        return X, Y, Z
    
    def visualize_tracks_3d(self, output_path=None, selected_points=None):
        """
        可视化三维轨迹
        
        参数:
        output_path: 保存图像的路径，None表示不保存
        selected_points: 要可视化的点ID列表，None表示显示所有点
        """
        if len(self.tracks_3d) == 0:
            print("错误: 没有可视化的三维轨迹数据，请先计算三维轨迹")
            return
            
        # 如果未指定点ID，则使用所有已计算的点
        if selected_points is None:
            selected_points = list(self.tracks_3d.keys())
            
        # 创建3D图表
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 定义不同点的颜色
        colors = plt.cm.jet(np.linspace(0, 1, len(selected_points)))
        
        # 绘制每个点的轨迹
        for i, point_id in enumerate(selected_points):
            if point_id not in self.tracks_3d:
                print(f"警告: 点ID {point_id} 没有三维轨迹数据，跳过")
                continue
                
            # 获取排序后的帧索引
            frames = sorted(self.tracks_3d[point_id].keys())
            
            # 提取X, Y, Z坐标
            X = [self.tracks_3d[point_id][f][0] for f in frames]
            Y = [self.tracks_3d[point_id][f][1] for f in frames]
            Z = [self.tracks_3d[point_id][f][2] for f in frames]
            
            # 绘制轨迹，使用渐变颜色
            ax.plot(X, Y, Z, 'o-', label=f'ID {point_id}', color=colors[i], alpha=0.7)
            
            # 标记起点和终点
            ax.scatter(X[0], Y[0], Z[0], color='green', s=100, marker='o')
            ax.scatter(X[-1], Y[-1], Z[-1], color='red', s=100, marker='x')
            
        # 设置坐标轴标签
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # 设置图表标题
        ax.set_title('特征点三维轨迹')
        
        # 添加图例
        ax.legend()
        
        # 设置坐标轴比例相等
        max_range = max([
            max([max(X) - min(X) for point_id in selected_points if point_id in self.tracks_3d]),
            max([max(Y) - min(Y) for point_id in selected_points if point_id in self.tracks_3d]),
            max([max(Z) - min(Z) for point_id in selected_points if point_id in self.tracks_3d])
        ])
        mid_x = np.mean([np.mean([self.tracks_3d[point_id][f][0] for f in self.tracks_3d[point_id].keys()]) for point_id in selected_points if point_id in self.tracks_3d])
        mid_y = np.mean([np.mean([self.tracks_3d[point_id][f][1] for f in self.tracks_3d[point_id].keys()]) for point_id in selected_points if point_id in self.tracks_3d])
        mid_z = np.mean([np.mean([self.tracks_3d[point_id][f][2] for f in self.tracks_3d[point_id].keys()]) for point_id in selected_points if point_id in self.tracks_3d])
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # 添加网格
        ax.grid(True)
        
        # 如果指定了输出路径，保存图像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"三维轨迹图已保存到: {output_path}")
            
        # 显示图表
        plt.show()
    
    def visualize_coordinate_changes(self, output_dir=None, selected_points=None):
        """
        可视化每个坐标分量随时间变化的曲线
        
        参数:
        output_dir: 保存图像的目录，None表示不保存
        selected_points: 要可视化的点ID列表，None表示显示所有点
        """
        if len(self.tracks_3d) == 0:
            print("错误: 没有可视化的三维轨迹数据，请先计算三维轨迹")
            return
            
        # 如果未指定点ID，则使用所有已计算的点
        if selected_points is None:
            selected_points = list(self.tracks_3d.keys())
            
        # 创建输出目录
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 对每个点生成坐标变化图
        for point_id in selected_points:
            if point_id not in self.tracks_3d:
                print(f"警告: 点ID {point_id} 没有三维轨迹数据，跳过")
                continue
                
            # 获取排序后的帧索引
            frames = sorted(self.tracks_3d[point_id].keys())
            
            # 提取X, Y, Z坐标
            X = [self.tracks_3d[point_id][f][0] for f in frames]
            Y = [self.tracks_3d[point_id][f][1] for f in frames]
            Z = [self.tracks_3d[point_id][f][2] for f in frames]
            
            # 创建图表
            fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
            
            # 绘制X坐标变化
            axs[0].plot(frames, X, 'r-', marker='o')
            axs[0].set_ylabel('X坐标 (mm)')
            axs[0].set_title(f'点ID {point_id} 的X坐标随帧变化')
            axs[0].grid(True)
            
            # 绘制Y坐标变化
            axs[1].plot(frames, Y, 'g-', marker='o')
            axs[1].set_ylabel('Y坐标 (mm)')
            axs[1].set_title(f'点ID {point_id} 的Y坐标随帧变化')
            axs[1].grid(True)
            
            # 绘制Z坐标变化
            axs[2].plot(frames, Z, 'b-', marker='o')
            axs[2].set_xlabel('帧索引')
            axs[2].set_ylabel('Z坐标 (mm)')
            axs[2].set_title(f'点ID {point_id} 的Z坐标随帧变化')
            axs[2].grid(True)
            
            plt.tight_layout()
            
            # 如果指定了输出目录，保存图像
            if output_dir:
                output_path = os.path.join(output_dir, f'point_{point_id}_coordinate_changes.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"点ID {point_id} 的坐标变化图已保存到: {output_path}")
            
            # 显示图表
            plt.show()
    
    def save_tracks_3d(self, output_path, format='csv'):
        """
        保存三维轨迹数据到文件
        
        参数:
        output_path: 输出文件路径
        format: 输出格式，'csv'或'json'
        """
        if len(self.tracks_3d) == 0:
            print("错误: 没有三维轨迹数据可保存，请先计算三维轨迹")
            return
            
        if format.lower() == 'csv':
            # 创建DataFrame
            data = []
            for point_id, frames in self.tracks_3d.items():
                for frame, (X, Y, Z) in frames.items():
                    data.append({
                        'point_id': point_id,
                        'frame_index': frame,
                        'X': X,
                        'Y': Y,
                        'Z': Z
                    })
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            print(f"三维轨迹数据已保存到CSV文件: {output_path}")
            
        elif format.lower() == 'json':
            # 转换为JSON可序列化格式
            json_data = {}
            for point_id, frames in self.tracks_3d.items():
                json_data[str(point_id)] = {
                    str(frame): {'X': float(X), 'Y': float(Y), 'Z': float(Z)}
                    for frame, (X, Y, Z) in frames.items()
                }
            
            # 保存为JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"三维轨迹数据已保存到JSON文件: {output_path}")
            
        else:
            raise ValueError(f"不支持的输出格式: {format}")


def main():
    """主函数示例"""
    # 创建立体追踪器
    calibration_path = r"E:\test-qingxie\camera_calibration"
    tracker = StereoTracker3D(calibration_path)
    
    # 左右相机轨迹数据CSV文件路径
    left_csv = r"E:\test-qingxie\1\left_tracking_data.csv"
    right_csv = r"E:\test-qingxie\1\right_tracking_data.csv"
    
    # 加载轨迹数据
    tracker.load_tracks_csv(left_csv, right_csv)
    
    # 计算三维轨迹
    tracker.calculate_3d_tracks()
    
    # 保存三维轨迹数据
    output_dir = r"E:\test-qingxie\1\3d_tracks"
    os.makedirs(output_dir, exist_ok=True)
    tracker.save_tracks_3d(os.path.join(output_dir, "3d_tracks.csv"), format='csv')
    tracker.save_tracks_3d(os.path.join(output_dir, "3d_tracks.json"), format='json')
    
    # 可视化三维轨迹
    tracker.visualize_tracks_3d(os.path.join(output_dir, "3d_trajectories.png"))
    
    # 可视化坐标变化
    tracker.visualize_coordinate_changes(output_dir)
    
    print("处理完成")


if __name__ == "__main__":
    main()

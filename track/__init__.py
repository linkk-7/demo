# 设置工作目录为下载项目的文件夹路径
import torch
torch.set_grad_enabled(False)
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
sys.path.append(r"E:\研究生\西安碑林\Superpoint双目位移监测代码_原始版0606\LightGlue-main")
from lightglue.utils import load_image, load_image1,rbd
from lightglue import viz2d
from matplotlib import pyplot as plt
import numpy as np
import logging
import cv2
from tqdm import tqdm
import os
import re


def track(images_list, extractor, matcher,init_idx):
    image0 = load_image1(images_list[0])
    # 检测特征点
    feats0 = extractor.extract(image0.to(device))
    his = []
    for i in range(1, len(images_list)):
        yes = 0
        image1 = load_image1(images_list[i])
        feats1_batch = extractor.extract(image1.to(device))
        # 匹配两张图中的特征点对
        matches01 = matcher({"image0": feats0, "image1": feats1_batch})
        # 去除批量的那个维度
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1_batch, matches01]
        ]

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        

        for idx in range(len(matches)):
            if init_idx == matches[idx][0]:
                yes = 1
                print('在第{}张图片找到特征点！'.format(i+1),  '原坐标：', kpts0[matches[idx, 0]], '找到的坐标：', kpts1[matches[idx, 1]],'原特征点ID：', init_idx, '新特征点ID：', matches[idx][1])

                init_idx = matches[idx, 1]
                his.append(kpts0[matches[idx, 0]].cpu().numpy())
                feats0 = feats1_batch
                break
        if yes == 0:
            print('在{}图片丢失特征点'.format(images_list[i-1]))
            his.append(kpts0[matches[idx, 1]].cpu().numpy())
            print('未匹配成功')
            return his
    his.append(kpts1[matches[idx, 1]].cpu().numpy())
    return his


def track_multiple_points(images_list, extractor, matcher, init_idxs):
    # 初始化数据结构以存储每个点的追踪历史
    his = [[] for _ in range(len(init_idxs))]
    image0 = load_image1(images_list[0])
    feats0 = extractor.extract(image0.to(device))

    for i in range(1, len(images_list)):
        image1 = load_image1(images_list[i])
        feats1_batch = extractor.extract(image1.to(device))

        # 匹配两张图中的特征点对
        matches01 = matcher({"image0": feats0, "image1": feats1_batch})
        # 去除批量的那个维度
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1_batch, matches01]
        ]

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

        for j, init_idx in enumerate(init_idxs):
            if init_idx == -1:
                his[j].append(np.array([0, 0]))  # 当未找到匹配时，记录 [0, 0]
                continue
            flag = False
            for match in matches:
                if init_idx == match[0]:
                    his[j].append(kpts0[init_idx].cpu().numpy())  # 记录模板图像中的坐标
                    init_idxs[j] = match[1]
                    flag = True
                    break
            if not flag:
                his[j].append(np.array([0, 0]))  # 当未找到匹配时，记录 [0, 0]
                init_idxs[j] = -1

        feats0 = feats1_batch  # 更新特征，以便进行下一次匹配

    # 在最后记录每个点在最后一张图像中的位置
    for j, init_idx in enumerate(init_idxs):
        if init_idx != -1:
            his[j].append(kpts1[init_idx].cpu().numpy())
        else:
            his[j].append(np.array([0, 0]))

    return his

def track_sift(images_list, init_point_x, init_point_y, visualize=False, save_path=None):
    """
    使用SIFT算法追踪特定点在图像序列中的运动
    
    参数:
    images_list: 图像路径列表
    init_point_x, init_point_y: 初始追踪点的坐标
    visualize: 是否可视化追踪过程
    save_path: 可视化结果保存路径
    
    返回:
    his: 追踪点的历史坐标列表
    """
    # 初始化SIFT特征检测器
    sift = cv2.SIFT_create(nfeatures=2000)  # 可以调整特征点数量
    
    # 特征匹配器 - 使用FLANN匹配器提高速度
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 或更高以获得更精确的结果
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # 加载第一张图像
    image0 = cv2.imread(images_list[0], cv2.IMREAD_GRAYSCALE)
    if image0 is None:
        print(f"无法读取第一张图像: {images_list[0]}")
        return []
    
    # 检测第一张图像的特征点和描述符
    kp0, des0 = sift.detectAndCompute(image0, None)
    
    # 找到距离初始追踪点最近的特征点
    init_idx = -1
    min_dist = float('inf')
    for i, kp in enumerate(kp0):
        x, y = kp.pt
        dist = np.sqrt((x - init_point_x)**2 + (y - init_point_y)**2)
        if dist < min_dist:
            min_dist = dist
            init_idx = i
    
    if init_idx == -1 or min_dist > 20:  # 阈值可调整
        print(f"在第一张图像中没有找到靠近 ({init_point_x}, {init_point_y}) 的特征点")
        if len(kp0) > 0:
            print(f"最近的特征点距离为 {min_dist} 像素")
            init_idx = 0  # 使用第一个特征点作为备选
        else:
            return []
    
    print(f"选择ID={init_idx}的特征点作为起始追踪点，坐标: {kp0[init_idx].pt}，与目标点距离: {min_dist:.2f}像素")
    
    # 初始化历史坐标列表
    his = [np.array(kp0[init_idx].pt)]
    
    # 可视化设置
    if visualize:
        visualizations = []
        # 绘制第一帧的追踪点
        vis_img = cv2.cvtColor(image0, cv2.COLOR_GRAY2BGR)
        x, y = kp0[init_idx].pt
        cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(vis_img, f"ID: {init_idx}", (int(x) + 10, int(y) + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        visualizations.append(vis_img)
    
    # 逐帧追踪
    for i in tqdm(range(1, len(images_list)), desc="SIFT追踪"):
        # 加载当前帧
        image1 = cv2.imread(images_list[i], cv2.IMREAD_GRAYSCALE)
        if image1 is None:
            print(f"警告: 无法读取图像 {images_list[i]}")
            if len(his) > 0:
                his.append(his[-1])  # 使用上一帧的坐标
            continue
        
        # 检测当前帧的特征点和描述符
        kp1, des1 = sift.detectAndCompute(image1, None)
        
        if des1 is None or len(des1) == 0:
            print(f"警告: 在图像 {i+1} 中没有检测到特征点")
            if len(his) > 0:
                his.append(his[-1])  # 使用上一帧的坐标
            continue
            
        if des0 is None or len(des0) == 0:
            print(f"警告: 在图像 {i} 中没有检测到特征点")
            if len(his) > 0:
                his.append(his[-1])  # 使用上一帧的坐标
            # 更新参考帧
            image0, kp0, des0 = image1, kp1, des1
            continue
        
        # 匹配特征点
        matches = flann.knnMatch(des0, des1, k=2)
        
        # 应用比率测试筛选良好匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # 比率测试阈值
                good_matches.append(m)
        
        if len(good_matches) < 2:
            print(f"警告: 在图像 {i+1} 中找到的良好匹配不足 ({len(good_matches)})")
            if len(his) > 0:
                his.append(his[-1])  # 使用上一帧的坐标
            continue
        
        # 查找跟踪目标特征点的匹配
        found = False
        for match in good_matches:
            if match.queryIdx == init_idx:
                # 在当前帧中找到了匹配点
                found = True
                matched_point = kp1[match.trainIdx].pt
                his.append(np.array(matched_point))
                print(f'在第 {i+1} 张图像找到特征点！', 
                      f'原坐标: {kp0[init_idx].pt}', 
                      f'新坐标: {matched_point}',
                      f'原特征点ID: {init_idx}', 
                      f'新特征点ID: {match.trainIdx}')
                
                # 更新参考帧和跟踪点ID
                init_idx = match.trainIdx
                image0, kp0, des0 = image1, kp1, des1
                
                # 可视化
                if visualize:
                    vis_img = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
                    x, y = matched_point
                    cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)
                    # 绘制运动轨迹
                    for j in range(1, len(his)):
                        pt1 = (int(his[j-1][0]), int(his[j-1][1]))
                        pt2 = (int(his[j][0]), int(his[j][1]))
                        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)
                    # 添加特征点ID和坐标信息
                    cv2.putText(vis_img, f"ID: {init_idx}, Pos: ({int(x)}, {int(y)})", 
                               (int(x) + 10, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    visualizations.append(vis_img)
                break
        
        # 如果没有找到匹配点
        if not found:
            print(f'在第 {i+1} 张图像中丢失特征点')
            if len(his) > 0:
                his.append(his[-1])  # 使用上一帧的坐标
            else:
                print('追踪失败')
                return []
    
    # 保存可视化结果
    if visualize and len(visualizations) > 0:
        if save_path:
            # 保存为视频
            height, width = visualizations[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(f"{save_path}/sift_tracking.avi", fourcc, 10.0, (width, height))
            
            for frame in visualizations:
                video_writer.write(frame)
            video_writer.release()
            print(f"追踪可视化视频已保存至: {save_path}/sift_tracking.avi")
            
            # 保存最终轨迹图
            plt.figure(figsize=(10, 8))
            his_array = np.array(his)
            plt.plot(his_array[:, 0], his_array[:, 1], 'r-')
            plt.plot(his_array[0, 0], his_array[0, 1], 'go', label='起点')
            plt.plot(his_array[-1, 0], his_array[-1, 1], 'bo', label='终点')
            plt.xlabel('X (像素)')
            plt.ylabel('Y (像素)')
            plt.title('SIFT特征点追踪轨迹')
            plt.legend()
            plt.grid(True)
            plt.gca().invert_yaxis()  # 图像坐标系y轴向下
            plt.savefig(f"{save_path}/sift_tracking_trajectory.png")
            plt.close()
            print(f"追踪轨迹图已保存至: {save_path}/sift_tracking_trajectory.png")
    
    # 转换为numpy数组并返回
    his_array = np.array(his)
    
    # 打印统计信息
    if len(his_array) > 0:
        print(f"追踪完成: 共 {len(his_array)} 个点")
        print(f"起始坐标: ({his_array[0][0]:.2f}, {his_array[0][1]:.2f})")
        print(f"结束坐标: ({his_array[-1][0]:.2f}, {his_array[-1][1]:.2f})")
        
        # 计算总位移
        total_displacement = np.sqrt(np.sum((his_array[-1] - his_array[0])**2))
        print(f"总位移: {total_displacement:.2f} 像素")
    
    return his_array

def enhanced_track(images_list, extractor, matcher, init_idx, template_size=41, search_margin=100):
    """
    增强版特征点追踪，在特征匹配失败时使用模板匹配作为备用
    
    参数:
        images_list: 图像路径列表
        extractor: 特征提取器
        matcher: 特征匹配器
        init_idx: 初始特征点ID
        template_size: 模板大小，应为奇数
        search_margin: 模板匹配搜索范围扩展边距
    """
    # 读取第一张图像
    image0_path = images_list[0]
    image0_cv = cv2.imread(image0_path)
    image0 = load_image1(image0_path)
    
    # 检测特征点
    feats0 = extractor.extract(image0.to(device))
    his = []
    
    # 记录上一次成功匹配的图像和特征点位置
    last_success_image = image0_cv
    last_success_point = None
    last_success_feats = feats0  # 保存上一次成功特征匹配的特征
    
    half_size = template_size // 2
    
    for i in range(1, len(images_list)):
        matched = False
        current_image_path = images_list[i]
        
        # 读取当前帧
        image1_cv = cv2.imread(current_image_path)
        image1 = load_image1(current_image_path)
        
        # 提取当前帧特征
        feats1_batch = extractor.extract(image1.to(device))
        
        # 尝试使用特征匹配 - 始终使用上一次成功特征匹配的特征
        matches01 = matcher({"image0": last_success_feats, "image1": feats1_batch})
        feats0_rbd, feats1, matches01 = [rbd(x) for x in [last_success_feats, feats1_batch, matches01]]
        
        kpts0, kpts1, matches = feats0_rbd["keypoints"], feats1["keypoints"], matches01["matches"]
        
        # 尝试特征匹配
        for idx in range(len(matches)):
            if init_idx == matches[idx][0]:
                matched = True
                current_point = kpts1[matches[idx, 1]]
                
                print(f'在第{i+1}张图片找到特征点！ '
                      f'原坐标：{kpts0[matches[idx, 0]]} '
                      f'找到的坐标：{current_point} '
                      f'原特征点ID：{init_idx} '
                      f'新特征点ID：{matches[idx][1]}')
                
                # 更新特征点ID
                init_idx = matches[idx, 1]
                his.append(kpts0[matches[idx, 0]].cpu().numpy())
                
                # 更新上一次成功匹配的信息
                last_success_feats = feats1_batch
                last_success_image = image1_cv
                last_success_point = (int(current_point[0]), int(current_point[1]))
                break
        
        # 如果特征匹配失败，尝试模板匹配
        if not matched and last_success_point is not None:
            print(f'在第{i}张图片中特征匹配失败，尝试模板匹配...')
            
            # 从上一帧成功的图像中提取模板
            x, y = last_success_point
            
            # 确保模板不会超出图像边界
            h, w = last_success_image.shape[:2]
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(w - 1, x + half_size)
            y2 = min(h - 1, y + half_size)
            
            # 提取模板
            template = last_success_image[y1:y2+1, x1:x2+1]
            template_height, template_width = template.shape[:2]
            
            if template_height > 0 and template_width > 0:
                # 定义搜索区域
                search_x1 = max(0, x - search_margin)
                search_y1 = max(0, y - search_margin)
                search_x2 = min(w - 1, x + search_margin)
                search_y2 = min(h - 1, y + search_margin)
                
                # 提取搜索区域
                search_area = image1_cv[search_y1:search_y2+1, search_x1:search_x2+1]
                
                if search_area.shape[0] > template.shape[0] and search_area.shape[1] > template.shape[1]:
                    # 执行模板匹配
                    result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    # 判断匹配结果是否可靠
                    if max_val > 0.7:  # 可调整阈值
                        # 计算全局坐标
                        match_x = search_x1 + max_loc[0] + template_width // 2
                        match_y = search_y1 + max_loc[1] + template_height // 2
                        
                        print(f'模板匹配成功！匹配分数: {max_val:.3f}, 坐标: ({match_x}, {match_y})')
                        
                        # 添加到历史记录
                        his.append(np.array([match_x, match_y]))
                        
                        # 这里是关键修改：我们不更新特征点ID，继续使用上次的ID
                        # 但我们更新last_success_point以用于下一次可能的模板匹配
                        last_success_point = (match_x, match_y)
                        
                        # 关键：我们不更新last_success_feats，继续使用最后一次特征匹配成功的特征
                        # 下一帧仍然尝试使用特征匹配，而不是基于模板匹配结果重新关联特征
                        
                        matched = True
                    else:
                        print(f'模板匹配分数过低: {max_val:.3f}，匹配失败')
                else:
                    print(f'搜索区域或模板过小，无法执行模板匹配')
            else:
                print(f'无法提取有效模板，模板大小: {template_width}x{template_height}')
        
        # 如果所有尝试都失败
        if not matched:
            print(f'在{images_list[i-1]}图片丢失特征点，追踪失败')
            if len(his) > 0:
                his.append(his[-1])  # 使用上一个位置作为估计
                print('使用上一帧位置作为估计')
            return his
    
    # 添加最后一帧的位置
    if matched and len(matches) > 0:
        his.append(kpts1[init_idx].cpu().numpy())
    
    return his

def predict_position(trajectory, method='velocity', frames_ahead=1):
    """
    基于历史轨迹预测未来位置
    
    参数:
        trajectory: 历史轨迹点列表
        method: 预测方法
            - 'velocity': 基于最近两帧的速度预测
            - 'acceleration': 考虑加速度的预测
            - 'average': 基于最近几帧的平均速度预测
        frames_ahead: 预测的帧数（默认为1，即下一帧）
    
    返回:
        predicted_point: 预测的点坐标 [x, y]
    """
    trajectory = np.array(trajectory)
    n = len(trajectory)
    
    if n < 2:
        return trajectory[-1]  # 如果只有一个点，无法预测，返回最后一个点
    
    if method == 'velocity':
        # 基于最后两帧的速度预测
        velocity = trajectory[-1] - trajectory[-2]
        return trajectory[-1] + velocity * frames_ahead
    
    elif method == 'acceleration':
        # 考虑加速度的预测
        if n < 3:
            # 如果点数不足，退化为速度预测
            velocity = trajectory[-1] - trajectory[-2]
            return trajectory[-1] + velocity * frames_ahead
        else:
            # 计算最近两个速度向量
            velocity1 = trajectory[-2] - trajectory[-3]
            velocity2 = trajectory[-1] - trajectory[-2]
            # 计算加速度
            acceleration = velocity2 - velocity1
            # 使用物理公式：s = s0 + v0*t + 0.5*a*t^2
            return trajectory[-1] + velocity2 * frames_ahead + 0.5 * acceleration * frames_ahead**2
    
    elif method == 'average':
        # 基于最近几帧的平均速度
        # 使用最后min(5, n-1)帧计算平均速度
        window = min(5, n-1)
        velocities = []
        for i in range(1, window+1):
            if n-i-1 >= 0:  # 确保索引有效
                velocities.append(trajectory[-i] - trajectory[-i-1])
        
        if not velocities:
            # 如果无法计算速度，退化为简单速度预测
            velocity = trajectory[-1] - trajectory[-2]
            return trajectory[-1] + velocity * frames_ahead
        
        # 计算平均速度
        avg_velocity = np.mean(velocities, axis=0)
        return trajectory[-1] + avg_velocity * frames_ahead
    
    else:
        raise ValueError(f"未知的预测方法: {method}")



def extract_frame_number(file_path):
    """
    从文件路径中提取数字（帧号）
    
    参数:
        file_path: 文件路径
        
    返回:
        frame_number: 提取的数字，如果没有找到则返回None
    """
    # 获取文件名（不含路径）
    file_name = os.path.basename(file_path)
    
    # 从文件名中提取数字
    numbers = re.findall(r'\d+', file_name)
    if numbers:
        return int(numbers[0])  # 返回第一个数字
    return None

def enhanced_track_with_prediction(images_list, extractor, matcher, init_idx, template_size=41, 
                                 search_margin=100, max_prediction_frames=30, prediction_method='velocity'):
    """
    增强版特征点追踪，结合特征匹配、模板匹配和轨迹预测，并记录遮挡时间段
    模板匹配和预测都视为遮挡
    
    参数:
        images_list: 图像路径列表
        extractor: 特征提取器
        matcher: 特征匹配器
        init_idx: 初始特征点ID
        template_size: 模板大小，应为奇数
        search_margin: 模板匹配搜索范围扩展边距
        max_prediction_frames: 最大连续预测帧数
        prediction_method: 预测方法 ('velocity', 'acceleration', 'average')
    
    返回:
        his: 追踪点的历史坐标列表
        occluded_frames: 遮挡帧列表，格式为[[frame1, frame2, ...], [...], ...]，每个子列表表示一段连续遮挡
    """
    # 读取第一张图像
    image0_path = images_list[0]
    image0_cv = cv2.imread(image0_path)
    image0 = load_image1(image0_path)
    
    # 检测特征点
    feats0 = extractor.extract(image0.to(device))
    his = []
    
    # 记录上一次成功匹配的图像和特征点位置
    last_success_image = image0_cv
    last_success_point = None
    last_success_feats = feats0  # 保存上一次成功特征匹配的特征
    
    # 预测相关变量
    prediction_frames = 0  # 当前连续预测的帧数
    prediction_mode = False  # 是否处于预测模式
    prediction_reliability = 1.0  # 预测可靠性（0-1之间，随预测时间降低）
    
    # 记录遮挡时间段
    occluded_frames = []  # 存储遮挡帧的文件名中的数字
    current_occlusion = []  # 当前遮挡序列
    
    half_size = template_size // 2
    
    for i in range(1, len(images_list)):
        matched = False
        current_image_path = images_list[i]
        
        # 读取当前帧
        image1_cv = cv2.imread(current_image_path)
        image1 = load_image1(current_image_path)
        
        # 提取当前帧特征
        feats1_batch = extractor.extract(image1.to(device))
        
        # 尝试使用特征匹配 - 始终使用上一次成功特征匹配的特征
        matches01 = matcher({"image0": last_success_feats, "image1": feats1_batch})
        feats0_rbd, feats1, matches01 = [rbd(x) for x in [last_success_feats, feats1_batch, matches01]]
        
        kpts0, kpts1, matches = feats0_rbd["keypoints"], feats1["keypoints"], matches01["matches"]
        
        # 尝试特征匹配
        for idx in range(len(matches)):
            if init_idx == matches[idx][0]:
                matched = True
                current_point = kpts1[matches[idx, 1]]
                
                print(f'在第{i+1}张图片找到特征点！ '
                      f'原坐标：{kpts0[matches[idx, 0]]} '
                      f'找到的坐标：{current_point} '
                      f'原特征点ID：{init_idx} '
                      f'新特征点ID：{matches[idx][1]}')
                
                # 更新特征点ID
                init_idx = matches[idx, 1]
                his.append(kpts0[matches[idx, 0]].cpu().numpy())
                
                # 更新上一次成功匹配的信息
                last_success_feats = feats1_batch
                last_success_image = image1_cv
                last_success_point = (int(current_point[0]), int(current_point[1]))
                
                # 重置预测相关变量
                prediction_frames = 0
                prediction_mode = False
                prediction_reliability = 1.0
                
                # 如果之前在预测模式，现在退出了预测模式，则保存当前遮挡序列
                if len(current_occlusion) > 0:
                    occluded_frames.append(current_occlusion)
                    current_occlusion = []
                
                break
        
        # 如果特征匹配失败，尝试模板匹配
        if not matched and last_success_point is not None:
            print(f'在第{i}张图片中特征匹配失败，尝试模板匹配...')
            
            # 修改：将当前帧号添加到遮挡序列，因为模板匹配也视为遮挡
            frame_number = extract_frame_number(current_image_path)
            if frame_number is not None:
                current_occlusion.append(frame_number)
            
            # 从上一帧成功的图像中提取模板
            x, y = last_success_point
            
            # 确保模板不会超出图像边界
            h, w = last_success_image.shape[:2]
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(w - 1, x + half_size)
            y2 = min(h - 1, y + half_size)
            
            # 提取模板
            template = last_success_image[y1:y2+1, x1:x2+1]
            template_height, template_width = template.shape[:2]
            
            if template_height > 0 and template_width > 0:
                # 定义搜索区域
                search_x1 = max(0, x - search_margin)
                search_y1 = max(0, y - search_margin)
                search_x2 = min(w - 1, x + search_margin)
                search_y2 = min(h - 1, y + search_margin)
                
                # 提取搜索区域
                search_area = image1_cv[search_y1:search_y2+1, search_x1:search_x2+1]
                
                if search_area.shape[0] > template.shape[0] and search_area.shape[1] > template.shape[1]:
                    # 执行模板匹配
                    result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    # 判断匹配结果是否可靠
                    if max_val > 0.7:  # 可调整阈值
                        # 计算全局坐标
                        match_x = search_x1 + max_loc[0] + template_width // 2
                        match_y = search_y1 + max_loc[1] + template_height // 2
                        
                        print(f'模板匹配成功！匹配分数: {max_val:.3f}, 坐标: ({match_x}, {match_y})')
                        
                        # 添加到历史记录
                        his.append(np.array([match_x, match_y]))
                        
                        # 不更新特征点ID，继续使用上次的ID
                        # 但更新last_success_point用于下次模板匹配
                        last_success_point = (match_x, match_y)
                        last_success_image = image1_cv
                        
                        # 重置预测相关变量
                        prediction_frames = 0
                        prediction_mode = False
                        prediction_reliability = 1.0
                        
                        # 修改：即使模板匹配成功，也将其视为遮挡的一部分
                        # 如果本次是该段遮挡的最后一帧，保存当前遮挡序列
                        # 不清空current_occlusion，因为模板匹配视为遮挡状态的一部分
                        
                        matched = True
                    else:
                        print(f'模板匹配分数过低: {max_val:.3f}，匹配失败')
                else:
                    print(f'搜索区域或模板过小，无法执行模板匹配')
            else:
                print(f'无法提取有效模板，模板大小: {template_width}x{template_height}')
        
        # 如果特征匹配和模板匹配都失败，使用轨迹预测
        if not matched and len(his) >= 2:
            prediction_mode = True
            prediction_frames += 1
            
            # 计算预测的可靠性 - 随着预测帧数增加而降低
            prediction_reliability = max(0.1, 1.0 - (prediction_frames / max_prediction_frames))
            
            # 预测下一个位置
            predicted_point = predict_position(his, prediction_method, prediction_frames)
            
            print(f'在第{i+1}张图片中匹配失败，使用轨迹预测. '
                 f'预测坐标: ({predicted_point[0]:.2f}, {predicted_point[1]:.2f}), '
                 f'预测可靠性: {prediction_reliability:.2f}, '
                 f'连续预测帧数: {prediction_frames}')
            
            # 添加预测点到历史记录
            his.append(predicted_point)
            
            # 更新last_success_point以便在下一帧恢复时使用
            last_success_point = (int(predicted_point[0]), int(predicted_point[1]))
            
            # 如果尚未添加帧号（在模板匹配阶段添加过），则添加
            frame_number = extract_frame_number(current_image_path)
            if frame_number is not None and frame_number not in current_occlusion:
                current_occlusion.append(frame_number)
            
            # 如果已经超过最大预测帧数，发出警告但继续预测
            if prediction_frames >= max_prediction_frames:
                print(f'警告: 已达到最大预测帧数({max_prediction_frames})，预测可能不准确!')
                
            matched = True  # 通过预测"匹配"成功
        
        # 如果所有尝试都失败（包括预测）
        if not matched:
            print(f'在{images_list[i-1]}图片丢失特征点，且无法预测，追踪失败')
            if len(his) > 0:
                his.append(his[-1])  # 使用上一个位置作为估计
                print('使用上一帧位置作为估计')
            
            # 保存最后一个遮挡序列（如果有）
            if len(current_occlusion) > 0:
                occluded_frames.append(current_occlusion)
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return his, occluded_frames  # 返回两个值
    
    # 处理追踪完成后可能还存在的遮挡序列
    if len(current_occlusion) > 0:
        occluded_frames.append(current_occlusion)
    
    # 添加最后一帧的位置
    if matched and not prediction_mode and len(matches) > 0:
        his.append(kpts1[init_idx].cpu().numpy())
    
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return his, occluded_frames  # 返回两个值

def track_optical_flow(images_list, init_point_x, init_point_y, visualize=False, save_path=None):
    """
    使用Lucas-Kanade光流法追踪特定点在图像序列中的运动
    
    参数:
    images_list: 图像路径列表
    init_point_x, init_point_y: 初始追踪点的坐标
    visualize: 是否可视化追踪过程
    save_path: 可视化结果保存路径
    
    返回:
    his: 追踪点的历史坐标列表
    """
    # 读取第一张图像
    image0 = cv2.imread(images_list[0], cv2.IMREAD_GRAYSCALE)
    if image0 is None:
        print(f"无法读取第一张图像: {images_list[0]}")
        return []
    
    # 初始化Shi-Tomasi角点检测器参数
    feature_params = dict(maxCorners=1000, 
                         qualityLevel=0.3,
                         minDistance=7,
                         blockSize=7)
    
    # 初始化Lucas-Kanade光流法参数
    lk_params = dict(winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # 在第一帧中检测特征点
    corners = cv2.goodFeaturesToTrack(image0, mask=None, **feature_params)
    
    # 找到距离初始追踪点最近的特征点
    init_idx = -1
    min_dist = float('inf')
    for i, corner in enumerate(corners):
        x, y = corner.ravel()
        dist = np.sqrt((x - init_point_x)**2 + (y - init_point_y)**2)
        if dist < min_dist:
            min_dist = dist
            init_idx = i
    
    if init_idx == -1 or min_dist > 20:  # 阈值可调整
        print(f"在第一张图像中没有找到靠近 ({init_point_x}, {init_point_y}) 的特征点")
        if len(corners) > 0:
            print(f"最近的特征点距离为 {min_dist} 像素")
            init_idx = 0  # 使用第一个特征点作为备选
        else:
            # 直接使用用户指定的点
            corners = np.array([[[init_point_x, init_point_y]]], dtype=np.float32)
            init_idx = 0
            print(f"使用初始指定点 ({init_point_x}, {init_point_y}) 作为起始追踪点")
    else:
        print(f"选择ID={init_idx}的特征点作为起始追踪点，坐标: {corners[init_idx].ravel()}，与目标点距离: {min_dist:.2f}像素")
    
    # 仅保留要追踪的点
    p0 = np.array([corners[init_idx]])
    
    # 初始化历史坐标列表
    his = [p0[0].ravel()]
    
    # 创建随机颜色
    color = np.random.randint(0, 255, (1, 3)).tolist()[0]
    
    # 可视化设置
    if visualize:
        visualizations = []
        # 绘制第一帧的追踪点
        vis_img = cv2.cvtColor(image0, cv2.COLOR_GRAY2BGR)
        x, y = p0[0].ravel()
        cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(vis_img, f"Start", (int(x) + 10, int(y) + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        visualizations.append(vis_img)
    
    # 创建掩膜图像
    mask = np.zeros_like(vis_img) if visualize else None
    
    # 逐帧追踪
    old_gray = image0
    
    for i in tqdm(range(1, len(images_list)), desc="光流法追踪"):
        # 读取当前帧
        frame = cv2.imread(images_list[i])
        if frame is None:
            print(f"警告: 无法读取图像 {images_list[i]}")
            if len(his) > 0:
                his.append(his[-1])  # 使用上一帧的坐标
                if visualize:
                    visualizations.append(visualizations[-1])
            continue
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算光流以获取新位置
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        # 检查是否成功跟踪
        if status[0][0] == 1:
            # 追踪成功
            new_point = p1[0].ravel()
            his.append(new_point)
            
            print(f'在第 {i+1} 张图像成功追踪点！', 
                  f'原坐标: {p0[0].ravel()}', 
                  f'新坐标: {new_point}')
            
            # 更新点的位置和前一帧
            p0 = p1.reshape(-1, 1, 2)
            old_gray = frame_gray.copy()
            
            # 可视化
            if visualize:
                vis_img = frame.copy()
                x, y = new_point
                
                # 绘制当前点
                cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)
                
                # 绘制运动轨迹
                cv2.line(mask, (int(his[-2][0]), int(his[-2][1])), 
                         (int(x), int(y)), color, 2)
                
                # 合并图像和轨迹
                vis_img = cv2.add(vis_img, mask)
                
                # 添加坐标信息
                cv2.putText(vis_img, f"Pos: ({int(x)}, {int(y)})", 
                           (int(x) + 10, int(y) + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                visualizations.append(vis_img)
        else:
            # 追踪失败
            print(f'在第 {i+1} 张图像中丢失追踪点')
            
            if len(his) > 0:
                # 使用上一帧的坐标并尝试在下一帧重新开始追踪
                his.append(his[-1])
                p0 = np.array([[his[-1]]], dtype=np.float32)
                
                # 可视化
                if visualize:
                    # 复制上一帧的可视化结果
                    if len(visualizations) > 0:
                        vis_img = visualizations[-1].copy()
                        x, y = his[-1]
                        cv2.putText(vis_img, "Lost", (int(x) + 10, int(y) + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        visualizations.append(vis_img)
            else:
                print('追踪失败')
                return []
    
    # 保存可视化结果
    if visualize and len(visualizations) > 0:
        if save_path:
            # 保存为视频
            height, width = visualizations[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(f"{save_path}/optical_flow_tracking.avi", 
                                          fourcc, 10.0, (width, height))
            
            for frame in visualizations:
                video_writer.write(frame)
            video_writer.release()
            print(f"追踪可视化视频已保存至: {save_path}/optical_flow_tracking.avi")
            
            # 保存最终轨迹图
            plt.figure(figsize=(10, 8))
            his_array = np.array(his)
            plt.plot(his_array[:, 0], his_array[:, 1], 'r-')
            plt.plot(his_array[0, 0], his_array[0, 1], 'go', label='起点')
            plt.plot(his_array[-1, 0], his_array[-1, 1], 'bo', label='终点')
            plt.xlabel('X (像素)')
            plt.ylabel('Y (像素)')
            plt.title('光流法特征点追踪轨迹')
            plt.legend()
            plt.grid(True)
            plt.gca().invert_yaxis()  # 图像坐标系y轴向下
            plt.savefig(f"{save_path}/optical_flow_tracking_trajectory.png")
            plt.close()
            print(f"追踪轨迹图已保存至: {save_path}/optical_flow_tracking_trajectory.png")
    
    # 转换为numpy数组并返回
    his_array = np.array(his)
    
    # 打印统计信息
    if len(his_array) > 0:
        print(f"追踪完成: 共 {len(his_array)} 个点")
        print(f"起始坐标: ({his_array[0][0]:.2f}, {his_array[0][1]:.2f})")
        print(f"结束坐标: ({his_array[-1][0]:.2f}, {his_array[-1][1]:.2f})")
        
        # 计算总位移
        total_displacement = np.sqrt(np.sum((his_array[-1] - his_array[0])**2))
        print(f"总位移: {total_displacement:.2f} 像素")
    
    return his_array





# ----------相机标定----------
# 功能：单目标定、双目标定、立体校正和可视化
# --------------------------

import cv2
import cv2.aruco as aruco
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont
import glob
import os


# ----------角点检测函数----------
# board：一个 cv2.aruco.CharucoBoard 对象，里面记录了棋盘格尺寸、字典、角点间距等先验信息
def detect_charuco_corners(image_paths, board, save_dir=None):
    all_corners = []  # 存储所有检测到的角点坐标
    all_ids = []  # 存储所有角点的ID
    all_images = []  # 存储所有灰度图像

    for i, path in enumerate(image_paths):
        image = cv2.imread(path)  # 读彩图
        if image is None: continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图

        # 检测ChArUco角点
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, board.getDictionary())

        # 保存角点检测图
        if len(corners) > 0 and save_dir is not None:
            # 绘制检测到的标记
            img_with_markers = image.copy()

            # 使用OpenCV绘制ArUco标记
            cv2.aruco.drawDetectedMarkers(img_with_markers, corners, ids)

            # 手动绘制大号ID标签，使其更明显
            if ids is not None:
                for j, corner in enumerate(corners):
                    # 计算标记的中心位置
                    center_x = int(np.mean(corner[0][:, 0]))
                    center_y = int(np.mean(corner[0][:, 1]))

                    # 使用明亮的红色和加粗字体绘制ID
                    marker_id = str(ids[j][0])
                    cv2.putText(img_with_markers, marker_id,
                                (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,  # 字体大小增大
                                (0, 0, 255),  # 明亮的红色
                                3,  # 线条粗细
                                cv2.LINE_AA)

            # 插值ChArUco角点
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)

            if ret > 0:
                # 更加明显地绘制ChArUco角点
                for j, corner in enumerate(charuco_corners):
                    # 绘制更大更明显的圆点
                    cv2.circle(img_with_markers,
                              (int(corner[0][0]), int(corner[0][1])),
                              8,  # 增大圆点大小
                              (0, 255, 0),  # 亮绿色
                              -1)  # 填充圆

                    # 在角点旁边添加ID号
                    charuco_id = str(charuco_ids[j][0])
                    cv2.putText(img_with_markers, charuco_id,
                               (int(corner[0][0]) + 10, int(corner[0][1]) + 10),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.8,  # 字体大小
                               (255, 0, 255),  # 洋红色
                               2,  # 线条粗细
                               cv2.LINE_AA)

                # 保存图像
                if i < 5:  # 只保存前5张图片作为示例
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    output_path = os.path.join(save_dir, f"charuco_detection_{i}.jpg")
                    cv2.imwrite(output_path, img_with_markers)

                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                all_images.append(gray)

    return all_corners, all_ids, all_images

def calculate_reprojection_errors(corners, ids, rvecs, tvecs, camera_matrix, dist_coeffs, board):
    """
    计算每张图像的平均重投影误差
    """
    errors = []
    for i in range(len(corners)):
        # 获取标定板的对象点
        objp = board.getChessboardCorners()
        # 筛选当前图像中实际检测到的角点
        detected_ids = ids[i].flatten()
        valid_points = []
        for j, point_id in enumerate(detected_ids):
            valid_points.append(objp[point_id])
        valid_points = np.array(valid_points)
        
        # 使用标定参数进行重投影
        img_points, _ = cv2.projectPoints(valid_points, rvecs[i], tvecs[i], 
                                         camera_matrix, dist_coeffs)
        
        # 计算重投影误差
        total_error = 0
        for j, (corner, img_point) in enumerate(zip(corners[i], img_points)):
            error = np.sqrt(np.sum((corner[0] - img_point[0])**2))
            total_error += error
        
        # 计算平均误差
        mean_error = total_error / len(corners[i])
        errors.append(mean_error)
    
    return errors


# -----------------以下为新增------------------
# -------------------------------------------
def calculate_stereo_reprojection_errors(
        left_corners, left_ids,
        right_corners, right_ids,
        board,
        left_camera_matrix, left_dist_coeffs,
        right_camera_matrix, right_dist_coeffs,
        R, T,
        min_common_ids=12):
    """
    计算双目标定的重投影误差

    返回：
        stereo_errors: 每对图像的双目平均误差
        left_errors:   每对图像左图平均误差
        right_errors:  每对图像右图平均误差
        valid_indices: 实际参与计算的图对索引
        overall_mean:  所有图对整体平均双目误差
    """
    all_corners3d = board.getChessboardCorners()

    stereo_errors = []
    left_errors = []
    right_errors = []
    valid_indices = []

    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)

    num_pairs = min(len(left_corners), len(right_corners))

    for i in range(num_pairs):
        if left_ids[i] is None or right_ids[i] is None:
            continue
        if len(left_ids[i]) < min_common_ids or len(right_ids[i]) < min_common_ids:
            continue

        left_ids_flat = left_ids[i].flatten()
        right_ids_flat = right_ids[i].flatten()
        common_ids = np.intersect1d(left_ids_flat, right_ids_flat)

        if len(common_ids) < min_common_ids:
            continue

        objp = []
        imgp_left = []
        imgp_right = []

        for id_val in common_ids:
            objp.append(all_corners3d[id_val])

            left_idx = np.where(left_ids_flat == id_val)[0][0]
            right_idx = np.where(right_ids_flat == id_val)[0][0]

            imgp_left.append(left_corners[i][left_idx][0])
            imgp_right.append(right_corners[i][right_idx][0])

        objp = np.asarray(objp, dtype=np.float32)
        imgp_left = np.asarray(imgp_left, dtype=np.float32)
        imgp_right = np.asarray(imgp_right, dtype=np.float32)

        # 先由左图 solvePnP 求当前板相对于左相机的位姿
        success, rvec_left, tvec_left = cv2.solvePnP(
            objp, imgp_left,
            left_camera_matrix, left_dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            continue

        # 左目重投影：  用左相机把这些 3D 点投影到左图，和左图检测角点比较
        proj_left, _ = cv2.projectPoints(
            objp, rvec_left, tvec_left,
            left_camera_matrix, left_dist_coeffs
        )
        proj_left = proj_left.reshape(-1, 2)

        left_point_errors = np.linalg.norm(imgp_left - proj_left, axis=1)
        left_mean_error = np.mean(left_point_errors)

        # 根据双目外参，把“板相对左相机位姿”变换到“板相对右相机位姿”
        R_left, _ = cv2.Rodrigues(rvec_left)
        R_right = R @ R_left
        t_right = R @ tvec_left + T
        rvec_right, _ = cv2.Rodrigues(R_right)

        # 右目重投影：  用右相机把这些 3D 点投影到右图，和右图检测角点比较
        proj_right, _ = cv2.projectPoints(
            objp, rvec_right, t_right,
            right_camera_matrix, right_dist_coeffs
        )
        proj_right = proj_right.reshape(-1, 2)

        right_point_errors = np.linalg.norm(imgp_right - proj_right, axis=1)
        right_mean_error = np.mean(right_point_errors)

        stereo_mean_error = (left_mean_error + right_mean_error) / 2.0

        left_errors.append(float(left_mean_error))
        right_errors.append(float(right_mean_error))
        stereo_errors.append(float(stereo_mean_error))
        valid_indices.append(i)

    if len(stereo_errors) == 0:
        overall_mean = None
    else:
        overall_mean = float(np.mean(stereo_errors))

    return stereo_errors, left_errors, right_errors, valid_indices, overall_mean

def plot_stereo_reprojection_errors(stereo_errors, left_errors, right_errors, valid_indices,
                                    save_dir, filename="stereo_reprojection_errors.png"):
    """
    绘制双目标定重投影误差柱状图
    """
    if len(stereo_errors) == 0:
        print("没有可用于绘制的双目标定重投影误差数据")
        return

    x = np.arange(len(valid_indices))
    labels = [str(idx) for idx in valid_indices]
    bar_width = 0.25

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(14, 8))
    plt.bar(x - bar_width, left_errors, width=bar_width, label='左目误差')
    plt.bar(x, right_errors, width=bar_width, label='右目误差')
    plt.bar(x + bar_width, stereo_errors, width=bar_width, label='双目平均误差')

    plt.xlabel("图对索引", fontsize=14)
    plt.ylabel("重投影误差 (像素)", fontsize=14)
    plt.title("双目标定各图对重投影误差", fontsize=16)
    plt.xticks(x, labels)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"双目标定重投影误差图已保存到 {save_path}")
# -------------------------------------------
# -------------------------------------------

# 单目标定
def calibrate_single_camera(corners, ids, images, board, save_dir=None, camera_name="camera", distortion_mode=0):
    # 获取图像尺寸
    print(images[0].shape)
    h, w = images[0].shape[:2]
    
    # 设置标定标志
    flags = 0
    if distortion_mode == 1:
        # 仅计算k1,k2,p1,p2
        flags = cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
        print(f"{camera_name}相机使用简化畸变模型(k1,k2,p1,p2)")
    elif distortion_mode == 2:
        # 仅计算k1,k2
        flags = cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6 + cv2.CALIB_ZERO_TANGENT_DIST
        print(f"{camera_name}相机使用最简畸变模型(仅k1,k2)")
    elif distortion_mode == 3:
        # 假设没有畸变
        flags = cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6 + cv2.CALIB_ZERO_TANGENT_DIST
        print(f"{camera_name}相机假设无畸变")
    else:
        print(f"{camera_name}相机使用完整畸变模型")
    
    # 标定相机内参
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        corners, ids, board, (w, h), None, None, flags=flags)
    
    # 计算每张图像的重投影误差
    reprojection_errors = calculate_reprojection_errors(corners, ids, rvecs, tvecs, camera_matrix, dist_coeffs, board)
    
    # 如果需要保存重投影误差图
    if save_dir is not None and len(corners) > 0:
        # 选择第一张图像进行重投影误差可视化
        img_idx = 0
        if img_idx < len(corners):
            img = cv2.cvtColor(images[img_idx], cv2.COLOR_GRAY2BGR)
            
            # 获取标定板的对象点 - 修复错误的调用方式
            # 无参数获取所有角点坐标
            objp = board.getChessboardCorners()
            # 筛选当前图像中实际检测到的角点
            detected_ids = ids[img_idx].flatten()
            valid_points = []
            for i, point_id in enumerate(detected_ids):
                valid_points.append(objp[point_id])
            valid_points = np.array(valid_points)
            
            # 使用标定参数进行重投影
            img_points, _ = cv2.projectPoints(valid_points, rvecs[img_idx], tvecs[img_idx], 
                                             camera_matrix, dist_coeffs)
            
            # 计算重投影误差
            total_error = 0
            for i, (corner, img_point) in enumerate(zip(corners[img_idx], img_points)):
                error = np.sqrt(np.sum((corner[0] - img_point[0])**2))
                total_error += error
            mean_error = total_error / len(corners[img_idx])


            # 创建一个更大尺寸的图像，以同时容纳顶部标题和底部图例
            h_img, w_img = img.shape[:2]
            top_margin = 80  # 顶部预留80像素给标题
            bottom_margin = 150  # 底部预留150像素给图例
            canvas = np.ones((h_img + top_margin + bottom_margin, w_img, 3), dtype=np.uint8) * 255
            # 将原始图像放置在预留顶部空间之后的位置
            canvas[top_margin:top_margin + h_img, :w_img] = img

            # 绘制原始检测角点（绿色）- 注意y坐标要加上top_margin偏移
            for j, corner in enumerate(corners[img_idx]):
                cv2.circle(canvas, (int(corner[0][0]), int(corner[0][1]) + top_margin), 12, (0, 255, 0), -1)
                cv2.putText(canvas, f"{detected_ids[j]}",
                            (int(corner[0][0]) + 15, int(corner[0][1]) + top_margin),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

            # 绘制重投影点（红色）- 同样要加上top_margin偏移
            for j, point in enumerate(img_points):
                cv2.circle(canvas, (int(point[0][0]), int(point[0][1]) + top_margin), 8, (0, 0, 255), -1)

            # 添加图例（底部位置要调整）
            cv2.rectangle(canvas, (50, h_img + top_margin + 30), (w_img - 50, h_img + top_margin + 120),
                          (220, 220, 220), -1)
            cv2.putText(canvas, "Legend:", (60, h_img + top_margin + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # 原始检测点图例
            cv2.circle(canvas, (180, h_img + top_margin + 55), 12, (0, 255, 0), -1)
            cv2.putText(canvas, "Original Detected Corners", (200, h_img + top_margin + 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2)

            # 重投影点图例
            cv2.circle(canvas, (450, h_img + top_margin + 55), 8, (0, 0, 255), -1)
            cv2.putText(canvas, "Reprojected Corners", (470, h_img + top_margin + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 2)

            # 添加重投影误差信息
            cv2.putText(canvas, f"Mean Reprojection Error: {mean_error:.4f} pixels",
                        (60, h_img + top_margin + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # 添加标题背景（现在在真正的顶部）
            cv2.rectangle(canvas, (50, 10), (w_img - 50, 70), (220, 220, 220), -1)

            # 标题文字
            title_text = f"{camera_name} Camera Reprojection Error Plot"
            (title_w, title_h), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            title_x = max(60, min(w_img // 2 - title_w // 2, w_img - title_w - 60))
            cv2.putText(canvas, title_text, (title_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            
            # 保存图像
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(os.path.join(save_dir, f"{camera_name}_reprojection.jpg"), canvas)
    
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_errors

def plot_reprojection_errors(left_errors, right_errors, save_dir, filename="reprojection_errors.png"):
    """
    绘制左右相机的重投影误差柱状图
    """
    # 确保两个列表长度相同 - 取较短的那个长度
    min_len = min(len(left_errors), len(right_errors))
    left_errors = left_errors[:min_len]
    right_errors = right_errors[:min_len]
    
    # 图像序号
    image_indices = np.arange(1, min_len + 1)
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 设置图表
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    
    # 绘制柱状图
    bars1 = plt.bar(image_indices - bar_width/2, left_errors, bar_width, color='blue', label='左相机')
    bars2 = plt.bar(image_indices + bar_width/2, right_errors, bar_width, color='red', label='右相机')
    
    # 添加标题和标签
    plt.title('左右相机各图像重投影误差对比', fontsize=16)
    plt.xlabel('图像序号', fontsize=14)
    plt.ylabel('重投影误差 (像素)', fontsize=14)
    plt.xticks(image_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 保存图表
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"重投影误差柱状图已保存到 {os.path.join(save_dir, filename)}")

# 双目标定
def stereo_calibration(left_corners, left_ids, right_corners, right_ids, left_camera_matrix, 
                      left_dist_coeffs, right_camera_matrix, right_dist_coeffs, board, image_size, 
                      stereo_mode=0, distortion_mode=0):
    # 创建对象点
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    
    # 提取成对的角点
    for i in range(len(left_corners)):
        # 确保两幅图像中检测到的角点ID匹配
        if len(left_ids[i]) > 4 and len(right_ids[i]) > 4:
            # 获取所有角点的3D坐标
            all_corners3d = board.getChessboardCorners()
            
            # 提取当前帧中检测到的角点ID
            left_ids_flat = left_ids[i].flatten()
            right_ids_flat = right_ids[i].flatten()
            
            # 找出两幅图像中共同的角点ID
            common_ids = np.intersect1d(left_ids_flat, right_ids_flat)
            
            if len(common_ids) > 12:  # 确保有足够的共同点
                # 根据共同ID提取3D坐标点
                objp = []
                imgp_left = []
                imgp_right = []
                
                for id_val in common_ids:
                    # 添加3D点
                    objp.append(all_corners3d[id_val])
                    
                    # 找出左图中对应ID的角点
                    left_idx = np.where(left_ids_flat == id_val)[0][0]
                    imgp_left.append(left_corners[i][left_idx][0])
                    
                    # 找出右图中对应ID的角点
                    right_idx = np.where(right_ids_flat == id_val)[0][0]
                    imgp_right.append(right_corners[i][right_idx][0])
                
                objpoints.append(np.array(objp, dtype=np.float32))
                imgpoints_left.append(np.array(imgp_left, dtype=np.float32))
                imgpoints_right.append(np.array(imgp_right, dtype=np.float32))
    
    # 设置标定模式与畸变模型对应的标志
    if stereo_mode == 0:
        # 固定内参
        flags = cv2.CALIB_FIX_INTRINSIC
        print("使用固定内参模式进行立体标定")
    elif stereo_mode == 1:
        # 优化内参
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        print("使用优化内参模式进行立体标定")
    elif stereo_mode == 2:
        # 优化内参且保持左右相机焦距相同
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_SAME_FOCAL_LENGTH
        print("使用优化内参+相同焦距模式进行立体标定")
    
    # 根据畸变模式添加对应的标志
    if distortion_mode == 1:
        # 仅计算k1,k2,p1,p2
        flags |= cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
        print("使用简化畸变模型(k1,k2,p1,p2)")
    elif distortion_mode == 2:
        # 仅计算k1,k2
        flags |= cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6 + cv2.CALIB_ZERO_TANGENT_DIST
        print("使用最简畸变模型(仅k1,k2)")
    elif distortion_mode == 3:
        # 假设没有畸变
        flags |= cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6 + cv2.CALIB_ZERO_TANGENT_DIST
        print("假设无畸变")
    else:
        print("使用完整畸变模型")
    
    # 执行立体标定
    ret, left_matrix, left_dist, right_matrix, right_dist, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, 
        left_camera_matrix, left_dist_coeffs, 
        right_camera_matrix, right_dist_coeffs, 
        image_size, flags=flags)
    
    # 返回标定结果，包括优化后的内参（如果允许优化）
    # return ret, left_matrix, left_dist, right_matrix, right_dist, R, T, E, F
    return ret, left_matrix, left_dist, right_matrix, right_dist, R, T, E, F, objpoints, imgpoints_left, imgpoints_right


# 立体校正
def stereo_rectify(left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, R, T, image_size):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_camera_matrix, left_dist_coeffs,
        right_camera_matrix, right_dist_coeffs,
        image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)  # 更正：alpha从-1改为0
    
    # 计算校正映射
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_camera_matrix, left_dist_coeffs, R1, P1, image_size, cv2.CV_32FC1)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_camera_matrix, right_dist_coeffs, R2, P2, image_size, cv2.CV_32FC1)
    
    return left_map1, left_map2, right_map1, right_map2, Q

# 保存标定得到的参数
def save_calibration_parameters(left_camera_matrix, left_dist_coeffs,
                               right_camera_matrix, right_dist_coeffs,
                               R, T, E, F, left_map1, left_map2, right_map1, right_map2, Q,
                               save_path="calibration_results"):

    # 创建保存目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 创建与原始代码相同格式的字典
    left = {'cameraMatrix': left_camera_matrix, 'distCoeffs': left_dist_coeffs}
    right = {'cameraMatrix': right_camera_matrix, 'distCoeffs': right_dist_coeffs}
    r = {'rotation': R}
    t = {'trans': T}
    
    # 保存参数，与camera_calibration模块格式一致
    np.save(os.path.join(save_path, 'calibration_1.npy'), left)
    np.save(os.path.join(save_path, 'calibration_2.npy'), right)
    np.save(os.path.join(save_path, 'R.npy'), r)
    np.save(os.path.join(save_path, 'T.npy'), t)
    
    # 保存额外参数
    np.save(os.path.join(save_path, 'E.npy'), {'essential': E})
    np.save(os.path.join(save_path, 'F.npy'), {'fundamental': F})
    np.save(os.path.join(save_path, 'Q.npy'), {'disparity_to_depth': Q})
    
    # 保存校正映射（可选）
    remap_data = {
        'left_map1': left_map1,
        'left_map2': left_map2,
        'right_map1': right_map1,
        'right_map2': right_map2
    }
    np.save(os.path.join(save_path, 'stereo_maps.npy'), remap_data)
    
    print(f"所有标定参数已保存到 {save_path} 目录")
    print(f"与calibration模块格式兼容")

    # 保存参数说明文件
    with open(os.path.join(save_path, "README.txt"), "w") as f:
        f.write("相机标定参数说明：\n")
        f.write("calibration_1.npy - 左相机内参矩阵与畸变系数\n")
        f.write("calibration_2.npy - 右相机内参矩阵与畸变系数\n")
        f.write("R.npy - 旋转矩阵\n")
        f.write("T.npy - 平移向量\n")
        f.write("E.npy - 本质矩阵\n")
        f.write("F.npy - 基础矩阵\n")
        f.write("Q.npy - 视差到深度映射矩阵\n")
        f.write("stereo_maps.npy - 立体校正映射\n")


def draw_coordinate_system(image,  camera_matrix, dist_coeffs, rvec, tvec, length=0.05):
    """
    在图像上绘制世界坐标系
    length: 坐标轴长度（米）
    """
    # 定义坐标系原点和三个轴的端点
    axis_length = length * 2.5  # 调整坐标轴长度
    points = np.float32([[0, 0, 0],  # 原点
                        [axis_length, 0, 0],  # X轴
                        [0, axis_length, 0],  # Y轴
                        [0, 0, axis_length]]) # Z轴
    
    # 定义标定板边框点（根据实际尺寸调整）
    board_width = 8 * 0.1  # 标定板宽度
    board_height = 7 * 0.1  # 标定板高度
    board_points = np.float32([[0, 0, 0],
                              [board_width, 0, 0],
                              [board_width, board_height, 0],
                              [0, board_height, 0]])
    
    # 将3D点投影到图像平面
    img_points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    board_img_points, _ = cv2.projectPoints(board_points, rvec, tvec, camera_matrix, dist_coeffs)
    
    img_points = img_points.astype(int)
    board_img_points = board_img_points.astype(int)
    
    # 转换为BGR图像以绘制彩色坐标轴
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 创建一个更大的画布，为标题和图例留出空间
    h, w = image.shape[:2]
    canvas = np.ones((h + 150, w, 3), dtype=np.uint8) * 255  # 减小底部空间
    canvas[100:h+100, :] = image
    
    # 绘制标定板边框
    cv2.polylines(canvas[100:h+100], [board_img_points], True, (128, 128, 128), 2)
    
    # 绘制坐标轴
    origin = tuple(img_points[0].ravel())
    origin = (origin[0], origin[1] + 100)  # 调整原点位置
    x_point = tuple(img_points[1].ravel())
    x_point = (x_point[0], x_point[1] + 100)
    y_point = tuple(img_points[2].ravel())
    y_point = (y_point[0], y_point[1] + 100)
    z_point = tuple(img_points[3].ravel())
    z_point = (z_point[0], z_point[1] + 100)
    
    # 绘制坐标轴线（使用更细的线条）
    cv2.line(canvas, origin, x_point, (0, 0, 255), 4)  # X轴 - 红色
    cv2.line(canvas, origin, y_point, (0, 255, 0), 4)  # Y轴 - 绿色
    cv2.line(canvas, origin, z_point, (255, 0, 0), 4)  # Z轴 - 蓝色
    
    # 在坐标轴端点绘制箭头
    arrow_length = 40  # 减小箭头长度
    arrow_angle = np.pi / 6
    for end_point, color in [(x_point, (0, 0, 255)), 
                           (y_point, (0, 255, 0)), 
                           (z_point, (255, 0, 0))]:
        # 计算方向向量
        dx = end_point[0] - origin[0]
        dy = end_point[1] - origin[1]
        norm = np.sqrt(dx*dx + dy*dy)
        if norm == 0: continue
        
        # 单位向量
        dx, dy = dx/norm, dy/norm
        
        # 计算箭头两个端点
        p1x = int(end_point[0] - arrow_length * (dx*np.cos(arrow_angle) + dy*np.sin(arrow_angle)))
        p1y = int(end_point[1] - arrow_length * (-dx*np.sin(arrow_angle) + dy*np.cos(arrow_angle)))
        p2x = int(end_point[0] - arrow_length * (dx*np.cos(arrow_angle) - dy*np.sin(arrow_angle)))
        p2y = int(end_point[1] - arrow_length * (dx*np.sin(arrow_angle) + dy*np.cos(arrow_angle)))
        
        cv2.line(canvas, end_point, (p1x, p1y), color, 4)  # 减小箭头线条粗细
        cv2.line(canvas, end_point, (p2x, p2y), color, 4)

    # 添加坐标轴标签（使用更小的字体）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0  # 减小字体大小
    thickness = 3  # 减小字体粗细
    offset = 40  # 减小标签偏移量

    # 使用 put_chinese_text 函数绘制中文标签
    cv2.putText(canvas, 'X', (x_point[0]+offset, x_point[1]), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(canvas, 'Y', (y_point[0]-offset, y_point[1]), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(canvas, 'Z', (z_point[0]-offset, z_point[1]-offset), font, font_scale, (255, 0, 0), thickness)
    
    # 添加标题
    title = "Charuco Board World Coordinate System"
    title_size = cv2.getTextSize(title, font, 1.5, 3)[0]  # 减小标题字体
    cv2.putText(canvas, title, (w//2 - title_size[0]//2, 60), font, 1.5, (0, 0, 0), 3)
    
    # 添加图例
    legend_x = 50
    legend_y = h + 130  # 调整图例位置
    legend_spacing = 250  # 减小图例间距
    
    # 绘制更小的图例
    cv2.line(canvas, (legend_x, legend_y), (legend_x + 60, legend_y), (0, 0, 255), 4)
    cv2.line(canvas, (legend_x + legend_spacing, legend_y), (legend_x + legend_spacing + 60, legend_y), (0, 255, 0), 4)
    cv2.line(canvas, (legend_x + 2*legend_spacing, legend_y), (legend_x + 2*legend_spacing + 60, legend_y), (255, 0, 0), 4)

    # 添加更小的图例文字
    legend_font_scale = 1.2  # 减小图例字体
    legend_thickness = 2  # 减小图例字体粗细
    cv2.putText(canvas, "X-axis", (legend_x + 80, legend_y + 10), font, legend_font_scale, (0, 0, 255), legend_thickness)
    cv2.putText(canvas, "Y-axis", (legend_x + legend_spacing + 80, legend_y + 10), font, legend_font_scale, (0, 255, 0), legend_thickness)
    cv2.putText(canvas, "Z-axis", (legend_x + 2*legend_spacing + 80, legend_y + 10), font, legend_font_scale, (255, 0, 0), legend_thickness)
    
    return canvas

def get_charuco_board_pose(corners, ids, board, camera_matrix, dist_coeffs):
    """
    获取Charuco标定板的位姿
    """
    if corners is None or ids is None:
        return None, None
    
    # 估计标定板的位姿
    ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        corners, ids, board, camera_matrix, dist_coeffs, None, None)
    
    if ret:
        return rvec, tvec
    return None, None


# ------- 主逻辑 -------
if __name__ == "__main__":
    # 创建与打印的标定板相同参数的board对象
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard((8, 7), 100, 75, dictionary)

    # 设置畸变模型和立体标定模式
    print("\n== 选择畸变模型 ==")
    print("[0] 完整畸变模型 (k1,k2,p1,p2,k3...)")
    print("[1] 简化畸变模型 (k1,k2,p1,p2)")
    print("[2] 最简畸变模型 (仅k1,k2)")
    print("[3] 无畸变 (假设相机没有畸变)")

    while True:
        try:
            distortion_mode = int(input("\n请选择畸变模型 [0-3]: "))
            if 0 <= distortion_mode <= 3:
                break
            else:
                print("输入超出范围，请重新输入")
        except ValueError:
            print("输入无效，请输入数字")

    print(f"已选择畸变模型: {['完整', '简化', '最简', '无畸变'][distortion_mode]}")

    print("\n== 选择立体标定模式 ==")
    print("[0] 固定内参 (使用单目标定结果作为最终内参)")
    print("[1] 优化内参 (在立体标定过程中优化内参)")
    print("[2] 优化内参 + 相同焦距 (优化内参，且保持左右相机焦距相同)")

    while True:
        try:
            stereo_mode = int(input("\n请选择立体标定模式 [0-2]: "))
            if 0 <= stereo_mode <= 2:
                break
            else:
                print("输入超出范围，请重新输入")
        except ValueError:
            print("输入无效，请输入数字")

    print(f"已选择立体标定模式: {['固定内参', '优化内参', '优化内参 + 相同焦距'][stereo_mode]}")

    # 1. 设置图像路径
    # 用于双目标定的图像
    left_image_folder = r"new_data5/cab/Camera_1"  # 左相机图像文件夹
    right_image_folder = r"new_data5/cab/Camera_0"  # 右相机图像文件夹
    # 用于各相机单独标定的图像
    left_image_folder_single = r"new_data5/cab/Camera_1"  # 左相机图像文件夹
    right_image_folder_single = r"new_data5/cab/Camera_0"  # 右相机图像文件夹

    calibration_save_path = r"new_data5/cab"  # 标定结果保存路径
    visualization_path = os.path.join(calibration_save_path, "visualization")

    if not os.path.exists(calibration_save_path):
        os.makedirs(calibration_save_path)

    # 获取用于双目标定的图像
    left_image_paths = sorted(glob.glob(os.path.join(left_image_folder, "*.bmp")))
    right_image_paths = sorted(glob.glob(os.path.join(right_image_folder, "*.bmp")))

    # 获取用于各相机单独标定的图像
    left_image_paths_single = sorted(glob.glob(os.path.join(left_image_folder_single, "*.bmp")))
    print(left_image_paths_single)
    right_image_paths_single = sorted(glob.glob(os.path.join(right_image_folder_single, "*.bmp")))

    # 2. 检测角点并保存检测图
    print("\n下面进行单目标定...")
    print("正在检测左相机图像角点...")
    left_corners, left_ids, left_images = detect_charuco_corners(left_image_paths_single, board, visualization_path)
    print(f"成功检测到 {len(left_corners)} 幅左相机图像的角点")

    print("正在检测右相机图像角点...")
    right_corners, right_ids, right_images = detect_charuco_corners(right_image_paths_single, board, visualization_path)
    print(f"成功检测到 {len(right_corners)} 幅右相机图像的角点")

    # 3. 单目标定并保存重投影误差图
    print("\n正在进行左相机标定...")
    if len(left_corners)>0:
        ret_left, left_camera_matrix, left_dist_coeffs, rvecs_left, tvecs_left, left_errors = calibrate_single_camera(
            left_corners, left_ids, left_images, board, visualization_path, "left", distortion_mode)
        print(f"左相机标定完成，重投影误差: {ret_left}")
    else:
        print("左相机数据不足，无法标定")

    print("\n正在进行右相机标定...")
    if len(right_corners)>0:
        ret_right, right_camera_matrix, right_dist_coeffs, rvecs_right, tvecs_right, right_errors = calibrate_single_camera(
            right_corners, right_ids, right_images, board, visualization_path, "right", distortion_mode)
        print(f"右相机标定完成，重投影误差: {ret_right}")
    else:
        print("右相机数据不足，无法标定")

    # 绘制左右相机重投影误差对比柱状图
    plot_reprojection_errors(left_errors, right_errors, visualization_path)

    # 4. 双目标定
    print("\n下面进行双目标定...")
    print("正在检测左相机图像角点...")
    left_corners, left_ids, left_images = detect_charuco_corners(left_image_paths, board, visualization_path)
    print(f"成功检测到 {len(left_corners)} 幅左相机图像的角点")

    print("正在检测右相机图像角点...")
    right_corners, right_ids, right_images = detect_charuco_corners(right_image_paths, board, visualization_path)
    print(f"成功检测到 {len(right_corners)} 幅右相机图像的角点")
    # 4. 双目标定
    print("\n正在进行双目标定...")
    if len(left_images) > 0:
        image_size = left_images[0].shape[::-1]
    else:
        # 如果没有检测到任何角点，使用第一张图像的尺寸
        test_img = cv2.imread(left_image_paths[0], cv2.IMREAD_GRAYSCALE)
        image_size = test_img.shape[::-1]

    # 5. 打印立体标定前的相机内参
    print("\n=== 立体标定前的相机内参 ===")
    print("左相机内参矩阵:")
    print(left_camera_matrix)
    print("\n左相机畸变系数:")
    print(left_dist_coeffs)
    print("\n右相机内参矩阵:")
    print(right_camera_matrix)
    print("\n右相机畸变系数:")
    print(right_dist_coeffs)

    ret, left_matrix, left_dist, right_matrix, right_dist, R, T, E, F, objpoints, imgpoints_left, imgpoints_right = stereo_calibration(
        left_corners, left_ids, right_corners, right_ids,
        left_camera_matrix, left_dist_coeffs,
        right_camera_matrix, right_dist_coeffs,
        board, image_size, stereo_mode, distortion_mode)

    # 6. 打印立体标定后的相机内参
    print("\n=== 立体标定后的相机内参 ===")
    print("左相机内参矩阵:")
    print(left_matrix)
    print("\n左相机畸变系数:")
    print(left_dist)
    print("\n右相机内参矩阵:")
    print(right_matrix)
    print("\n右相机畸变系数:")
    print(right_dist)

    print("双目标定完成")
    print("旋转矩阵R:", R)
    print("平移向量T:", T)

    # ---------------以下为新增------------------
    # 计算双目标定的误差
    stereo_errors, stereo_left_errors, stereo_right_errors, valid_pair_indices, stereo_overall_mean = \
        calculate_stereo_reprojection_errors(
            left_corners, left_ids,
            right_corners, right_ids,
            board,
            left_matrix, left_dist,
            right_matrix, right_dist,
            R, T,
            min_common_ids=12
        )

    print("\n=== 双目标定重投影误差 ===")
    if stereo_overall_mean is not None:
        print(f"双目标定整体平均重投影误差: {stereo_overall_mean:.4f} 像素")
        print(f"参与误差计算的有效图对数: {len(valid_pair_indices)}")
        # for idx, le, re, se in zip(valid_pair_indices, stereo_left_errors, stereo_right_errors, stereo_errors):
            # print(f"图对 {idx}: 左目={le:.4f} px, 右目={re:.4f} px, 双目平均={se:.4f} px")
    else:
        print("没有足够有效图对，无法计算双目标定重投影误差")

    plot_stereo_reprojection_errors(
        stereo_errors,
        stereo_left_errors,
        stereo_right_errors,
        valid_pair_indices,
        visualization_path
    )
    # ------------------------------------------

    # 7. 立体校正
    print("\n正在进行立体校正...")
    left_map1, left_map2, right_map1, right_map2, Q = stereo_rectify(
        left_matrix, left_dist,
        right_matrix, right_dist,
        R, T, image_size)
    print("立体校正完成")

    # 创建并保存校正前后左右相机拼接图
    if len(left_images) > 0 and len(right_images) > 0:
        print("正在生成校正前后拼接图...")

        # 获取第一张图像
        left_img = left_images[0]
        right_img = right_images[0]

        # 确保两张图像有相同尺寸
        if left_img.shape != right_img.shape:
            # 调整尺寸
            h, w = min(left_img.shape[0], right_img.shape[0]), min(left_img.shape[1], right_img.shape[1])
            left_img = left_img[:h, :w]
            right_img = right_img[:h, :w]

        # 转换为三通道图像用于彩色显示
        left_img_color = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        right_img_color = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

        # 校正前拼接图
        before_rectify = np.hstack((left_img_color, right_img_color))

        # 校正后的图像
        left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)

        # 转换为三通道图像
        left_rectified_color = cv2.cvtColor(left_rectified, cv2.COLOR_GRAY2BGR)
        right_rectified_color = cv2.cvtColor(right_rectified, cv2.COLOR_GRAY2BGR)

        # 校正后拼接图
        after_rectify = np.hstack((left_rectified_color, right_rectified_color))

        # 在图像上添加水平线，以便更清楚地看到校正效果
        line_interval = 100
        h, w = before_rectify.shape[:2]

        for i in range(0, h, line_interval):
            # 校正前图像添加水平线
            cv2.line(before_rectify, (0, i), (w, i), (0, 255, 0), 2)

            # 校正后图像添加水平线
            cv2.line(after_rectify, (0, i), (w, i), (0, 255, 0), 2)

        # 添加左右相机标识和校正前后标识
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 绘制中线
        cv2.line(before_rectify, (w//2, 0), (w//2, h), (255, 0, 0), 2)
        cv2.line(after_rectify, (w//2, 0), (w//2, h), (255, 0, 0), 2)

        # 保存拼接图像
        if not os.path.exists(visualization_path):
            os.makedirs(visualization_path)
        cv2.imwrite(os.path.join(visualization_path, "before_rectify.png"), before_rectify)
        cv2.imwrite(os.path.join(visualization_path, "after_rectify.png"), after_rectify)

        print(f"拼接图已保存到 {visualization_path} 目录")

    #对图像进行立体校正并保存校正结果
    for i in range(len(left_images)):
        left_image = left_images[i]
        right_image = right_images[i]
        left_undistorted = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
        right_undistorted = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(visualization_path, f"left_undistorted_{i}.jpg"), left_undistorted)
        cv2.imwrite(os.path.join(visualization_path, f"right_undistorted_{i}.jpg"), right_undistorted)
    # 6. 保存标定结果
    # 更正：这里保存的是立体标定前的参数，如果使用优化参数的模式，则会出错
    # save_calibration_parameters(
    #     left_camera_matrix, left_dist_coeffs,
    #     right_camera_matrix, right_dist_coeffs,
    #     R, T, E, F, left_map1, left_map2, right_map1, right_map2, Q, calibration_save_path)

    save_calibration_parameters(
        left_matrix, left_dist,
        right_matrix, right_dist,
        R, T, E, F, left_map1, left_map2, right_map1, right_map2, Q, calibration_save_path)

    # 在主程序中添加以下代码（在标定完成后）：
    print("正在绘制世界坐标系...")

    # 对第一张检测到角点的图像进行处理
    if len(left_corners) > 0 and len(left_ids) > 0:
        # 获取左相机第一张图像
        left_img = left_images[0]

        # 获取标定板位姿
        rvec, tvec = get_charuco_board_pose(
            left_corners[0], left_ids[0], board,
            left_camera_matrix, left_dist_coeffs)

        if rvec is not None and tvec is not None:
            # 绘制世界坐标系
            coord_img = draw_coordinate_system(
                left_img.copy(), left_camera_matrix,
                left_dist_coeffs, rvec, tvec, length=0.03)  # 减小坐标轴长度

            # 保存高分辨率图像
            cv2.imwrite(os.path.join(visualization_path, "world_coordinate_system.png"), coord_img,
                        [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 使用无损PNG格式
            print(f"世界坐标系可视化结果已保存到 {visualization_path}")
        else:
            print("无法估计标定板位姿")
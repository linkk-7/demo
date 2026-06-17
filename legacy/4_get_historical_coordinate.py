import os
from track import track, enhanced_track, enhanced_track_with_prediction
import logging
from lightglue import LightGlue, SuperPoint, DISK, SIFT
import numpy as np
import torch

# 在左、右立体校正后的图像序列中，分别追踪某一个指定 ID 的特征点，得到它在每一帧中的历史像素坐标，并保存成 .npy 文件。
# 从 new_data3\cab1\left_rec 和 new_data3\cab1\right_rec 读取校正后的左右图像序列，对编号为 29 的目标点分别进行逐帧跟踪，
# 并把左右相机中的历史像素坐标保存成 .npy 文件。

torch.set_grad_enabled(False)
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=1000).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)
'''
设定好特征提取器和匹配器，分别在左右图像序列中计算要追踪的点的像素历史坐标并保存为npy文件
'''

#root_path = r"E:\shuangmu shuju\20250919\3"
# 运动数据的总文件夹
root_path = r"new_data5\cab3"



images_list_l = [os.path.join(os.path.join(root_path, "left_rec"), file) for file in
                 os.listdir(os.path.join(root_path, "left_rec"))]
# 对文件名按照数字排序
images_list_l = sorted(images_list_l, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

images_list_r = [os.path.join(os.path.join(root_path, "right_rec"), file) for file in
                 os.listdir(os.path.join(root_path, "right_rec"))]

# 对文件名按照数字排序
images_list_r = sorted(images_list_r, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))


# 要追踪的初始点 ID 是 29
hisl,t_zhedang_l = enhanced_track_with_prediction(images_list_l, extractor, matcher, 333)
np.save(os.path.join(root_path, '左相机历史坐标-354.npy'), hisl)
# np.save(os.path.join(root_path, '左相机遮挡时间段.npy'), np.array(t_zhedang_l, dtype=object))
hisr,t_zhedang_r = enhanced_track_with_prediction(images_list_r, extractor, matcher, 336)
np.save(os.path.join(root_path, '右相机历史坐标-363.npy'), hisr)
# np.save(os.path.join(root_path, '右相机遮挡时间段.npy'), np.array(t_zhedang_r, dtype=object))
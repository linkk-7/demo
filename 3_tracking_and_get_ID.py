import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('Qt5Agg')   # 或 'TkAgg'，或 'Agg'（非互动）
import matplotlib.pyplot as plt


from get_init_id import get_init_id, get_init_id_aliked, get_init_id_sift

'''
对初始图片进行匹配，确认追踪点的ID
'''
#img0_path = r"E:\shuangmu shuju\20250930\1\left_rec\1l.jpg"
#img1_path = r"E:\shuangmu shuju\20250930\1\right_rec\1r.jpg"
#save_dir = r"E:\shuangmu shuju\20250930\1"

# 左右校正后的第一对图像
img0_path = r"new_data5\cab3\left_rec\1l.jpg"
img1_path = r"new_data5\cab3\right_rec\1r.jpg"
save_dir = r"new_data5\cab3"

get_init_id(img0_path, img1_path, save_dir)
# get_init_id_sift(img0_path, img1_path, save_dir)
# get_init_id_aliked(img0_path, img1_path, save_dir)

"""
旧离线批处理立体校正入口（legacy/offline）。

说明：
1. 本脚本用于读取目录中的左右图像并批量执行立体校正。
2. 本脚本不属于当前在线主流程。
3. 当前在线主流程应使用：
   StereoRectifier.rectify_frame_packet(...)
"""

import os

from stereo_rectify import stereo_rectification


# -------------------- 配置区（离线批处理） --------------------
# 历史数据目录（示例：包含 Camera_0 / Camera_1 子目录）
DATA_ROOT = r"new_data5/cab3"

# 输入目录
LEFT_DIR = os.path.join(DATA_ROOT, "Camera_0")
RIGHT_DIR = os.path.join(DATA_ROOT, "Camera_1")

# 输出目录（校正后）
LEFT_RECT_DIR = os.path.join(DATA_ROOT, "left_rec")
RIGHT_RECT_DIR = os.path.join(DATA_ROOT, "right_rec")

# 标定参数目录（应包含 calibration_1.npy / calibration_2.npy / R.npy / T.npy）
CALIBRATION_FOLDER = r"new_data5/cab"

CALIBRATION_FILE_1 = os.path.join(CALIBRATION_FOLDER, "calibration_1.npy")
CALIBRATION_FILE_2 = os.path.join(CALIBRATION_FOLDER, "calibration_2.npy")
R_FILE = os.path.join(CALIBRATION_FOLDER, "R.npy")
T_FILE = os.path.join(CALIBRATION_FOLDER, "T.npy")

MODE = "python"


def main():
    stereo_rectification(
        input_folder1=LEFT_DIR,
        input_folder2=RIGHT_DIR,
        output_folder1=LEFT_RECT_DIR,
        output_folder2=RIGHT_RECT_DIR,
        calibration_file1=CALIBRATION_FILE_1,
        calibration_file2=CALIBRATION_FILE_2,
        r=R_FILE,
        t=T_FILE,
        mode=MODE,
    )


if __name__ == "__main__":
    main()

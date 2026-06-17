"""
相机标定
"""
import cv2
import os
import numpy as np
import yaml

from typing import Any, Dict, List, Optional, Tuple, Sequence
from cv2.typing import MatLike
from pydantic import BaseModel, Field

from utils.file_utils import make_new_dir
from utils.ini_utils import load_config
from utils.num_utils import numerical_sort
from utils.time_utils import get_current_time_str


def calibrate_camera(object_points: Sequence[MatLike], corners: Sequence[MatLike], imgsize: Tuple[int, int]):
    """
    相机标定
    """
    ret = cv2.calibrateCamera(object_points, corners, imgsize, None, None)
    return ret

# 立体矫正和显示
def stereo_rectify_and_display(output_name, cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, img_size, object_points, corners_left, corners_right) -> str:
    print("img_size",img_size)
    print("camera_matrix_l",cameraMatrix_l)
    print("camera_matrix_r",cameraMatrix_r)

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.0001)
    (ret_stereo, newcameramtx_left, dist_left, newcameramtx_right, dist_right, R, T, E, F,) = cv2.stereoCalibrate(object_points, corners_left, corners_right,
                                    cameraMatrix_l, distCoeffs_l,
                                    cameraMatrix_r, distCoeffs_r,
                                    img_size, criteria=criteria_stereo,
                                    flags=flags
                                    )


    # 立体校正
    R1, R2, P1, P2, Q, _, _ = (cv2.stereoRectify(newcameramtx_left, dist_left, newcameramtx_right, dist_right, img_size, R, T, 1, (0,0),))

    print("P1",P1)
    print("P2",P2)

    # Extract adjusted intrinsic parameters from P1 and P2
    fx = P1[0, 0]
    fy = P1[1, 1]
    cx = P1[0, 2]
    cy = P1[1, 2]

    # Calculate baseline b from the translation vector T
    B = np.linalg.norm(T)  # or use b = np.abs(T[0]) for horizontal baseline
    print(f"Baseline (b): {B}")

    print(f"Adjusted focal length in pixels after rectification: fx = {fx}, fy = {fy}")
    print(f"Adjusted principal point after rectification: cx = {cx}, cy = {cy}")

    save_calibration_to_yaml(f'./{output_name}', newcameramtx_left, dist_left, newcameramtx_right, dist_right, P1, P2, R1, R2, R, T, E, F, B)
    return f"两摄像头焦距距离为{B:.2f}mm，焦距分别为{fx:.2f}mm和{fy:.2f}mm"


class LoadImagesOutput(BaseModel):
    imgs: Optional[List[Tuple[MatLike, MatLike]]] = Field(default=[])
    status: bool = Field(default=True)
    fail_photo_indexs: Optional[List[int]] = Field(default=[])  #失败的相片目录
    class Config:
        arbitrary_types_allowed = True  # 允许任意类型


def load_images(idxs: List[int], image_paths: List[str]) -> LoadImagesOutput:
    """
    加载摄像头图片文件夹并将里面的彩图转换为灰度图
    """
    res = LoadImagesOutput()
    img_list = []
    error_list = []
    for i in range(len(image_paths)):
        img_path = image_paths[i]
        frame = cv2.imread(img_path)
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_list.append((frame, gray))
        else:
            print(f"无法读取图像: {img_path}")
            error_list.append(idxs[i])
    if len(error_list) != 0:
        res.status = False
        res.fail_photo_indexs = error_list
        return res
    res.imgs = img_list
    return res


class GetCornersOutput(BaseModel):
    corners: Optional[List[Any]] = Field(default=[])
    status: bool = Field(default=True)
    fail_photo_indexs: Optional[List[int]] = Field(default=[])  #失败的相片目录


def get_corners(idxs, imgs, pattern_size, folder) -> GetCornersOutput:
    """
    检测棋盘格角点
    """
    make_new_dir(folder, True)
    corners = []
    count = 0
    error_list = []
    for frame, gray in imgs:
        ret, c = cv2.findChessboardCorners(gray, pattern_size)     #ret 表示是否成功找到棋盘格角点，c 是一个数组，包含了检测到的角点的坐标
        if not ret:
            print("未能检测到棋盘格角点", idxs[count])
            error_list.append(idxs[count])
            count = count + 1
            continue
        c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))     #cv2.cornerSubPix 函数用于提高棋盘格角点的精确度，对初始检测到的角点坐标 c 进行优化
        corners.append(c)      #将优化后的角点坐标 c 添加到 corners 列表中

        # 绘制角点并显示
        vis = frame.copy()
        cv2.drawChessboardCorners(vis, pattern_size, c, ret)
        cv2.imwrite(folder+f'{count}.png',vis)
        count = count + 1
    res = GetCornersOutput()
    if len(error_list) > 0:
        res.status = False
        res.fail_photo_indexs = error_list
        return res
    res.corners = corners
    return res


def save_calibration_to_yaml(file_path, cameraMatrix_l: MatLike, distCoeffs_l: MatLike, cameraMatrix_r: MatLike, distCoeffs_r: MatLike, P1: MatLike, P2: MatLike, R1, R2, R, T, E, F, B):
    data = {
        'camera_matrix_left': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': cameraMatrix_l.flatten().tolist()
        },
        'dist_coeff_left': {
            'rows': 1,
            'cols': 5,
            'dt': 'd',
            'data': distCoeffs_l.flatten().tolist()
        },
        'camera_matrix_right': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': cameraMatrix_r.flatten().tolist()
        },
        'dist_coeff_right': {
            'rows': 1,
            'cols': 5,
            'dt': 'd',
            'data': distCoeffs_r.flatten().tolist()
        },
        'P1': {
            'rows': 3,
            'cols': 4,
            'dt': 'd',
            'data': P1.flatten().tolist()
        },
        'P2': {
            'rows': 3,
            'cols': 4,
            'dt': 'd',
            'data': P2.flatten().tolist()
        },
        'R1': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': R1.flatten().tolist()
        },
        'R2': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': R2.flatten().tolist()
        },
        'R': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': R.flatten().tolist()
        },
        'T': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': T.flatten().tolist()
        },
        'E': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': E.flatten().tolist()
        },
        'F': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': F.flatten().tolist()
        },
        'fx': {
            'rows': 1,
            'cols': 1,
            'dt': 'd',
            'data': P1[0, 0].tolist()
        },
        'fy': {
            'rows': 1,
            'cols': 1,
            'dt': 'd',
            'data': P1[1, 1].tolist()
        },
        'cx': {
            'rows': 1,
            'cols': 1,
            'dt': 'd',
            'data': P1[0, 2].tolist()
        },
        'cy': {
            'rows': 1,
            'cols': 1,
            'dt': 'd',
            'data': P1[1, 2].tolist()
        },
        'Baseline': {
            'rows': 1,
            'cols': 1,
            'dt': 'd',
            'data': B.tolist()
        }
        
    }

    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    print(f"Calibration parameters saved to {file_path}")

class MakeCalibrationInput(BaseModel):
    left_images_folder: Optional[str] = Field(default=None) #左侧图片标定原始数据所在文件夹
    right_images_folder: Optional[str] = Field(default=None) #右侧图片标定原始数据所在文件夹
    pattern_size: Optional[Tuple[int, int]] = Field(default=None) #标定使用的棋盘格尺寸
    test_root_folder: Optional[str] = Field(default=None)   #测试数据存储的根文件夹
    actual_size: Optional[float] = Field(default=None)  #棋盘格的实际尺寸 单位：mm  actual size of chessboard cell  unit：mm
    output_name: Optional[str] = Field(default=None)    #输出文件名


class MakeCalibrationOutput(BaseModel):
    fail_photo_indexs: Optional[List[int]] = Field(default=[])  #失败的相片目录
    result_str: Optional[str] = Field(default=None)  #最后评价
    status: bool = Field(default=True)  #是否成功
    paramId: Optional[int] = Field(default=None) 
    type: str = Field(default="calibration")



def find_unique_elements(a, b):
    """
    找到两个数组中对方不存在的元素。

    :param a: 第一个数组
    :param b: 第二个数组
    :return: 包含两个数组中对方不存在的元素的列表
    """
    # 将数组转换为集合
    set_a = set(a)
    set_b = set(b)

    # 找到 a 中有但 b 中没有的元素，以及 b 中有但 a 中没有的元素
    unique_in_a = set_a - set_b
    unique_in_b = set_b - set_a

    # 合并结果并返回
    return list(unique_in_a) + list(unique_in_b)


def make_calibration(input_data: MakeCalibrationInput) -> MakeCalibrationOutput:
    """
    执行标定过程
    """
    # 获取图像文件名称列表并排序
    left_image_paths = sorted([os.path.join(input_data.left_images_folder, f) for f in os.listdir(input_data.left_images_folder)], key= lambda x:numerical_sort(x, 0))
    right_image_paths = sorted([os.path.join(input_data.right_images_folder, f) for f in os.listdir(input_data.right_images_folder)], key= lambda x:numerical_sort(x, 0))
    idxs_left = sorted([numerical_sort(i, -2) for i in left_image_paths], key= lambda x:x)
    idxs_right = sorted([numerical_sort(i, -2) for i in right_image_paths], key= lambda x:x)
    res = MakeCalibrationOutput()

    print("idxs_left", idxs_left, "left_image_paths", left_image_paths)

    error_list = find_unique_elements(idxs_left, idxs_right)
    if len(error_list) != 0:
        res.fail_photo_indexs = error_list
        res.status = False
        res.result_str = "左右两个画面的图像数量不匹配，不匹配的标号如下，请删除如下标定图片后再重新标定"
        return res
    left_images_res: LoadImagesOutput = load_images(idxs_left, left_image_paths)
    right_images_res: LoadImagesOutput = load_images(idxs_right, right_image_paths)

    if not left_images_res.status or not right_images_res:
        res.fail_photo_indexs = left_images_res.fail_photo_indexs + right_images_res.fail_photo_indexs
        res.status = False
        res.result_str = "有图像无法加载，无法加载的图像标号如下，请删除如下标定图片后再重新标定"
        return res

    current_time = get_current_time_str()

    corners_left_res = get_corners(idxs_left, left_images_res.imgs, input_data.pattern_size, input_data.test_root_folder + "/" + current_time + "_left/")       #corners_left的长度表示检测到棋盘格角点的图像数量。corners_left[i] 和 corners_right_res.corners[i] 中存储了第 i 张图像检测到的棋盘格角点的二维坐标。
    corners_right_res = get_corners(idxs_right, right_images_res.imgs, input_data.pattern_size,  input_data.test_root_folder + "/" + current_time + "_right/")


    if not corners_left_res.status or not corners_right_res.status:
        res.fail_photo_indexs = corners_left_res.fail_photo_indexs + corners_right_res.fail_photo_indexs
        res.status = False
        res.result_str = "有图像无法检测到角点，无法检测的图像标号如下，请删除如下标定图片后再重新标定"
        return res

    # 准备标定所需数据
    points = np.zeros((input_data.pattern_size[0] * input_data.pattern_size[1], 3), dtype=np.float32)   #创建零矩阵，用于存储棋盘格的三维坐标点。棋盘格的大小是 11 行 8 列，88 个内角点。数据类型为 np.float32，这是一张图的，因为一个角点对应一个三维坐标
    points[:, :2] = np.mgrid[0:input_data.pattern_size[0], 0:input_data.pattern_size[1]].T.reshape(-1, 2) * input_data.actual_size  #给这些点赋予实际的物理坐标，* 24 是因为每个棋盘格的大小为 24mm

    object_points = [points] * len(left_images_res.imgs)     #包含了所有图像中棋盘格的三维物理坐标点 points。这里假设所有图像中棋盘格的物理坐标是相同的，因此用 points 复制 len(corners_left_res.corners) 次。
    imgsize = left_images_res.imgs[0][1].shape[::-1]     #img_list_left[0] 是左相机图像列表中的第一张图像。img_list_left[0][1] 是该图像的灰度图像。shape[::-1] 取灰度图像的宽度和高度，并反转顺序，以符合 calibrateCamera 函数的要求。

    print('开始左相机标定')
    _, cameraMatrix_l, distCoeffs_l, _, _ = calibrate_camera(object_points, corners_left_res.corners, imgsize)    #object_points表示标定板上检测到的棋盘格角点的三维坐标；corners_left[i]表示棋盘格角点在图像中的二维坐标；imgsize表示图像大小
    print("distCoeffs_l", distCoeffs_l, distCoeffs_l.shape)
    distCoeffs_l[0, 0] = 0  # k1
    distCoeffs_l[0, 1] = 0  # k2
    distCoeffs_l[0, 4] = 0  # k3

    # distCoeffs_l[0, 2] = 0  # k1
    # distCoeffs_l[0, 3] = 0  # k1
    print('开始右相机标定')
    _, cameraMatrix_r, distCoeffs_r, _, _ = calibrate_camera(object_points, corners_right_res.corners, imgsize)

    distCoeffs_r[0, 0] = 0  # k1
    distCoeffs_r[0, 1] = 0  # k2
    distCoeffs_r[0, 4] = 0  # k3 

    # distCoeffs_r[0, 2] = 0  # k1
    # distCoeffs_r[0, 3] = 0  # k1
    print("distCoeffs_r", distCoeffs_r, distCoeffs_r.shape)
    result_str = stereo_rectify_and_display(input_data.output_name, cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, imgsize, object_points, corners_left_res.corners, corners_right_res.corners)
    res.fail_photo_indexs = []
    res.result_str = result_str
    res.status = True
    return res

def count_param_vactor_by_pos(a:int, b: int, c: int, d: int, px1: float, py1: float, pz1: float, px2: float, py2: float, pz2: float ):
    t = c*b-a*d
    res_x = (c*px1 - a*px2)/t
    res_y = (c*py1 - a*py2)/t
    res_z = (c*pz1 - a*pz2)/t
    res = {}
    res["x"] = res_x
    res["y"] = res_y
    res["z"] = res_z
    return res
    
if __name__ == "__main__":
    pass
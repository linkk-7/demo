# -*- coding: utf-8 -*-
"""
图像超分放大单输出
"""
import cv2
from cv2 import dnn_superres
from cv2.typing import MatLike
def get_super_pixel_image(img: MatLike, scale: float, algorithm: str = "bilinear"):
    """
    可选择算法，bilinear, bicubic, edsr, espcn, fsrcnn or lapsrn

    放大比例，可输入值2，3，4
    """
    # 模型路径
    path = "./model/EDSR_x2.pb"

    # 创建模型
    sr = dnn_superres.DnnSuperResImpl_create()
    if algorithm == "bilinear":
        img_new = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    elif algorithm == "bicubic":
        img_new = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif algorithm == "edsr" or algorithm == "espcn" or algorithm == "fsrcnn" or algorithm == "lapsrn":
        # 读取模型
        sr.readModel(path)
        #  设定算法和放大比例
        sr.setModel(algorithm, scale)
        # 放大图像
        img_new = sr.upsample(img)
    else:
        print("Algorithm not recognized")
    # 如果失败
    if img_new is None:
        print("Upsampling failed")
    return img_new

if __name__ == '__main__':
    get_super_pixel_image()

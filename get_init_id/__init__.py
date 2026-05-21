import os
import re
from datetime import datetime
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
# 设置工作目录为下载项目的文件夹路径
import sys
sys.path.append(r"E:\研究生\西安碑林\西安碑林双目相机位移监测系统_第二版\LightGlue-main")
# os.chdir(r"E:\研究生学习\研究方向\双目视觉\repo\LightGlue-main")
from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK,SIFT,ALIKED
from lightglue.utils import load_image, load_image1,rbd
from lightglue import viz2d
import torch
import time



def get_init_id1(img0_path,img1_path,img00_path,img01_path):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #初始帧追踪点选择
    '''
                关键点提取器与匹配器的导入
    '''
    extractor = SuperPoint(max_num_keypoints=1000).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    image0 = load_image1(img0_path)
    img00 = load_image1(img00_path)
    img01 = load_image1(img01_path)
    image1 = load_image1(img1_path)
    #检测特征点
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    #匹配两张图中的特征点对
    matches01 = matcher({"image0": feats0, "image1": feats1})
    #去除批量的那个维度
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    print(matches)

    #设置绘图参数
    #画匹配
    axes = viz2d.plot_images([img00, img01])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    #画检测到的特征点
    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([img00, img01])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    plt.show()

def get_init_id(img0_path,img1_path,save_dir):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #初始帧追踪点选择
    '''
                关键点提取器与匹配器的导入
    '''
    extractor = SuperPoint(max_num_keypoints=1000).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    image0 = load_image1(img0_path)
    image1 = load_image1(img1_path)
    #检测特征点
    #检测特征点
    start_time = time.time()
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    end_time = time.time()
    detection_time = end_time - start_time
    print(f"特征点检测用时: {detection_time:.4f} 秒")
    #匹配两张图中的特征点对
    matches01 = matcher({"image0": feats0, "image1": feats1})
    #去除批量的那个维度
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    print(matches)

    #设置绘图参数
    #画匹配
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    #viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    plt.savefig(os.path.join(save_dir, 'matches.png'), dpi=300, bbox_inches='tight')

    #画检测到的特征点
    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    plt.show()
    return kpts0,kpts1

def get_init_id_aliked(img0_path,img1_path,save_dir):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #初始帧追踪点选择
    '''
                关键点提取器与匹配器的导入
    '''
    extractor = ALIKED(max_num_keypoints=500).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)

    image0 = load_image1(img0_path)
    image1 = load_image1(img1_path)
    #检测特征点
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    #匹配两张图中的特征点对
    matches01 = matcher({"image0": feats0, "image1": feats1})
    #去除批量的那个维度
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    print(matches)

    #设置绘图参数
    #画匹配
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    #viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    plt.savefig(os.path.join(save_dir, 'matches.png'), dpi=300, bbox_inches='tight')

    #画检测到的特征点
    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    plt.show()
    return kpts0,kpts1

def get_init_id_sift(img0_path,img1_path,save_dir):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #初始帧追踪点选择
    '''
                关键点提取器与匹配器的导入
    '''
    extractor = SIFT(max_num_keypoints=500).eval().to(device)
    matcher = LightGlue(features="sift").eval().to(device)

    image0 = load_image1(img0_path)
    image1 = load_image1(img1_path)
    #检测特征点
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    #匹配两张图中的特征点对
    matches01 = matcher({"image0": feats0, "image1": feats1})
    #去除批量的那个维度
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    print(matches)

    #设置绘图参数
    #画匹配
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    #viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    plt.savefig(os.path.join(save_dir, 'matches.png'), dpi=300, bbox_inches='tight')

    #画检测到的特征点
    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    plt.show()
    return kpts0,kpts1
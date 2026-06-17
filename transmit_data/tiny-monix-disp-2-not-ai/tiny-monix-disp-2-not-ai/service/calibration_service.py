# 标定相关函数

import socket
import os
import threading
import time
from typing import Optional
from compress_image import compress_images
import cv2
from normal_generate import NormalGenTestInput, normal_gen_test
from pydantic import BaseModel, Field
from PIL import Image
import base64

from datas.camera import Camera
from datas.recv_message import RecvData
from rectify_params import get_recify_picture
from utils.byte_utils import get_length_prefix_bytes
from utils.calibration_utils import MakeCalibrationInput, count_param_vactor_by_pos, make_calibration
from utils.file_utils import delete_file, delete_files_by_prefix, get_file_names_in_folder, make_new_dir
from utils.ini_utils import load_config
from utils.num_utils import numerical_sort
from utils.real_camera_utils import get_current_camera_params
from utils.time_utils import get_current_timestamp
from utils.video_utils import check_rtsp_stream, extract_keyframes, record_rtsp_stream

from dao.camera_dao import cameraDao
from dao.monitor_dao import monitorDao
import math


# =================================start函数所需类定义=======================================================

# generate_calibration_png函数的入参
class GenerateCalibrationPngInput(BaseModel):
    mp4_file_name: Optional[str] = Field(default=None)
    png_file_name: Optional[str] = Field(default=None)
    rtsp_url1: Optional[str] = Field(default=None)
    rtsp_url2: Optional[str] = Field(default=None)
    left_output_dir: Optional[str] = Field(default=None)
    right_output_dir: Optional[str] = Field(default=None)
    left_compress_dir: Optional[str] = Field(default=None)
    right_compress_dir: Optional[str] = Field(default=None)
    duration: Optional[float] = Field(default=0.5)
    keyframe_interval: Optional[int] = Field(default=10)
    start_index: Optional[int] = Field(default=2)
    only_one: bool = False
    need_compress: bool = Field(default=False)

# generate_calibration_png函数的出参
class GenerateCalibrationPngOutput(BaseModel):
    status: bool = False   #拍摄状态
    timestamp: Optional[int] = Field(default=None)  #拍摄时刻的时间戳
    left_compress_path: Optional[str] = Field(default=None)   #压缩后左侧图片的路径
    right_compress_path: Optional[str] = Field(default=None)  #压缩后右侧图片的路径

    file_name: Optional[str] = Field(default=None)

# ===================================end函数所需类定义=======================================================

def generate_calibration_png(input_data: GenerateCalibrationPngInput) -> GenerateCalibrationPngOutput:
    """
    通过两个视频流生成标定图片
    file_name:        过程文件名称
    rtsp_url1:        rtsp流1
    rtsp_url2:        rtsp流2
    left_output_dir:  左侧摄像头画面文件夹
    right_output_dir: 右侧摄像头画面文件夹
    duration: 视频时长
    keyframe_interval: 关键帧间隔
    start_index: 舍弃start_index之前的图片，比如start_index = 2，则frame_001.png将被舍弃
    only_one: 无论什么情况，都只保留一张图片
    """
    #如果没有rtsp流，则不进行视频提取
    if input_data.rtsp_url1 == None or input_data.rtsp_url2 == None:
        return GenerateCalibrationPngOutput() 
    url1_vaild = check_rtsp_stream(input_data.rtsp_url1)
    url2_vaild = check_rtsp_stream(input_data.rtsp_url2)
    if (not url1_vaild) or (not url2_vaild):
        #如果有rtsp流无效，则不进行视频提取
        return GenerateCalibrationPngOutput() 
    if input_data.only_one:
        tmp_duration = 0.5
    else:
        tmp_duration = input_data.duration


    # 可以不修改
    output_file1 = f'{input_data.mp4_file_name}_1.mp4'
    output_file2 = f'{input_data.mp4_file_name}_2.mp4'

    delete_file(output_file1)
    delete_file(output_file2)

    # 创建线程
    thread1 = threading.Thread(target=record_rtsp_stream, args=(input_data.rtsp_url1, output_file1, tmp_duration, input_data.keyframe_interval))
    thread2 = threading.Thread(target=record_rtsp_stream, args=(input_data.rtsp_url2, output_file2, tmp_duration, input_data.keyframe_interval))

    # 启动线程
    thread1.start()
    thread2.start()
    start_time = get_current_timestamp()
    

    # 等待两个线程都执行完毕
    thread1.join()
    thread2.join()

    output_pattern = input_data.png_file_name + '__%03d.png'  # 输出图片文件的命名模式
    left_output_file = input_data.left_output_dir + output_pattern
    right_output_file = input_data.right_output_dir + output_pattern

    #创建文件夹
    make_new_dir(input_data.left_output_dir, False)
    make_new_dir(input_data.right_output_dir, False)

    extract_keyframes(output_file1, left_output_file)
    extract_keyframes(output_file2, right_output_file)

    #起始状态下由于丢帧图片会有全灰色，所以需要去除
    if input_data.start_index >= 2:
        for index in range(1, input_data.start_index):
            left_delete_file_name = input_data.left_output_dir + f'{input_data.png_file_name}__{index:03d}.png'
            right_delete_file_name = input_data.right_output_dir + f'{input_data.png_file_name}__{index:03d}.png'
        if os.path.exists(left_delete_file_name):
            os.remove(left_delete_file_name)
        if os.path.exists(right_delete_file_name):
            os.remove(right_delete_file_name)
    
    #删除多余的不匹配的图片
    left_file_names = get_file_names_in_folder(input_data.left_output_dir)
    left_file_names_set = set(left_file_names)

    right_file_names = get_file_names_in_folder(input_data.right_output_dir)
    right_file_names_set = set(right_file_names)

    common_files = left_file_names_set & right_file_names_set

    for left_file_name in left_file_names:
        if left_file_name not in common_files and (len(left_file_name.split("__")) > 0 and left_file_name.split("__")[0] == input_data.png_file_name):
            if os.path.exists(input_data.left_output_dir+left_file_name):
                os.remove(input_data.left_output_dir+left_file_name)
                
    for right_file_name in right_file_names:
        if right_file_name not in common_files and (len(right_file_name.split("__")) > 0 and right_file_name.split("__")[0] == input_data.png_file_name):
            if os.path.exists(input_data.right_output_dir+right_file_name):
                os.remove(input_data.right_output_dir+right_file_name)


    if input_data.only_one:
        left_image_paths = sorted([os.path.join(input_data.left_output_dir, f) for f in os.listdir(input_data.left_output_dir) if (len(f.split("__")) > 0 and f.split("__")[0] == input_data.png_file_name)], key=numerical_sort)
        right_image_paths = sorted([os.path.join(input_data.right_output_dir, f) for f in os.listdir(input_data.right_output_dir) if (len(f.split("__")) > 0 and f.split("__")[0] == input_data.png_file_name)], key=numerical_sort)
        if len(left_image_paths) > 1 or len(right_image_paths) > 1:
            for i in range(1, len(left_image_paths)):
                if os.path.exists(left_image_paths[i]):
                    os.remove(left_image_paths[i])    
            for i in range(1, len(right_image_paths)):
                if os.path.exists(right_image_paths[i]):
                    os.remove(right_image_paths[i])               

    res = GenerateCalibrationPngOutput()
    res.status = True
    res.timestamp = start_time
    res.file_name = input_data.png_file_name + '__002.png'
    
    #生成压缩图片（减小传输带宽）
    if input_data.need_compress:
        make_new_dir(input_data.left_compress_dir, False)
        make_new_dir(input_data.right_compress_dir, False)
        left_image_paths = sorted([os.path.join(input_data.left_output_dir, f) for f in os.listdir(input_data.left_output_dir) if (len(f.split("__")) > 0 and f.split("__")[0] == input_data.png_file_name)], key=numerical_sort)
        right_image_paths = sorted([os.path.join(input_data.right_output_dir, f) for f in os.listdir(input_data.right_output_dir) if (len(f.split("__")) > 0 and f.split("__")[0] == input_data.png_file_name)], key=numerical_sort)
        for i in left_image_paths:
            if os.path.exists(i):
                # 打开图像文件
                with Image.open(i) as img:
                    # 构建输出文件路径，保持文件名不变，但更改后缀为.jpg
                    output_path = os.path.join(input_data.left_compress_dir, f"{i.split('/')[-1].split(r'.')[0]}.jpg")
                    # 将图像转换为RGB模式（PNG可能有透明通道，JPEG不支持透明）
                    img = img.convert('RGB')
                    print("output_path left_compress_dir", output_path)
                    # 保存为JPEG，指定图像质量
                    img.save(output_path, 'JPEG', quality=100)
                    res.left_compress_path = output_path
        for i in right_image_paths:
            if os.path.exists(i):
                # 打开图像文件
                with Image.open(i) as img:
                    # 构建输出文件路径，保持文件名不变，但更改后缀为.jpg
                    output_path = os.path.join(input_data.right_compress_dir, f"{i.split('/')[-1].split(r'.')[0]}.jpg")
                    # 将图像转换为RGB模式（PNG可能有透明通道，JPEG不支持透明）
                    img = img.convert('RGB')
                    print("output_path right_compress_dir", output_path)
                    # 保存为JPEG，指定图像质量
                    img.save(output_path, 'JPEG', quality=100)
                    res.right_compress_path = output_path
    return res

# 拍摄当前的照片，并进行矫正，存储矫正后的原图片与压缩图片，将压缩图片发送到后端存储，前端调用获取
def recv_param_get_current_picture(client_socket: socket.socket, recv_data: RecvData):
    print("recv_data", recv_data)
    param_id = recv_data.param_id
    # 输出文件夹 output dir
    left_output = f"./rectify_img/{param_id}/origin/left/"
    right_output = f"./rectify_img/{param_id}/origin/right/"
    left_compress = f"./rectify_img/{param_id}/origin/compress_left/"
    right_compress = f"./rectify_img/{param_id}/origin/compress_right/"
    # 矫正后的图片存储
    rectify_output = f"./rectify_img/{param_id}/rectify/"
    rectify_compress_left =  f"./rectify_img/{param_id}/rectify/compress/left/"
    rectify_compress_right =  f"./rectify_img/{param_id}/rectify/compress/right/"
    make_new_dir(left_output, True)
    make_new_dir(right_output, True)
    make_new_dir(left_compress, True)
    make_new_dir(right_compress, True)
    
    make_new_dir(rectify_output, True)
    make_new_dir(rectify_compress_left, True)
    make_new_dir(rectify_compress_right, True)

    # 根据 param_id 查找 monitor
    # 根据查到的 monitor 获取两个 sensor_param_id
    # 获取两个 camera
    # 根据两个 camera 拼接得到两个 url
    monitors = monitorDao.find_by_param_id(param_id)
    if len(monitors) > 0:
        print(monitors)
        monitor = monitors[0]
        sensor_param_1 = monitor.sp_id1
        sensor_param_2 = monitor.sp_id2
        cameras_1 = cameraDao.find_by_sensor_param_id(sensor_param_1)
        cameras_2 = cameraDao.find_by_sensor_param_id(sensor_param_2)
        print("cameras_2", cameras_2, "cameras_1", cameras_1)
        if len(cameras_1) < 1 or len(cameras_2) < 1:
            print("相机不在库中！")
            return
        camera1 = cameras_1[0]
        camera2 = cameras_2[0]
        timestamp = get_current_timestamp()
        rtsp_url1 = f'rtsp://{camera1.user}:{camera1.password}@{camera1.local_ip}:554/media2/video1'  
        rtsp_url2 = f'rtsp://{camera2.user}:{camera2.password}@{camera2.local_ip}:554/media2/video1'
        input_data = GenerateCalibrationPngInput()
        input_data.mp4_file_name = f"{param_id}"
        input_data.png_file_name = f"{monitor.calibration_id}"
        input_data.rtsp_url1 = rtsp_url1
        input_data.rtsp_url2 = rtsp_url2
        input_data.left_output_dir = left_output
        input_data.right_output_dir = right_output
        input_data.only_one = True
        input_data.need_compress = True
        input_data.left_compress_dir = left_compress
        input_data.right_compress_dir = right_compress

        res = generate_calibration_png(input_data)

        if res.status:
            # 获取矫正图片
            get_recify_picture(left_output, right_output, rectify_output)
            # 获取矫正压缩图片
            compress_images(rectify_output + "left", rectify_compress_left, 20)
            compress_images(rectify_output + "right", rectify_compress_right, 20)
            # 将压缩图片返回后端
            # 左侧图片
            file_name = res.file_name
            file_name = file_name.replace('.png', '.jpg')
            image_left = cv2.imread(rectify_compress_left + file_name)
            _, left_buffer = cv2.imencode('.jpg', image_left, [cv2.IMWRITE_JPEG_QUALITY, 25])
            head = f"RECTIFY-JPG&&{param_id}&&LEFT&&{file_name}&&"
            client_socket.sendall(get_length_prefix_bytes(head.encode('utf-8') + base64.b64encode(left_buffer)))
            # 右侧图片
            image_right = cv2.imread(rectify_compress_right + file_name)
            _, right_buffer = cv2.imencode('.jpg', image_right, [cv2.IMWRITE_JPEG_QUALITY, 25])
            head = f"RECTIFY-JPG&&{param_id}&&RIGHT&&{file_name}&&"
            client_socket.sendall(get_length_prefix_bytes(head.encode('utf-8') + base64.b64encode(right_buffer)))

def recv_calibration_to_get_picture(client_socket: socket.socket, recv_data: RecvData):
    """
    接收到需要拍摄标定图片的回调函数
    """
    print("recv_data", recv_data)
    param_id = recv_data.param_id
    # 输出文件夹  output dir
    left_output = f"./calibration_type/{param_id}/raw_left/"
    right_output = f"./calibration_type/{param_id}/raw_right/"
    left_compress = f"./calibration_type/{param_id}/compress_left/"
    right_compress = f"./calibration_type/{param_id}/compress_right/"
    

    # 根据 param_id 查找 monitor
    # 根据查到的 monitor 获取两个 sensor_param_id
    # 获取两个 camera
    # 根据两个 camera 拼接得到两个 url
    monitors = monitorDao.find_by_param_id(param_id)
    if len(monitors) > 0:
        print(monitors)
        monitor = monitors[0]
        sensor_param_1 = monitor.sp_id1
        sensor_param_2 = monitor.sp_id2
        cameras_1 = cameraDao.find_by_sensor_param_id(sensor_param_1)
        cameras_2 = cameraDao.find_by_sensor_param_id(sensor_param_2)
        print("cameras_2", cameras_2, "cameras_1", cameras_1)
        if len(cameras_1) < 1 or len(cameras_2) < 1:
            print("相机不在库中！")
            return
        camera1 = cameras_1[0]
        camera2 = cameras_2[0]
        timestamp = get_current_timestamp()
        rtsp_url1 = f'rtsp://{camera1.user}:{camera1.password}@{camera1.local_ip}:554/media2/video1'  
        rtsp_url2 = f'rtsp://{camera2.user}:{camera2.password}@{camera2.local_ip}:554/media2/video1'
        input_data = GenerateCalibrationPngInput()
        input_data.mp4_file_name = f"{param_id}"
        input_data.png_file_name = f"{monitor.calibration_id}"
        input_data.rtsp_url1 = rtsp_url1
        input_data.rtsp_url2 = rtsp_url2
        input_data.left_output_dir = left_output
        input_data.right_output_dir = right_output
        input_data.only_one = True
        input_data.need_compress = True
        input_data.left_compress_dir = left_compress
        input_data.right_compress_dir = right_compress

        res = generate_calibration_png(input_data)

        if res.status:
            if res.left_compress_path != None:
                image = cv2.imread(res.left_compress_path)
                _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 25])
                # 发送左边
                head = f"CALIBRATION-JPG&&{param_id}&&{timestamp}&&LEFT&&{monitor.calibration_id}&&"
                client_socket.sendall(get_length_prefix_bytes(head.encode('utf-8') + base64.b64encode(buffer)))
                time.sleep(0.3)
            if res.right_compress_path != None:
                # 发送右边
                image = cv2.imread(res.right_compress_path)
                _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 25])
                head = f"CALIBRATION-JPG&&{param_id}&&{timestamp}&&RIGHT&&{monitor.calibration_id}&&"
                client_socket.sendall(get_length_prefix_bytes(head.encode('utf-8') + base64.b64encode(buffer)))

            #标定id自增
            monitor.calibration_id = monitor.calibration_id + 1
            monitorDao.save(monitor)
    else:
        print("未查询到相关监测参数项目，param_id=>", recv_data.param_id)
        return

def generate_vector_picture(client_socket: socket.socket, recv_data: RecvData):
    print()
    file_name = recv_data.file_name
    file_name = file_name.replace(".jpg", ".png")
    param_id = recv_data.param_id
    # 获取矫正图片进行网格划分
    input_data = NormalGenTestInput()
    input_data.x_max_left = recv_data.left_max_x
    input_data.x_min_left = recv_data.left_min_x
    input_data.y_max_left = recv_data.left_max_y
    input_data.y_min_left = recv_data.left_min_y

    input_data.x_max_right = recv_data.right_max_x
    input_data.x_min_right = recv_data.right_min_x
    input_data.y_max_right = recv_data.right_max_y
    input_data.y_min_right = recv_data.right_min_y
    # 矫正后的图片存储
    rectify_output = f"./rectify_img/{param_id}/rectify/"
    # 标定棋盘格图片
    normal_gen_output = f'./normal_gen_pic/{param_id}/output/'
    # 标定棋盘格压缩图片
    normal_compress_out_put = f"./normal_gen_pic/{param_id}/compress/"

    normal_gen_test(
        rectify_output + "left/" + file_name,
        rectify_output + "right/" + file_name,
        normal_gen_output, input_data, param_id)
    # 获取图片进行压缩处理
    compress_images(normal_gen_output, normal_compress_out_put, 20)
    # 将压缩文件返回
    for root, _, filenames in os.walk(normal_compress_out_put):
        for filename in filenames:
            mid_image = cv2.imread(normal_compress_out_put+filename)
            _, buffer = cv2.imencode('.jpg', mid_image, [cv2.IMWRITE_JPEG_QUALITY, 25])
            head = f"PARAM-PICTURE&&{param_id}&&{filename}&&"
            client_socket.sendall(get_length_prefix_bytes(head.encode('utf-8') + base64.b64encode(buffer)))


def recv_count_param_vector(data: RecvData):
    # print()
    param_id = data.param_id
    monitors = monitorDao.find_by_param_id(param_id)
    if len(monitors) > 0:
        res = count_param_vactor_by_pos(data.a, data.b, data.c, data.d, data.px1, data.py1, data.pz1, data.px2, data.py2, data.pz2)
        monitor = monitors[0]
        vector_len = math.sqrt(res['x']**2 + res['y']**2+res['z']**2)
        monitor.normal_x = res['x'] / vector_len
        monitor.normal_y = res['y'] / vector_len
        monitor.normal_z = res['z'] / vector_len
        monitorDao.save(monitor)

def recv_calibration_message(recv_data: RecvData):
    """
    收到修改相机参数的回调（ip、用户名、密码）
    """
    print("recv_data", recv_data)
    cameras = cameraDao.find_by_sensor_param_id(recv_data.sensor_param_id)
    if len(cameras) == 0: # 数据库中不存在相机参数
        new_camera = Camera()
        new_camera.local_ip = recv_data.local_ip
        new_camera.sensor_param_id = recv_data.sensor_param_id
        new_camera.user = recv_data.user
        new_camera.password = recv_data.password
        cameraDao.save(new_camera)
        print("监测对象id没有匹配的相机")
        #监测对象id没有匹配的相机
    else:
        print("监测对象id有匹配的相机")
        old_camera = cameras[0]
        old_camera.local_ip = recv_data.local_ip
        old_camera.sensor_param_id = recv_data.sensor_param_id
        old_camera.user = recv_data.user
        old_camera.password = recv_data.password
        cameraDao.save(old_camera)


def recv_calibration_picture_delete(recv_data: RecvData):
    """
    收到需要删除标定对象中的某一个具体图片的回调
    """
    param_id = recv_data.param_id
    #输出文件夹  output dir
    left_output = f"./calibration_type/{param_id}/raw_left/"
    right_output = f"./calibration_type/{param_id}/raw_right/"
    left_compress = f"./calibration_type/{param_id}/compress_left/"
    right_compress = f"./calibration_type/{param_id}/compress_right/"   
    delete_files_by_prefix(left_output, f'{recv_data.num}')
    delete_files_by_prefix(right_output, f'{recv_data.num}')
    delete_files_by_prefix(left_compress, f'{recv_data.num}')
    delete_files_by_prefix(right_compress, f'{recv_data.num}')
    pass

def recv_fix_position(client_socket: socket.socket, recv_data: RecvData):
    """
    收到需要固定相机经纬度后的回调
    """
    param_id = recv_data.param_id
    monitors = monitorDao.find_by_param_id(param_id)
    if len(monitors) > 0:
        print(monitors)
        monitor = monitors[0]
        sensor_param_1 = monitor.sp_id1
        sensor_param_2 = monitor.sp_id2
        get_current_camera_params(sensor_param_1)
        get_current_camera_params(sensor_param_2)

def generate_calibration_result(client_socket: socket.socket, recv_data: RecvData, test: bool = False):
    """
    生成标定结果
    """
    cfg = load_config("./config/local.ini")
    config = {
        "left_images_folder": f"./calibration_type/{recv_data.param_id}/raw_left",    #棋盘格尺寸
        "right_images_folder": f"./calibration_type/{recv_data.param_id}/raw_right",  #棋盘格尺寸

        "pattern_size": (cfg.getint('calibration', 'pattern_size_1'), cfg.getint('calibration', 'pattern_size_2')),   #棋盘格尺寸
        "test_root_folder": "./test/",  #用来存储测试结果的根文件夹
        "actual_size": cfg.getfloat('calibration', 'actual_size'),  #棋盘格的实际尺寸 单位：mm  actual size of chessboard cell  unit：mm
        "output_name": f"./calibration_parameters_{recv_data.param_id}.yaml"
    }

    if test:
        config['left_images_folder'] = f"./test_data/left"
        config['right_images_folder'] = f"./test_data/right"
        config['output_name'] = f"./calibration_parameters_{recv_data.param_id}_test.yaml"
        
    input_data = MakeCalibrationInput.model_validate(config)
    res = make_calibration(input_data)
    res.paramId = recv_data.param_id
    res_bytes = res.model_dump_json().encode('utf-8')
    client_socket.sendall(get_length_prefix_bytes(res_bytes))
    

def recv_match_range_message(recv_data: RecvData):
    """
    收到需要修改画面识别范围的回调
    """
    monitors = monitorDao.find_by_param_id(recv_data.param_id)
    if len(monitors) > 0:
        monitor = monitors[0]
        print("修改前的monitor", monitor)
        if monitor.sp_id1 in recv_data.video_param_ids: #left
            if recv_data.x_min >= 0:
                monitor.x_min_left = recv_data.x_min
            if recv_data.x_max >= 0:
                monitor.x_max_left = recv_data.x_max
            if recv_data.y_min >= 0:
                monitor.y_min_left = recv_data.y_min
            if recv_data.y_max >= 0:
                monitor.y_max_left = recv_data.y_max
        if monitor.sp_id2 in recv_data.video_param_ids: #right
            if recv_data.x_min >= 0:
                monitor.x_min_right = recv_data.x_min
            if recv_data.x_max >= 0:
                monitor.x_max_right = recv_data.x_max
            if recv_data.y_min >= 0:
                monitor.y_min_right = recv_data.y_min
            if recv_data.y_max >= 0:
                monitor.y_max_right = recv_data.y_max
        print("修改后的monitor", monitor)
        monitorDao.save(monitor)

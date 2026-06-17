"""
用来将所有相机校正为初始状态
"""

import sqlite3
from typing import Dict, Optional
from pydantic import BaseModel, Field
import requests
from requests.auth import HTTPDigestAuth
from utils.sqlite_utils import find_all

from dao.camera_dao import cameraDao

import cv2
import numpy as np


class GetCurrentCameraParamsOutput(BaseModel):
    longitude: Optional[float] = Field(default=None)
    latitude: Optional[float] = Field(default=None)
    focal_zoom: Optional[float] = Field(default=None)

    


def snapshot(local_ip: str, user: str, password: str, channel_id: str = '2'):
    url = f"http://{local_ip}/LAPI/V1.0/Channels/{channel_id}/Media/Video/Streams/2/Snapshot"
    try:
        response = requests.get(url, auth=HTTPDigestAuth(user, password), timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return

    if response.status_code == 200:
        # 获取返回的图片数据
        image_data = response.content

        # 将 image_data 转换为 numpy.ndarray
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 使用 IMREAD_COLOR 读取彩色图像
        if frame is not None:
            return frame
        else:
            return None
    else:
        return None
    
def get_current_camera_params(sensor_param: int) -> GetCurrentCameraParamsOutput:
    """
    获取当前相机参数（经纬度、缩放系数）
    """
    cameras = cameraDao.find_by_sensor_param_id(sensor_param)
    if len(cameras) == 0: #查询不到则返回空数组
        return GetCurrentCameraParamsOutput()
    res = GetCurrentCameraParamsOutput()
    camera = cameras[0]
    # 经纬度api
    move_url = f"http://{camera.local_ip}/LAPI/V1.0/Channels/2/PTZ/AbsoluteMove"
    # 发送带有 Digest 鉴权的 GET 请求
    response = requests.get(move_url, auth=HTTPDigestAuth(camera.user, camera.password))
    if response.status_code == 200:
        # 输出返回的 JSON 数据
        res_dict: Dict = response.json()['Response']['Data']
        longitude = res_dict['Longitude']
        latitude = res_dict['Latitude']
        res.longitude = longitude
        res.latitude = latitude
        print(f'{camera.local_ip} 经纬度数据: f{res_dict}')
    
    # 缩放系数api
    zoom_url = f"http://{camera.local_ip}/LAPI/V1.0/Channels/2/PTZ/AbsoluteZoom"
    # 发送带有 Digest 鉴权的 GET 请求
    response = requests.get(zoom_url, auth=HTTPDigestAuth(camera.user, camera.password))

    # 检查请求是否成功
    if response.status_code == 200:
        # 输出返回的 JSON 数据
        res_dict: Dict = response.json()['Response']['Data']
        focal_zoom = res_dict['ZoomRatio']
        res.focal_zoom = focal_zoom
        print(f'{camera.local_ip} 缩放数据: f{res_dict}')
    
    if res.focal_zoom != None:
        camera.focal_zoom = res.focal_zoom
    if res.latitude != None:
        camera.latitude = res.latitude
    if res.longitude != None:
        camera.longitude = res.longitude
    cameraDao.save(camera)
    return res


def reset_camera_params(cur: sqlite3.Cursor, update: bool = True):
    res = find_all(cur, "select id, local_ip, focal_zoom, longitude, latitude, user, password from camera")
    print("res", res)
    for i in res:
        id, local_ip, focal_zoom, longitude, latitude, user, password = i
        # 经纬度api
        move_url = f"http://{local_ip}/LAPI/V1.0/Channels/2/PTZ/AbsoluteMove"
        # 发送带有 Digest 鉴权的 GET 请求
        response = requests.get(move_url, auth=HTTPDigestAuth(user, password))

        # 检查请求是否成功
        if response.status_code == 200 and longitude is not None and latitude is not None:
            # 输出返回的 JSON 数据
            res_dict: Dict = response.json()['Response']['Data']
            real_longitude = res_dict['Longitude']
            real_latitude = res_dict['Latitude']
            print(f'{local_ip} 经纬度数据: f{res_dict}')
            if update:
                need_update = False
                if abs(longitude - real_longitude) > 1e-4:
                    need_update = True
                if abs(latitude - real_latitude) > 1e-4:
                    need_update = True
                if need_update:
                    #标定的经纬度与实际不一致，将，实际经纬度转化为标定经纬度
                    data = {
                                "Longitude": longitude,
                                "Latitude":	latitude,
                                "MoveSpeed":	0
                            }
                    print("需要复原", data)
                    response = requests.put(move_url, auth=HTTPDigestAuth(user, password), json=data)
        else:
            print(f"请求失败，状态码：{response.status_code}")

        zoom_url = f"http://{local_ip}/LAPI/V1.0/Channels/2/PTZ/AbsoluteZoom"
        # 发送带有 Digest 鉴权的 GET 请求
        response = requests.get(zoom_url, auth=HTTPDigestAuth(user, password))

        # 检查请求是否成功
        if response.status_code == 200 and focal_zoom is not None:
            # 输出返回的 JSON 数据
            res_dict: Dict = response.json()['Response']['Data']
            print(f'{local_ip} 缩放数据: f{res_dict}')
            if update:
                real_focal_zoom = res_dict['ZoomRatio']
                if abs(real_focal_zoom - focal_zoom) > 1e-4:
                    #标定的度与实际不一致，将，实际经纬度转化为标定经纬度
                    data = {
                                "ZoomRatio": focal_zoom,
                                "ZoomSpeed":	0
                            }
                    print("需要复原", data)
                    response = requests.put(zoom_url, auth=HTTPDigestAuth(user, password), json=data)

        else:
            print(f"请求失败，状态码：{response.status_code}")


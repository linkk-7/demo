"""
视频操作工具类
"""
import cv2
import subprocess
import os
from typing import Tuple
from utils.file_utils import make_new_dir
from utils.time_utils import get_current_timestamp
from utils.real_camera_utils import snapshot

def check_rtsp_stream(rtsp_url: str) -> bool:
    """
    检查rtsp流是否有效
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return False
    ret, _ = cap.read() 
    if ret:
        cap.release()  # 释放资源
        return True
    else:
        cap.release()  # 释放资源
        return False
    
def extract_keyframes(video_path: str, output_pattern: str) -> None:
    """
    从视频中抽取所有关键帧，并保存在指定文件目录下
    """
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', "select='eq(pict_type\\,I)'",
        '-vsync', 'vfr',
        output_pattern
    ]
    
    # 使用 subprocess 运行命令
    subprocess.run(command, check=True)

def record_rtsp_stream(rtsp_url: str, output_file: str, duration: float, keyframe_interval: int = 10) -> None:
    """
    从rtsp流中提取指定时长与关键帧间隔的视频
    rtsp_url: rtsp流地址
    output_file: 输出文件的地址
    duration: 时长
    keyframe_interval: 关键帧间隔
    """
    command = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-buffer_size', '2000000',     #增大缓冲区，减小关键帧丢失带来的影响
        '-i', rtsp_url,
        '-t', str(duration),
        '-c:v', 'libx264',
        '-an',
        '-g', str(keyframe_interval),  # 设置关键帧间隔（GOP 长度）
        output_file
    ]
    subprocess.run(command)

def get_snapshot_frames(left_snapshot_parmas: tuple, right_snapshot_parmas: tuple, left_output_dir: str, right_output_dir: str) -> Tuple[bool, int]:
    left_frame = snapshot(left_snapshot_parmas[0], left_snapshot_parmas[1], left_snapshot_parmas[2], '1')
    right_frame = snapshot(right_snapshot_parmas[0], right_snapshot_parmas[1], right_snapshot_parmas[2], '2')
    if left_frame is None:
        print("left frame not captured")
        return
    if right_frame is None:
        print("right frame not captured")
        return
    print("left frame shape: ", left_frame.shape)
    print("right frame shape", right_frame.shape)

    make_new_dir(left_output_dir, True)
    make_new_dir(right_output_dir, True)

    start_time = get_current_timestamp()
    print("start time:", start_time)
    left_image_path = os.path.join(left_output_dir, 'frame_001.png')
    right_image_path = os.path.join(right_output_dir, 'frame_001.png')

    cv2.imwrite(left_image_path, left_frame)
    cv2.imwrite(right_image_path, right_frame)

    return True, start_time
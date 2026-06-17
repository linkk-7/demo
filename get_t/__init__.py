"""
Legacy 兼容工具：旧式文件名时间解析与顺序重命名。

定位说明：
1. get_t()：面向历史旧数据目录的离线时间数组导出工具。
2. rename()：面向历史旧数据目录的顺序重命名工具。
3. 当前新采集层/主流程已在采集阶段提供 FramePacket 元信息和标准命名，
   不再依赖本模块函数作为主流程步骤。
"""

import os
import re
import warnings
from datetime import datetime

import numpy as np


def get_t(folder_path):
    """
    Legacy: 从“旧式文件名中的日期时间串”解析时间并导出时间数组。

    说明：
    - 本函数主要用于历史旧数据离线兼容处理。
    - 当前新主流程不再依赖该函数。
    - 期望文件名中包含可匹配 `%Y%m%d%H%M%S%f` 的 19 位时间串。
    """
    warnings.warn(
        "get_t() is a legacy/offline compatibility tool. "
        "Current main pipeline no longer depends on it.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    last_dir = os.path.basename(folder_path)
    save_path = os.path.dirname(folder_path)

    time_pattern = re.compile(r"(\d{19})")

    records = []

    for filename in os.listdir(folder_path):
        match = time_pattern.search(filename)
        if match:
            time_str = match.group(1)
            dt = datetime.strptime(time_str, "%Y%m%d%H%M%S%f")
            ts_ms = int(dt.timestamp() * 1000)
            records.append((filename, time_str, dt, ts_ms))

    if not records:
        raise ValueError(f"No parseable datetime token found in folder: {folder_path}")

    records.sort(key=lambda x: x[3])

    filenames_sorted = np.array([r[0] for r in records])
    datetime_str = np.array([r[2].strftime("%Y-%m-%d %H:%M:%S.%f") for r in records])
    absolute_timestamps_ms = np.array([r[3] for r in records], dtype=np.int64)

    first_timestamp_ms = absolute_timestamps_ms[0]
    relative_timestamps_ms = absolute_timestamps_ms - first_timestamp_ms

    np.save(os.path.join(save_path, f"filenames_{last_dir}.npy"), filenames_sorted)
    np.save(os.path.join(save_path, f"datetime_str_{last_dir}.npy"), datetime_str)
    np.save(os.path.join(save_path, f"t_abs_{last_dir}.npy"), absolute_timestamps_ms)
    np.save(os.path.join(save_path, f"t_rel_{last_dir}.npy"), relative_timestamps_ms)

    return {
        "filenames": filenames_sorted,
        "datetime_str": datetime_str,
        "t_abs": absolute_timestamps_ms,
        "t_rel": relative_timestamps_ms,
        "first_timestamp_ms": int(first_timestamp_ms),
    }


def rename(folder_path, mode):
    """
    Legacy: 将图片按排序重命名为 1l.jpg/1r.jpg ...

    说明：
    - 本函数主要用于历史旧数据离线兼容处理。
    - 当前新主流程不再依赖该函数。
    """
    warnings.warn(
        "rename() is a legacy/offline compatibility tool. "
        "Current main pipeline no longer depends on it.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    files = os.listdir(folder_path)

    def extract_time_from_filename(filename):
        time_str = filename.split(".")[0]
        return int(time_str)

    sorted_files = sorted(files, key=extract_time_from_filename)

    if mode == "l":
        for idx, filename in enumerate(sorted_files, start=1):
            new_filename = f"{idx}l.jpg"
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(old_file_path, new_file_path)

        print("[LEGACY] left files renamed to sequential '*l.jpg'.")
    elif mode == "r":
        for idx, filename in enumerate(sorted_files, start=1):
            new_filename = f"{idx}r.jpg"
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(old_file_path, new_file_path)

        print("[LEGACY] right files renamed to sequential '*r.jpg'.")
    else:
        raise ValueError("mode must be 'l' or 'r'.")

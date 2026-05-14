import os
import re
from typing import Optional

from .capture import CameraController
from .models import FramePacket


FRAME_ID_MAX = 999999  # 最大帧号
_PAIR_NAME_PATTERNS = {
    "L": re.compile(r"^L_\d+_(\d+)\.png$"),
    "R": re.compile(r"^R_\d+_(\d+)\.png$"),
}


def _normalize_frame_id(frame_id: int, frame_id_max: int = FRAME_ID_MAX) -> int:
    if frame_id_max <= 0:
        raise ValueError("frame_id_max must be positive.")
    return ((int(frame_id) - 1) % frame_id_max) + 1


def _next_frame_id(frame_id: int, frame_id_max: int = FRAME_ID_MAX) -> int:
    return 1 if frame_id >= frame_id_max else frame_id + 1


def _extract_frame_id_from_filename(filename: str, prefix: str) -> Optional[int]:
    pattern = _PAIR_NAME_PATTERNS[prefix]
    match = pattern.match(filename)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def infer_start_frame_id_from_cache(save_folder_base: str, frame_id_max: int = FRAME_ID_MAX) -> int:
    """
    从 left/ 和 right/ 历史文件推断下一次起始 frame_id。
    规则：
    - 扫描并构造已占用 frame_id 集合（跳过异常文件）
    - 候选起点优先为 (max_frame_id + 1)，超过上限回绕到 1
    - 从候选起点循环查找第一个未占用 frame_id
    """
    max_found = 0
    occupied_ids = set()
    for prefix, subdir in (("L", "left"), ("R", "right")):
        folder = os.path.join(save_folder_base, subdir)
        if not os.path.isdir(folder):
            continue
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            frame_id = _extract_frame_id_from_filename(name, prefix)
            if frame_id is None:
                continue
            if 1 <= frame_id <= frame_id_max:
                occupied_ids.add(frame_id)
                max_found = max(max_found, frame_id)

    if not occupied_ids:
        return 1

    if len(occupied_ids) >= frame_id_max:
        raise RuntimeError("No available frame_id: all frame ids are currently occupied in cache directories.")

    candidate = _next_frame_id(max_found, frame_id_max)
    current = candidate
    for _ in range(frame_id_max):
        if current not in occupied_ids:
            return current
        current = _next_frame_id(current, frame_id_max)

    raise RuntimeError("No available frame_id found after full scan.")


# 统一包接口：让顶层脚本只调用包 API，不再直接拼底层实现细节。
def capture_and_save_frame_pair(
    save_folder_base: str,
    frame_id: int,
    keep_last_n_pairs: Optional[int] = None,
    include_images: bool = False,
    left_camera_index: int = 0,
    right_camera_index: int = 1,
) -> FramePacket:
    controller = CameraController(save_folder_base=save_folder_base, keep_last_n_pairs=keep_last_n_pairs)
    return controller.capture_and_save_frame_pair(
        frame_id=frame_id,
        left_camera_index=left_camera_index,
        right_camera_index=right_camera_index,
        include_images=include_images,
    )


def continuous_capture(
    save_folder_base: str,
    start_frame_id: Optional[int] = None,
    keep_last_n_pairs: Optional[int] = None,
    include_images: bool = False,
    max_frames: Optional[int] = None,
    left_camera_index: int = 0,
    right_camera_index: int = 1,
    frame_id_max: int = FRAME_ID_MAX,
) -> Optional[FramePacket]:
    controller = CameraController(save_folder_base=save_folder_base, keep_last_n_pairs=keep_last_n_pairs)
    cams = controller.open_cameras()
    if start_frame_id is None:
        frame_id = infer_start_frame_id_from_cache(save_folder_base, frame_id_max=frame_id_max)
    else:
        frame_id = _normalize_frame_id(start_frame_id, frame_id_max=frame_id_max)
    captured_count = 0
    last_packet: Optional[FramePacket] = None

    try:
        while True:
            if max_frames is not None and captured_count >= max_frames:
                break

            last_packet = controller.capture_and_save_frame_pair(
                frame_id=frame_id,
                left_camera_index=left_camera_index,
                right_camera_index=right_camera_index,
                include_images=include_images,
                cams=cams,
            )
            captured_count += 1
            frame_id = _next_frame_id(frame_id, frame_id_max=frame_id_max)
            print(
                f"saved pair #{last_packet.frame_id}: "
                f"{last_packet.left_filename} | {last_packet.right_filename}"
            )
    except KeyboardInterrupt:
        print("capture interrupted by user")
    finally:
        controller.close_cameras(cams)

    return last_packet

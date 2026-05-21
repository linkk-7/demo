import os
import re
import time
from typing import Callable, Dict, Optional

from .benchmark import benchmark_capture as run_benchmark_capture
from .capture import CameraController
from .models import FramePacket
from .save_utils import AsyncImageSaver


FRAME_ID_MAX = 999999
_PAIR_NAME_PATTERNS = {
    "L": re.compile(r"^L_\d+_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE),
    "R": re.compile(r"^R_\d+_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE),
}


def _normalize_frame_id(frame_id: int, frame_id_max: int = FRAME_ID_MAX) -> int:
    """将任意整数 frame_id 归一化到 [1, frame_id_max]。"""
    if frame_id_max <= 0:
        raise ValueError("frame_id_max must be positive.")
    return ((int(frame_id) - 1) % frame_id_max) + 1


def _next_frame_id(frame_id: int, frame_id_max: int = FRAME_ID_MAX) -> int:
    """获取下一个 frame_id（超过上限后回绕到 1）。"""
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
    """从 left/right 目录推断安全起始 frame_id，避免重启撞号。"""
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


def _capture_single_pair(
    save_folder_base: str,
    frame_id: int,
    keep_last_n_pairs: Optional[int],
    include_images: bool,
    save_images: bool,
    left_camera_index: int,
    right_camera_index: int,
    image_ext: str,
    jpeg_quality: int,
    png_compression: int,
    exposure_time_us: float,
    target_acquisition_fps: float,
    log_camera_nodes: bool,
) -> FramePacket:
    """内部单对采集函数（不对外导出）。"""
    controller = CameraController(
        save_folder_base=save_folder_base,
        keep_last_n_pairs=keep_last_n_pairs,
        exposure_time_us=exposure_time_us,
        target_acquisition_fps=target_acquisition_fps,
        log_camera_nodes=log_camera_nodes,
    )
    return controller.capture_and_save_frame_pair(
        frame_id=frame_id,
        left_camera_index=left_camera_index,
        right_camera_index=right_camera_index,
        include_images=include_images,
        save_images=save_images,
        image_ext=image_ext,
        jpeg_quality=jpeg_quality,
        png_compression=png_compression,
    )


def _continuous_capture_core(
    save_folder_base: str,
    start_frame_id: Optional[int],
    keep_last_n_pairs: Optional[int],
    include_images: bool,
    save_images: bool,
    max_frames: Optional[int],
    left_camera_index: int,
    right_camera_index: int,
    frame_id_max: int,
    image_ext: str,
    jpeg_quality: int,
    png_compression: int,
    log_every_n: int,
    exposure_time_us: float,
    target_acquisition_fps: float,
    log_camera_nodes: bool,
    use_async_saver: bool,
    async_queue_size: int,
    async_save_policy: str,
    on_frame: Optional[Callable[[FramePacket], None]] = None,
) -> Optional[FramePacket]:
    """连续采集内部核心循环。"""
    controller = CameraController(
        save_folder_base=save_folder_base,
        keep_last_n_pairs=keep_last_n_pairs,
        exposure_time_us=exposure_time_us,
        target_acquisition_fps=target_acquisition_fps,
        log_camera_nodes=log_camera_nodes,
    )
    cams = controller.open_cameras()

    async_saver: Optional[AsyncImageSaver] = None
    if use_async_saver and save_images:
        async_saver = AsyncImageSaver(
            save_fn=controller._write_pair_to_paths,
            on_saved=lambda task: controller._track_and_enforce_pair_retention(
                frame_id=task.frame_id,
                left_path=task.left_path,
                right_path=task.right_path,
            ),
            maxsize=async_queue_size,
            save_policy=async_save_policy,
        )
        async_saver.start()
    elif use_async_saver and not save_images:
        print("[capture] use_async_saver=True ignored because save_images=False")

    if start_frame_id is None:
        frame_id = infer_start_frame_id_from_cache(save_folder_base, frame_id_max=frame_id_max)
    else:
        frame_id = _normalize_frame_id(start_frame_id, frame_id_max=frame_id_max)

    captured_count = 0
    last_packet: Optional[FramePacket] = None
    loop_start_ts = time.monotonic()

    try:
        while True:
            if max_frames is not None and captured_count >= max_frames:
                break

            last_packet = controller.capture_and_save_frame_pair(
                frame_id=frame_id,
                left_camera_index=left_camera_index,
                right_camera_index=right_camera_index,
                include_images=include_images,
                save_images=save_images,
                image_ext=image_ext,
                jpeg_quality=jpeg_quality,
                png_compression=png_compression,
                cams=cams,
                async_saver=async_saver,
            )
            captured_count += 1
            frame_id = _next_frame_id(frame_id, frame_id_max=frame_id_max)

            if on_frame is not None:
                on_frame(last_packet)

            if log_every_n > 0 and captured_count % log_every_n == 0:
                elapsed = time.monotonic() - loop_start_ts
                pair_fps = captured_count / elapsed if elapsed > 0 else 0.0
                msg = f"[capture] pairs={captured_count}, elapsed={elapsed:.2f}s, pair_fps={pair_fps:.2f}"
                if async_saver is not None:
                    msg += (
                        f", async_q={async_saver.queue_size}, "
                        f"saved={async_saver.saved_pairs}, dropped={async_saver.dropped_pairs}, failed={async_saver.failed_pairs}"
                    )
                print(msg)
    except KeyboardInterrupt:
        print("capture interrupted by user")
    finally:
        if async_saver is not None:
            async_saver.flush()
            async_saver.stop()
            print(
                "[capture] async saver summary: "
                f"saved={async_saver.saved_pairs}, dropped={async_saver.dropped_pairs}, failed={async_saver.failed_pairs}"
            )

        controller.close_cameras(cams)

    return last_packet


def continuous_capture_online(
    save_folder_base: str,
    on_frame: Optional[Callable[[FramePacket], None]] = None,
    start_frame_id: Optional[int] = None,
    max_frames: Optional[int] = None,
    left_camera_index: int = 0,
    right_camera_index: int = 1,
    frame_id_max: int = FRAME_ID_MAX,
    log_every_n: int = 30,
    exposure_time_us: float = 8000.0,
    target_acquisition_fps: float = 30.0,
    log_camera_nodes: bool = False,
) -> Optional[FramePacket]:
    """在线采集模式：默认不落盘，packet 携带图像数组。"""
    return _continuous_capture_core(
        save_folder_base=save_folder_base,
        start_frame_id=start_frame_id,
        keep_last_n_pairs=None,
        include_images=True,
        save_images=False,
        max_frames=max_frames,
        left_camera_index=left_camera_index,
        right_camera_index=right_camera_index,
        frame_id_max=frame_id_max,
        image_ext=".jpg",
        jpeg_quality=90,
        png_compression=1,
        log_every_n=log_every_n,
        exposure_time_us=exposure_time_us,
        target_acquisition_fps=target_acquisition_fps,
        log_camera_nodes=log_camera_nodes,
        use_async_saver=False,
        async_queue_size=256,
        async_save_policy="block",
        on_frame=on_frame,
    )


def continuous_capture_disk(
    save_folder_base: str,
    keep_last_n_pairs: Optional[int] = 1000,
    jpeg_quality: int = 90,
    include_images: bool = False,
    on_frame: Optional[Callable[[FramePacket], None]] = None,
    start_frame_id: Optional[int] = None,
    max_frames: Optional[int] = None,
    left_camera_index: int = 0,
    right_camera_index: int = 1,
    frame_id_max: int = FRAME_ID_MAX,
    log_every_n: int = 30,
    exposure_time_us: float = 8000.0,
    target_acquisition_fps: float = 30.0,
    log_camera_nodes: bool = False,
    use_async_saver: bool = False,
    async_queue_size: int = 256,
    async_save_policy: str = "block",
) -> Optional[FramePacket]:
    """存盘循环覆盖模式：保存 JPG，并按 keep_last_n_pairs 保留最近 N 对。"""
    return _continuous_capture_core(
        save_folder_base=save_folder_base,
        start_frame_id=start_frame_id,
        keep_last_n_pairs=keep_last_n_pairs,
        include_images=include_images,
        save_images=True,
        max_frames=max_frames,
        left_camera_index=left_camera_index,
        right_camera_index=right_camera_index,
        frame_id_max=frame_id_max,
        image_ext=".jpg",
        jpeg_quality=jpeg_quality,
        png_compression=1,
        log_every_n=log_every_n,
        exposure_time_us=exposure_time_us,
        target_acquisition_fps=target_acquisition_fps,
        log_camera_nodes=log_camera_nodes,
        use_async_saver=use_async_saver,
        async_queue_size=async_queue_size,
        async_save_policy=async_save_policy,
        on_frame=on_frame,
    )


def benchmark_capture(
    save_folder_base: str,
    duration_sec: float = 10.0,
    save_images: bool = False,
    convert_to_bgr: bool = True,
    image_ext: str = ".png",
    jpeg_quality: int = 90,
    png_compression: int = 1,
    left_camera_index: int = 0,
    right_camera_index: int = 1,
    log_every_n: int = 30,
    warmup_sec: float = 1.0,
    exposure_time_us: float = 8000.0,
    target_acquisition_fps: float = 30.0,
    log_camera_nodes: bool = False,
) -> Dict[str, float]:
    """内部调试工具转发接口（不作为主入口推荐）。"""
    return run_benchmark_capture(
        save_folder_base=save_folder_base,
        duration_sec=duration_sec,
        save_images=save_images,
        convert_to_bgr=convert_to_bgr,
        image_ext=image_ext,
        jpeg_quality=jpeg_quality,
        png_compression=png_compression,
        left_camera_index=left_camera_index,
        right_camera_index=right_camera_index,
        log_every_n=log_every_n,
        warmup_sec=warmup_sec,
        exposure_time_us=exposure_time_us,
        target_acquisition_fps=target_acquisition_fps,
        log_camera_nodes=log_camera_nodes,
    )


def continuous_capture(*args, **kwargs):
    """兼容旧入口：默认转到 continuous_capture_online。"""
    return continuous_capture_online(*args, **kwargs)


import time
from typing import Dict, Optional

from .capture import CameraController


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
    """采集性能测试工具（内部调试使用）。

    常见模式：
    - 仅采集：save_images=False, convert_to_bgr=False
    - 采集+BGR：save_images=False, convert_to_bgr=True
    - 采集+保存：save_images=True, convert_to_bgr=True
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")
    if save_images and not convert_to_bgr:
        print("[WARN] save_images=True requires convert_to_bgr; force convert_to_bgr=True")
        convert_to_bgr = True

    controller = CameraController(
        save_folder_base=save_folder_base,
        keep_last_n_pairs=None,
        exposure_time_us=exposure_time_us,
        target_acquisition_fps=target_acquisition_fps,
        log_camera_nodes=log_camera_nodes,
    )

    ext = controller._normalize_image_ext(image_ext)
    cams = controller.open_cameras()

    left_count = 0
    right_count = 0
    pair_count = 0
    sum_pair_interval = 0.0
    min_pair_interval = float("inf")
    max_pair_interval = 0.0
    last_pair_ts: Optional[float] = None

    try:
        warmup_deadline = time.monotonic() + max(0.0, warmup_sec)
        while time.monotonic() < warmup_deadline:
            _ = controller._get_image(cams[left_camera_index], convert_to_bgr=convert_to_bgr)
            _ = controller._get_image(cams[right_camera_index], convert_to_bgr=convert_to_bgr)

        start_ts = time.monotonic()
        while True:
            now_ts = time.monotonic()
            if now_ts - start_ts >= duration_sec:
                break

            left_img, left_meta = controller._get_image(cams[left_camera_index], convert_to_bgr=convert_to_bgr)
            right_img, right_meta = controller._get_image(cams[right_camera_index], convert_to_bgr=convert_to_bgr)

            if left_meta is None or right_meta is None:
                continue

            left_count += 1
            right_count += 1
            pair_count += 1

            pair_ts = time.monotonic()
            if last_pair_ts is not None:
                interval = pair_ts - last_pair_ts
                sum_pair_interval += interval
                min_pair_interval = min(min_pair_interval, interval)
                max_pair_interval = max(max_pair_interval, interval)
            last_pair_ts = pair_ts

            if save_images:
                controller._save_pair_images(
                    left_image=left_img,
                    right_image=right_img,
                    frame_id=pair_count,
                    left_host_timestamp=left_meta.host_timestamp,
                    right_host_timestamp=right_meta.host_timestamp,
                    image_ext=ext,
                    jpeg_quality=jpeg_quality,
                    png_compression=png_compression,
                )

            if log_every_n > 0 and pair_count % log_every_n == 0:
                elapsed = pair_ts - start_ts
                fps = pair_count / elapsed if elapsed > 0 else 0.0
                print(f"[benchmark] pairs={pair_count}, elapsed={elapsed:.2f}s, pair_fps={fps:.2f}")

        end_ts = time.monotonic()
        total_elapsed = end_ts - start_ts

        left_fps = left_count / total_elapsed if total_elapsed > 0 else 0.0
        right_fps = right_count / total_elapsed if total_elapsed > 0 else 0.0
        pair_fps = pair_count / total_elapsed if total_elapsed > 0 else 0.0

        if pair_count > 1:
            avg_interval_ms = (sum_pair_interval / (pair_count - 1)) * 1000.0
            min_interval_ms = min_pair_interval * 1000.0
            max_interval_ms = max_pair_interval * 1000.0
        else:
            avg_interval_ms = 0.0
            min_interval_ms = 0.0
            max_interval_ms = 0.0

        stats: Dict[str, float] = {
            "total_elapsed_sec": total_elapsed,
            "left_count": float(left_count),
            "right_count": float(right_count),
            "pair_count": float(pair_count),
            "left_fps": left_fps,
            "right_fps": right_fps,
            "pair_fps": pair_fps,
            "average_pair_interval_ms": avg_interval_ms,
            "min_pair_interval_ms": min_interval_ms,
            "max_pair_interval_ms": max_interval_ms,
        }

        print("[benchmark] summary:")
        print(f"  total_elapsed_sec={stats['total_elapsed_sec']:.3f}")
        print(f"  left_count={int(stats['left_count'])}, right_count={int(stats['right_count'])}, pair_count={int(stats['pair_count'])}")
        print(f"  left_fps={stats['left_fps']:.3f}, right_fps={stats['right_fps']:.3f}, pair_fps={stats['pair_fps']:.3f}")
        print(f"  average_pair_interval_ms={stats['average_pair_interval_ms']:.3f}")
        print(f"  min_pair_interval_ms={stats['min_pair_interval_ms']:.3f}, max_pair_interval_ms={stats['max_pair_interval_ms']:.3f}")

        return stats
    finally:
        controller.close_cameras(cams)


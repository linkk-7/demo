# 保留旧批处理的逻辑

import os
from typing import Optional

import cv2

from .rectify import StereoRectifier
from .utils import load_rectification_params


def stereo_rectification(
    input_folder1: str,
    input_folder2: str,
    output_folder1: str,
    output_folder2: str,
    calibration_file1: str,
    calibration_file2: str,
    r: str,
    t: str,
    mode: str,
    d1_matlab: Optional[str] = None,
    d2_matlab: Optional[str] = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> bool:
    """
    Legacy/offline batch rectification entry.
    Kept for compatibility with existing scripts such as 2.2.0_rectification.py.
    """
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)

    image_files1 = sorted(
        [f for f in os.listdir(input_folder1) if f.lower().endswith((".jpg", ".bmp", ".png"))]
    )
    image_files2 = sorted(
        [f for f in os.listdir(input_folder2) if f.lower().endswith((".jpg", ".bmp", ".png"))]
    )
    print(image_files1)

    try:
        params = load_rectification_params(
            calibration_file1=calibration_file1,
            calibration_file2=calibration_file2,
            r=r,
            t=t,
            mode=mode,
            d1_matlab=d1_matlab,
            d2_matlab=d2_matlab,
        )
        rectifier = StereoRectifier(
            rectification_params=params,
            interpolation=interpolation,
        )
        for pair_idx, (file1, file2) in enumerate(zip(image_files1, image_files2), start=1):
            left_path = os.path.join(input_folder1, file1)
            right_path = os.path.join(input_folder2, file2)

            packet = rectifier.rectify_frame_packet(
                left_path=left_path,
                right_path=right_path,
                output_folder1=output_folder1,
                output_folder2=output_folder2,
                frame_id=pair_idx,
                include_images=False,
                save_images=True,
                image_ext="jpg",
                append_rect_suffix=False,
            )
            print(f"saved rectified pair: {packet.left_rect_path}, {packet.right_rect_path}")

        print("stereo rectification complete")
        return True

    except Exception as e:
        print(f"stereo rectification failed: {str(e)}")
        return False

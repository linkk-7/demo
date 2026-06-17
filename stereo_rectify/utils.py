# 参数加载、读图、输出路径推断等辅助函数

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from .models import RectificationParams


def load_rectification_params(
    calibration_file1: str,
    calibration_file2: str,
    r: str,
    t: str,
    mode: str,
    d1_matlab: Optional[str] = None,
    d2_matlab: Optional[str] = None,
    calibration_tag: Optional[str] = None,
) -> RectificationParams:
    if mode == "python" and os.path.exists(calibration_file1) and os.path.exists(calibration_file2):
        calibration_data1 = np.load(calibration_file1, allow_pickle=True).item()
        calibration_data2 = np.load(calibration_file2, allow_pickle=True).item()
        camera_matrix_1 = calibration_data1["cameraMatrix"]
        dist_coeffs_1 = calibration_data1["distCoeffs"]
        camera_matrix_2 = calibration_data2["cameraMatrix"]
        dist_coeffs_2 = calibration_data2["distCoeffs"]
        rotation = np.load(r, allow_pickle=True).item()["rotation"]
        translation = np.load(t, allow_pickle=True).item()["trans"]
        return RectificationParams(
            camera_matrix_1=camera_matrix_1,
            dist_coeffs_1=dist_coeffs_1,
            camera_matrix_2=camera_matrix_2,
            dist_coeffs_2=dist_coeffs_2,
            rotation=rotation,
            translation=translation,
            calibration_tag=calibration_tag,
        )

    if mode == "matlab" and os.path.exists(calibration_file1) and os.path.exists(calibration_file2):
        if d1_matlab is None or d2_matlab is None:
            raise ValueError("matlab mode requires d1_matlab and d2_matlab.")

        camera_matrix_1 = np.load(calibration_file1, allow_pickle=True)
        camera_matrix_2 = np.load(calibration_file2, allow_pickle=True)
        dist_coeffs_1 = np.load(d1_matlab, allow_pickle=True)
        dist_coeffs_2 = np.load(d2_matlab, allow_pickle=True)
        rotation = np.load(r, allow_pickle=True)
        translation = np.load(t, allow_pickle=True)
        return RectificationParams(
            camera_matrix_1=camera_matrix_1,
            dist_coeffs_1=dist_coeffs_1,
            camera_matrix_2=camera_matrix_2,
            dist_coeffs_2=dist_coeffs_2,
            rotation=rotation,
            translation=translation,
            calibration_tag=calibration_tag,
        )

    raise FileNotFoundError("calibration files not found or mode is invalid.")


def read_image(path: str, side: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"failed to read {side} image: {path}")
    return image


def infer_output_dirs(
    frame_packet,
    left_path: Optional[str],
    right_path: Optional[str],
    output_folder1: Optional[str],
    output_folder2: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Backward-compatible output inference.
    Prefer explicit output folders in new main pipeline.
    """
    if output_folder1 and output_folder2:
        return output_folder1, output_folder2

    candidate_left = left_path
    candidate_right = right_path
    if frame_packet is not None:
        candidate_left = candidate_left or getattr(frame_packet, "left_path", None)
        candidate_right = candidate_right or getattr(frame_packet, "right_path", None)

    if not candidate_left or not candidate_right:
        return output_folder1, output_folder2

    left_parent = os.path.dirname(candidate_left)
    right_parent = os.path.dirname(candidate_right)
    if os.path.basename(left_parent).lower() == "left" and os.path.basename(right_parent).lower() == "right":
        base = os.path.dirname(left_parent)
        return os.path.join(base, "left_rec"), os.path.join(base, "right_rec")

    return output_folder1, output_folder2


def build_output_path(
    output_dir: str,
    source_path: Optional[str],
    prefix: str,
    frame_id: int,
    image_ext: str,
    append_rect_suffix: bool,
) -> str:
    if source_path:
        stem = os.path.splitext(os.path.basename(source_path))[0]
    else:
        stem = f"{prefix}_{frame_id:06d}"
    if append_rect_suffix:
        stem = f"{stem}_rect"
    return os.path.join(output_dir, f"{stem}.{image_ext}")


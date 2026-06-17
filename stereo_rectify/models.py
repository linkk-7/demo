# 放数据结构，避免和算法/IO混在一起

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class RectificationParams:
    """Stereo rectification calibration parameters."""

    camera_matrix_1: np.ndarray
    dist_coeffs_1: np.ndarray
    camera_matrix_2: np.ndarray
    dist_coeffs_2: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    calibration_tag: Optional[str] = None


@dataclass(slots=True)
class RectifiedPacket:
    """Single-pair rectification result packet for main pipeline."""

    frame_id: int
    left_rect_path: Optional[str]
    right_rect_path: Optional[str]
    left_rect_image: Optional[np.ndarray] = None
    right_rect_image: Optional[np.ndarray] = None
    q_matrix: Optional[np.ndarray] = None
    calibration_tag: Optional[str] = None


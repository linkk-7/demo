"""Single-frame stereo 3D and displacement helpers for online monitoring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CalibrationData:
    rotation: np.ndarray
    translation: np.ndarray
    left_intrinsic: np.ndarray
    right_intrinsic: np.ndarray
    q_matrix: Optional[np.ndarray] = None


def _load_npy_item(path: str, key: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    try:
        item = data.item()
    except ValueError as exc:
        raise ValueError(f"{path} is not a dict-like npy file.") from exc
    if key not in item:
        raise KeyError(f"{path} missing key: {key}")
    return np.asarray(item[key], dtype=float)


@lru_cache(maxsize=8)
def load_calibration_data(calibration_folder: str) -> CalibrationData:
    """Load calibration matrices used by the legacy displacement calculation."""
    folder = os.fspath(calibration_folder)
    rotation = _load_npy_item(os.path.join(folder, "R.npy"), "rotation")
    translation = _load_npy_item(os.path.join(folder, "T.npy"), "trans").reshape(-1)
    left_intrinsic = _load_npy_item(os.path.join(folder, "calibration_1.npy"), "cameraMatrix")
    right_intrinsic = _load_npy_item(os.path.join(folder, "calibration_2.npy"), "cameraMatrix")

    q_matrix = None
    q_path = os.path.join(folder, "Q.npy")
    if os.path.exists(q_path):
        try:
            q_matrix = _load_npy_item(q_path, "disparity_to_depth")
        except Exception:
            q_matrix = None

    return CalibrationData(
        rotation=rotation,
        translation=translation,
        left_intrinsic=left_intrinsic,
        right_intrinsic=right_intrinsic,
        q_matrix=q_matrix,
    )


def compute_xyz_from_stereo_points(
    left_xy: Sequence[float],
    right_xy: Sequence[float],
    calibration_folder: Optional[str] = None,
    q_matrix: Optional[np.ndarray] = None,
    min_disparity: float = 1e-5,
) -> np.ndarray:
    """Compute one 3D point from rectified left/right image coordinates.

    The calibration-folder path copies the minimal single-frame formula from
    ``5_calculate_disp.py::cal3d3point`` so online results stay aligned with the
    current legacy displacement script.
    """
    left_x, left_y = float(left_xy[0]), float(left_xy[1])
    right_x, _right_y = float(right_xy[0]), float(right_xy[1])
    disparity = left_x - right_x
    if abs(disparity) <= float(min_disparity):
        raise ValueError(f"invalid disparity={disparity:.6f}")

    if calibration_folder is None and q_matrix is not None:
        point = np.asarray([left_x, left_y, disparity, 1.0], dtype=float)
        xyz_w = np.asarray(q_matrix, dtype=float) @ point
        if abs(float(xyz_w[3])) <= 1e-12:
            raise ValueError("Q reprojection produced zero homogeneous scale.")
        xyz = xyz_w[:3] / xyz_w[3]
    else:
        if calibration_folder is None:
            raise ValueError("calibration_folder is required when q_matrix is not used.")
        calib = load_calibration_data(os.fspath(calibration_folder))
        left_intrinsic = calib.left_intrinsic
        right_intrinsic = calib.right_intrinsic
        xl_norm = (left_x - left_intrinsic[0, 2]) / left_intrinsic[0, 0]
        yl_norm = (left_y - left_intrinsic[1, 2]) / left_intrinsic[1, 1]
        fl = (left_intrinsic[0, 0] + left_intrinsic[1, 1]) / 2.0
        _fr = (right_intrinsic[0, 0] + right_intrinsic[1, 1]) / 2.0
        baseline = abs(float(calib.translation.reshape(-1)[0]))
        legacy_disparity = right_x - left_x
        if abs(legacy_disparity) <= float(min_disparity):
            raise ValueError(f"invalid legacy disparity={legacy_disparity:.6f}")
        z = fl * baseline / legacy_disparity
        x = z * xl_norm / fl * left_intrinsic[0, 0]
        y = z * yl_norm / fl * left_intrinsic[1, 1]
        xyz = np.asarray([x, y, z], dtype=float)

    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError(f"invalid xyz computed from stereo points: {xyz}")
    return xyz


def compute_xyz_from_stereo_points_temp_calibration(
    left_xy: Sequence[float],
    right_xy: Sequence[float],
    calibration_folder: str,
    baseline_mm: float = 70.0,
    min_disparity: float = 1e-5,
) -> np.ndarray:
    """Compute temporary demo 3D using old intrinsics and measured baseline.

    This mode is for platform/demo validation when a real stereo calibration is
    not available yet. It uses the left camera intrinsics, a manually measured
    baseline, and absolute disparity so left/right ordering does not produce a
    negative depth. Do not use it as final measurement calibration.
    """
    left_x, left_y = float(left_xy[0]), float(left_xy[1])
    right_x = float(right_xy[0])
    disparity = abs(right_x - left_x)
    if disparity <= float(min_disparity):
        raise ValueError(f"invalid temporary disparity={disparity:.6f}")

    calib = load_calibration_data(os.fspath(calibration_folder))
    left_intrinsic = calib.left_intrinsic
    fx = float(left_intrinsic[0, 0])
    fy = float(left_intrinsic[1, 1])
    cx = float(left_intrinsic[0, 2])
    cy = float(left_intrinsic[1, 2])
    baseline = abs(float(baseline_mm))
    if baseline <= 0:
        raise ValueError(f"invalid temporary baseline={baseline_mm}")

    z = fx * baseline / disparity
    x = (left_x - cx) * z / fx
    y = (left_y - cy) * z / fy
    xyz = np.asarray([x, y, z], dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError(f"invalid temporary xyz computed from stereo points: {xyz}")
    return xyz


def compute_displacement_from_points(
    left_xy: Sequence[float],
    right_xy: Sequence[float],
    reference_xyz: Sequence[float],
    calibration_folder: Optional[str] = None,
    q_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return current xyz and displacement relative to reference_xyz."""
    xyz = compute_xyz_from_stereo_points(
        left_xy=left_xy,
        right_xy=right_xy,
        calibration_folder=calibration_folder,
        q_matrix=q_matrix,
    )
    displacement = xyz - np.asarray(reference_xyz, dtype=float)
    return xyz, displacement


def median_displacement(displacements: Sequence[Sequence[float]]) -> np.ndarray:
    if not displacements:
        raise ValueError("no valid displacements.")
    arr = np.asarray(displacements, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected Nx3 displacements, got shape={arr.shape}")
    return np.median(arr, axis=0)

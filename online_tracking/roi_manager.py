"""ROI runtime configuration helpers for online tracking."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

from .config_models import (
    CameraInfo,
    ImageInfo,
    InitPolicy,
    ROIConfig,
    ROIRegion,
    ReferenceInfo,
    ValidationResult,
)


def ensure_runtime_dirs(
    runtime_dir: str = "runtime_state",
    reference_subdir: str = "roi_reference",
) -> Tuple[str, str]:
    """Create runtime directories used by ROI configuration."""
    reference_dir = os.path.join(runtime_dir, reference_subdir)
    os.makedirs(runtime_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)
    return runtime_dir, reference_dir


def save_roi_config(
    config: ROIConfig,
    path: str = os.path.join("runtime_state", "roi_config.json"),
) -> None:
    """Save ROIConfig to JSON."""
    config.save(path)


def load_roi_config(
    path: str = os.path.join("runtime_state", "roi_config.json"),
) -> ROIConfig:
    """Load ROIConfig from JSON."""
    return ROIConfig.load(path)


def validate_roi_config(
    config: ROIConfig,
    image_width: int,
    image_height: int,
    calibration_tag: Optional[str] = None,
    calibration_folder: Optional[str] = None,
) -> ValidationResult:
    """Validate ROIConfig against the current image and calibration metadata."""
    return config.validate_for_image(
        width=image_width,
        height=image_height,
        calibration_tag=calibration_tag,
        calibration_folder=calibration_folder,
    )


def save_reference_images(
    left_rect_image: np.ndarray,
    right_rect_image: Optional[np.ndarray] = None,
    reference_dir: str = os.path.join("runtime_state", "roi_reference"),
    prefix: str = "roi_reference",
) -> Tuple[str, Optional[str]]:
    """Save rectified reference images and return their paths."""
    if left_rect_image is None:
        raise ValueError("left_rect_image is None.")
    if not isinstance(left_rect_image, np.ndarray):
        raise ValueError("left_rect_image must be a numpy.ndarray.")
    if right_rect_image is not None and not isinstance(right_rect_image, np.ndarray):
        raise ValueError("right_rect_image must be a numpy.ndarray or None.")

    os.makedirs(reference_dir, exist_ok=True)
    left_path = os.path.join(reference_dir, f"{prefix}_left_rect.jpg")
    right_path = (
        os.path.join(reference_dir, f"{prefix}_right_rect.jpg")
        if right_rect_image is not None
        else None
    )

    if not cv2.imwrite(left_path, left_rect_image):
        raise IOError(f"failed to save left reference image: {left_path}")
    if right_rect_image is not None and right_path is not None:
        if not cv2.imwrite(right_path, right_rect_image):
            raise IOError(f"failed to save right reference image: {right_path}")

    return left_path, right_path


def select_roi_opencv(
    image: np.ndarray,
    window_name: str = "Select ROI",
    max_display_width: int = 1280,
    max_display_height: int = 720,
) -> ROIRegion:
    """Select ROI interactively with OpenCV and return it as ROIRegion."""
    if image is None:
        raise ValueError("image is None.")
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy.ndarray.")

    image_h, image_w = image.shape[:2]
    scale = min(
        float(max_display_width) / float(image_w),
        float(max_display_height) / float(image_h),
        1.0,
    )
    display_image = image
    if scale < 1.0:
        display_size = (
            max(1, int(round(image_w * scale))),
            max(1, int(round(image_h * scale))),
        )
        display_image = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)
        print(
            "[roi] selectROI display scaled: "
            f"{image_w}x{image_h} -> {display_size[0]}x{display_size[1]}, "
            f"scale={scale:.4f}"
        )

    x, y, w, h = cv2.selectROI(
        window_name,
        display_image,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow(window_name)
    if scale < 1.0:
        roi = ROIRegion(
            x=int(round(float(x) / scale)),
            y=int(round(float(y) / scale)),
            w=int(round(float(w) / scale)),
            h=int(round(float(h) / scale)),
        ).clip_to_image(width=image_w, height=image_h)
    else:
        roi = ROIRegion(x=int(x), y=int(y), w=int(w), h=int(h))
    if not roi.is_valid():
        raise ValueError("selected ROI is invalid or empty.")
    return roi


def create_roi_config_from_selection(
    left_rect_image: np.ndarray,
    right_rect_image: Optional[np.ndarray] = None,
    roi_left: Optional[ROIRegion] = None,
    calibration_tag: Optional[str] = None,
    calibration_folder: Optional[str] = None,
    camera_info: Optional[CameraInfo] = None,
    init_policy: Optional[InitPolicy] = None,
    runtime_dir: str = "runtime_state",
    reference_frame_id: Optional[int] = None,
    save_reference_images_flag: bool = True,
) -> ROIConfig:
    """Create an ROIConfig from current rectified images and an ROI selection."""
    if left_rect_image is None:
        raise ValueError("left_rect_image is None.")
    if not isinstance(left_rect_image, np.ndarray):
        raise ValueError("left_rect_image must be a numpy.ndarray.")
    if right_rect_image is not None and not isinstance(right_rect_image, np.ndarray):
        raise ValueError("right_rect_image must be a numpy.ndarray or None.")

    if roi_left is None:
        roi_left = select_roi_opencv(left_rect_image)

    image_height, image_width = left_rect_image.shape[:2]
    clipped_roi = roi_left.clip_to_image(width=image_width, height=image_height)
    if not clipped_roi.is_valid():
        raise ValueError("roi_left is invalid after clipping to image bounds.")

    _, reference_dir = ensure_runtime_dirs(runtime_dir=runtime_dir)

    left_reference_path = None
    right_reference_path = None
    if save_reference_images_flag:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        left_reference_path, right_reference_path = save_reference_images(
            left_rect_image=left_rect_image,
            right_rect_image=right_rect_image,
            reference_dir=reference_dir,
            prefix=f"roi_reference_{timestamp}",
        )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    return ROIConfig(
        version=1,
        created_time=now,
        updated_time=now,
        calibration_tag=calibration_tag,
        calibration_folder=calibration_folder,
        image=ImageInfo(
            coordinate_space="rectified",
            source="left_rect_image",
            width=image_width,
            height=image_height,
        ),
        roi_left=clipped_roi,
        roi_right=None,
        camera=camera_info or CameraInfo(),
        init_policy=init_policy or InitPolicy(),
        reference=ReferenceInfo(
            reference_frame_id=reference_frame_id,
            save_reference_images=save_reference_images_flag,
            left_rect_image_path=left_reference_path,
            right_rect_image_path=right_reference_path,
        ),
    )

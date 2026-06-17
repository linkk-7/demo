"""Data models for TCP image uploading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


TimestampLike = Union[int, float, str]
ScalarLike = Union[int, float, str]


@dataclass
class ImageUploadConfig:
    """Runtime configuration for image upload and sensor-param sync."""

    host: str = "t4.tncet.com"
    upload_port: int = 9812
    config_port: int = 9879
    auth_code: str = ""
    imei: str = "F44EB4DF6F99"
    heartbeat_interval_sec: float = 15.0
    left_sensor_param_id: Optional[str] = None
    right_sensor_param_id: Optional[str] = None
    jpeg_quality: int = 20
    connect_timeout: float = 5.0
    send_timeout: float = 5.0


@dataclass
class ImageUploadTask:
    """Single upload task.

    Notes:
    - `sensor_param_id` should come from BIM3/manual config/9879 sync logic.
    - `image` is expected to be an OpenCV ndarray.
    """

    frame_id: Optional[int]
    sensor_param_id: str
    image: np.ndarray
    timestamp: TimestampLike
    bad: ScalarLike = 0
    displacement_value: ScalarLike = 0
    side: Optional[str] = None

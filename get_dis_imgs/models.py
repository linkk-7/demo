from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class FrameMeta:
    dev_timestamp_raw: int
    host_timestamp: int
    frame_num: int


@dataclass(slots=True)
class FramePacket:
    frame_id: int
    left_host_timestamp: int
    right_host_timestamp: int
    left_dev_timestamp_raw: int
    right_dev_timestamp_raw: int
    left_frame_num: int
    right_frame_num: int
    left_filename: str
    right_filename: str
    left_path: str
    right_path: str
    left_image: Optional[np.ndarray] = None
    right_image: Optional[np.ndarray] = None


@dataclass(slots=True)
class RetentionRecord:
    frame_id: int
    left_path: str
    right_path: str

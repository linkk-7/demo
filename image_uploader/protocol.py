"""Protocol helpers for external TCP image interface."""

from __future__ import annotations

import base64
from typing import Union

import cv2
import numpy as np

# Reuse existing project implementation first.
try:
    from utils.byte_utils import get_length_prefix_bytes as _project_length_prefix
except Exception:  # pragma: no cover
    _project_length_prefix = None


def encode_jpg_base64(image: np.ndarray, jpeg_quality: int = 20) -> bytes:
    """Encode ndarray image as JPG bytes then base64 bytes.

    Raises:
    - ValueError: invalid input image
    - RuntimeError: JPG encoding failed
    """
    if image is None:
        raise ValueError("image must not be None.")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image)}.")
    if image.size == 0:
        raise ValueError("image is empty.")

    quality = int(jpeg_quality)
    quality = max(0, min(100, quality))
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok or buffer is None:
        raise RuntimeError("cv2.imencode('.jpg', image) failed.")
    return base64.b64encode(buffer.tobytes())


def build_image_payload(
    auth_code: str,
    sensor_param_id: Union[str, int],
    timestamp: Union[str, int, float],
    bad: Union[str, int, float],
    displacement_value: Union[str, int, float],
    jpg_base64: bytes,
) -> bytes:
    """Build payload bytes:
    IMAGE-JPG&&{authCode}&&{sensor_param_id}&&{timestamp}&&{bad}&&{displacement_value}&& + jpg_base64
    """
    if jpg_base64 is None or len(jpg_base64) == 0:
        raise ValueError("jpg_base64 must not be empty.")

    head = (
        f"IMAGE-JPG&&{auth_code}&&{sensor_param_id}&&{timestamp}&&"
        f"{bad}&&{displacement_value}&&"
    )
    return head.encode("utf-8") + jpg_base64


def get_length_prefix_bytes(payload: bytes) -> bytes:
    """Apply length-prefix framing.

    Priority:
    1) Reuse project function `utils.byte_utils.get_length_prefix_bytes`.
    2) Fallback to a conservative 4-byte big-endian implementation.

    TODO:
    - Confirm with platform whether the framing is exactly 4-byte big-endian length.
    """
    if payload is None:
        raise ValueError("payload must not be None.")
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"payload must be bytes-like, got {type(payload)}.")

    if _project_length_prefix is not None:
        return _project_length_prefix(bytes(payload))

    # Fallback only when shared utility is unavailable.
    import struct

    body = bytes(payload)
    return struct.pack("!I", len(body)) + body


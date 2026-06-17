"""Synchronous TCP client for IMAGE-JPG upload protocol."""

from __future__ import annotations

import socket
from typing import Optional, Union

import numpy as np

from .models import ImageUploadConfig, ImageUploadTask
from .protocol import (
    build_image_payload,
    encode_jpg_base64,
    get_length_prefix_bytes,
)


class ImageUploadClient:
    """Simple and reliable sync uploader.

    Design:
    - Keep sync send path simple.
    - Auto-connect on first send.
    - Do not auto-retry sendall, because a failed send may have partially reached
      the platform and the test authCode has a limited upload quota.
    """

    def __init__(self, config: ImageUploadConfig) -> None:
        self.config = config
        self._sock: Optional[socket.socket] = None

    def connect(self) -> None:
        """Connect to upload server."""
        self.close()
        sock: Optional[socket.socket] = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(float(self.config.connect_timeout))
            sock.connect((self.config.host, int(self.config.upload_port)))
            sock.settimeout(float(self.config.send_timeout))
            self._sock = sock
        except OSError as exc:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
            raise ConnectionError(
                f"Failed to connect upload server {self.config.host}:{self.config.upload_port}: {exc}"
            ) from exc

    def close(self) -> None:
        """Close socket safely."""
        if self._sock is None:
            return
        try:
            self._sock.close()
        finally:
            self._sock = None

    def _ensure_connected(self) -> None:
        if self._sock is None:
            self.connect()

    def send_image(self, task: ImageUploadTask) -> None:
        """Encode and send one image task."""
        if not task.sensor_param_id:
            raise ValueError("task.sensor_param_id must not be empty.")

        jpg_base64 = encode_jpg_base64(
            image=task.image,
            jpeg_quality=self.config.jpeg_quality,
        )
        payload = build_image_payload(
            auth_code=self.config.auth_code,
            sensor_param_id=task.sensor_param_id,
            timestamp=task.timestamp,
            bad=task.bad,
            displacement_value=task.displacement_value,
            jpg_base64=jpg_base64,
        )
        packet = get_length_prefix_bytes(payload)

        self._ensure_connected()
        try:
            self._sock.sendall(packet)  # type: ignore[union-attr]
        except OSError as exc:
            self.close()
            raise ConnectionError(f"sendall failed: {exc}") from exc

    def send_array(
        self,
        image: np.ndarray,
        sensor_param_id: Union[str, int],
        timestamp: Union[str, int, float],
        bad: Union[str, int, float] = 0,
        displacement_value: Union[str, int, float] = 0,
        frame_id: Optional[int] = None,
        side: Optional[str] = None,
    ) -> None:
        """Convenience wrapper for direct ndarray upload."""
        task = ImageUploadTask(
            frame_id=frame_id,
            sensor_param_id=str(sensor_param_id),
            image=image,
            timestamp=timestamp,
            bad=bad,
            displacement_value=displacement_value,
            side=side,
        )
        self.send_image(task)

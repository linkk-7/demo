"""Upload one local image file to platform TCP image interface.

Run:
    python test_image_upload_from_file.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from image_uploader import ImageUploadClient, ImageUploadConfig


# =========================
# User config (edit first)
# =========================
IMAGE_PATH = ""
AUTH_CODE = "xian_beilin_test"
SENSOR_PARAM_ID = ""  # fill your real sensor_param_id
HOST = "t4.tncet.com"
PORT = 9812
JPEG_QUALITY = 20
CONFIRM_SEND = False


def print_config() -> None:
    print("Platform image upload config:")
    print(f"  host: {HOST}")
    print(f"  port: {PORT}")
    print(f"  sensor_param_id: {SENSOR_PARAM_ID!r}")
    print(f"  image_path: {IMAGE_PATH!r}")
    print(f"  jpeg_quality: {JPEG_QUALITY}")
    print(f"  confirm_send: {CONFIRM_SEND}")


def main() -> None:
    print_config()

    if not CONFIRM_SEND:
        print(
            "当前不会发送。确认 sensor_param_id 和图片路径后，将 CONFIRM_SEND 改为 True 再运行。"
        )
        return

    if not SENSOR_PARAM_ID:
        raise ValueError("Please set SENSOR_PARAM_ID in script before running.")
    if not IMAGE_PATH:
        raise ValueError("Please set IMAGE_PATH in script before running.")
    if not Path(IMAGE_PATH).is_file():
        raise FileNotFoundError(f"IMAGE_PATH does not exist: {IMAGE_PATH}")

    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cv2.imread failed: {IMAGE_PATH}")

    config = ImageUploadConfig(
        host=HOST,
        upload_port=PORT,
        auth_code=AUTH_CODE,
        jpeg_quality=JPEG_QUALITY,
    )
    client = ImageUploadClient(config=config)
    try:
        client.send_array(
            image=image,
            sensor_param_id=SENSOR_PARAM_ID,
            timestamp=int(time.time() * 1000),
            bad=0,
            displacement_value=0,
            frame_id=None,
            side=None,
        )
        print("Upload sent successfully.")
    finally:
        client.close()


if __name__ == "__main__":
    main()

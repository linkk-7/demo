"""Send exactly one local image to the platform TCP image interface.

Edit the configuration section first, then run:
    python send_one_image_to_platform.py
"""

from __future__ import annotations

import os
import time

import cv2

from image_uploader import ImageUploadClient, ImageUploadConfig


# =========================
# User config (edit first)
# =========================
HOST = "t4.tncet.com"
PORT = 9812
AUTH_CODE = "xian_beilin_test"
SENSOR_PARAM_ID = ""
IMAGE_PATH = "new_data5/cab/Camera_0/1r.bmp"
JPEG_QUALITY = 20
BAD = 0
DISPLACEMENT_VALUE = 0
CONFIRM_SEND = True


def print_config() -> None:
    print("Platform image upload config:")
    print(f"  host: {HOST}")
    print(f"  port: {PORT}")
    print(f"  sensor_param_id: {SENSOR_PARAM_ID!r}")
    print(f"  image_path: {IMAGE_PATH!r}")
    print(f"  jpeg_quality: {JPEG_QUALITY}")
    print(f"  bad: {BAD}")
    print(f"  displacement_value: {DISPLACEMENT_VALUE}")
    print(f"  confirm_send: {CONFIRM_SEND}")


def main() -> None:
    print_config()

    if not CONFIRM_SEND:
        print(
            "当前不会发送。确认 sensor_param_id 和图片路径后，将 CONFIRM_SEND 改为 True 再运行。"
        )
        return

    if not SENSOR_PARAM_ID:
        raise ValueError("Please set SENSOR_PARAM_ID before sending.")
    if not IMAGE_PATH:
        raise ValueError("Please set IMAGE_PATH before sending.")
    if not os.path.isfile(IMAGE_PATH):
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
            bad=BAD,
            displacement_value=DISPLACEMENT_VALUE,
        )
    finally:
        client.close()

    print("send finished, please check platform")


if __name__ == "__main__":
    main()

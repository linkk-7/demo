import os
import threading
from collections import deque
from ctypes import *
from typing import Any, Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from MvImport.CameraParams_const import *
from MvImport.CameraParams_header import *
from MvImport.MvCameraControl_class import *
from MvImport.MvErrorDefine_const import *
from MvImport.PixelType_header import *

from .models import FrameMeta, FramePacket, RetentionRecord
from .save_utils import AsyncImageSaver, PairSaveTask, imwrite_image, normalize_image_ext


class CameraController:
    """双目采集核心控制器。

    作用：
    - 打开相机并设置关键参数（含 30fps 相关设置）；
    - 抓取左右图像并转为 BGR 数组；
    - 生成 FramePacket；
    - 在 save_images=True 时执行可选落盘与 retention。
    """

    def __init__(
        self,
        save_folder_base: str,
        keep_last_n_pairs: Optional[int] = None,
        exposure_time_us: float = 8000.0,
        target_acquisition_fps: float = 30.0,
        log_camera_nodes: bool = False,
    ):
        self.save_folder_base = save_folder_base
        self.keep_last_n_pairs = keep_last_n_pairs if keep_last_n_pairs and keep_last_n_pairs > 0 else None
        self._retention_records: Deque[RetentionRecord] = deque()
        self._retention_lock = threading.Lock()

        self.exposure_time_us = exposure_time_us
        self.target_acquisition_fps = target_acquisition_fps
        self.log_camera_nodes = log_camera_nodes

    # ---------- 文件系统工具 ----------
    @staticmethod
    def _ensure_dir(path: str):
        """确保目录存在。"""
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def _safe_remove(path: str):
        """安全删除文件，删除失败时静默跳过。"""
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    @staticmethod
    def _normalize_image_ext(image_ext: str) -> str:
        """标准化扩展名（png/jpg/jpeg）。"""
        return normalize_image_ext(image_ext)

    @staticmethod
    def _format_pair_filename(prefix: str, host_timestamp: int, frame_id: int, image_ext: str = "png") -> str:
        """生成统一命名：L/R_时间戳_帧号.后缀。"""
        return f"{prefix}_{host_timestamp}_{frame_id:06d}.{image_ext}"

    def _build_pair_paths(
        self,
        frame_id: int,
        left_host_timestamp: int,
        right_host_timestamp: int,
        image_ext: str,
        create_dirs: bool = True,
    ) -> Tuple[str, str, str, str]:
        """构造左右图文件名与路径。

        - create_dirs=True：会创建 left/right 目录（用于真实保存）。
        - create_dirs=False：仅生成潜在路径字符串（在线模式常用）。
        """
        ext = self._normalize_image_ext(image_ext)
        left_dir = os.path.join(self.save_folder_base, "left")
        right_dir = os.path.join(self.save_folder_base, "right")
        if create_dirs:
            self._ensure_dir(left_dir)
            self._ensure_dir(right_dir)

        left_filename = self._format_pair_filename("L", left_host_timestamp, frame_id, ext)
        right_filename = self._format_pair_filename("R", right_host_timestamp, frame_id, ext)
        left_path = os.path.join(left_dir, left_filename)
        right_path = os.path.join(right_dir, right_filename)
        return left_filename, right_filename, left_path, right_path

    def _write_pair_to_paths(self, task: PairSaveTask):
        """同步写盘：把一对图像写到 left_path/right_path。"""
        ok_left = imwrite_image(
            task.left_path,
            task.left_image,
            image_ext=task.image_ext,
            jpeg_quality=task.jpeg_quality,
            png_compression=task.png_compression,
        )
        ok_right = imwrite_image(
            task.right_path,
            task.right_image,
            image_ext=task.image_ext,
            jpeg_quality=task.jpeg_quality,
            png_compression=task.png_compression,
        )
        if not ok_left or not ok_right:
            if ok_left:
                self._safe_remove(task.left_path)
            if ok_right:
                self._safe_remove(task.right_path)
            raise IOError("Failed to save one or both paired images.")

    def _save_pair_images(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        frame_id: int,
        left_host_timestamp: int,
        right_host_timestamp: int,
        image_ext: str = "png",
        jpeg_quality: int = 90,
        png_compression: int = 1,
    ) -> Tuple[str, str, str, str]:
        """保存一对图并返回文件名/路径（供 benchmark 或调试使用）。"""
        left_filename, right_filename, left_path, right_path = self._build_pair_paths(
            frame_id=frame_id,
            left_host_timestamp=left_host_timestamp,
            right_host_timestamp=right_host_timestamp,
            image_ext=image_ext,
            create_dirs=True,
        )
        task = PairSaveTask(
            frame_id=frame_id,
            left_path=left_path,
            right_path=right_path,
            left_image=left_image,
            right_image=right_image,
            image_ext=self._normalize_image_ext(image_ext),
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
        )
        self._write_pair_to_paths(task)
        return left_filename, right_filename, left_path, right_path

    # ---------- retention ----------
    def _track_and_enforce_pair_retention(self, frame_id: int, left_path: str, right_path: str):
        """仅在保存模式下维护最近 N 对图像。"""
        if self.keep_last_n_pairs is None:
            return

        with self._retention_lock:
            self._retention_records.append(
                RetentionRecord(frame_id=frame_id, left_path=left_path, right_path=right_path)
            )
            while len(self._retention_records) > self.keep_last_n_pairs:
                old_record = self._retention_records.popleft()
                self._safe_remove(old_record.left_path)
                self._safe_remove(old_record.right_path)

    # ---------- 节点读写 ----------
    def _warn_node(self, msg: str):
        print(f"[WARN] {msg}")

    def safe_get_enum(self, cam, node: str, prefix: str = "") -> Optional[int]:
        """安全读取 Enum 节点值。"""
        try:
            st = MVCC_ENUMVALUE()
            ret = cam.MV_CC_GetEnumValue(node, st)
            if ret != 0:
                self._warn_node(f"{prefix} read enum {node} failed ret=0x{ret:x}")
                return None
            return int(st.nCurValue)
        except Exception as exc:
            self._warn_node(f"{prefix} read enum {node} exception: {exc}")
            return None

    def safe_get_float(self, cam, node: str, prefix: str = "") -> Optional[float]:
        """安全读取 Float 节点值。"""
        try:
            st = MVCC_FLOATVALUE()
            ret = cam.MV_CC_GetFloatValue(node, st)
            if ret != 0:
                self._warn_node(f"{prefix} read float {node} failed ret=0x{ret:x}")
                return None
            return float(st.fCurValue)
        except Exception as exc:
            self._warn_node(f"{prefix} read float {node} exception: {exc}")
            return None

    def safe_get_int(self, cam, node: str, prefix: str = "") -> Optional[int]:
        """安全读取 Int 节点值。"""
        try:
            st = MVCC_INTVALUE()
            ret = cam.MV_CC_GetIntValue(node, st)
            if ret != 0:
                self._warn_node(f"{prefix} read int {node} failed ret=0x{ret:x}")
                return None
            return int(st.nCurValue)
        except Exception as exc:
            self._warn_node(f"{prefix} read int {node} exception: {exc}")
            return None

    def safe_get_bool(self, cam, node: str, prefix: str = "") -> Optional[bool]:
        """安全读取 Bool 节点值。"""
        try:
            val = c_bool(False)
            ret = cam.MV_CC_GetBoolValue(node, val)
            if ret != 0:
                self._warn_node(f"{prefix} read bool {node} failed ret=0x{ret:x}")
                return None
            return bool(val.value)
        except Exception as exc:
            self._warn_node(f"{prefix} read bool {node} exception: {exc}")
            return None

    def safe_set_enum(self, cam, node: str, value: int, prefix: str = ""):
        """安全设置 Enum 节点，失败仅告警。"""
        try:
            ret = cam.MV_CC_SetEnumValue(node, int(value))
            if ret != 0:
                self._warn_node(f"{prefix} set enum {node}={value} failed ret=0x{ret:x}")
        except Exception as exc:
            self._warn_node(f"{prefix} set enum {node}={value} exception: {exc}")

    def safe_set_float(self, cam, node: str, value: float, prefix: str = ""):
        """安全设置 Float 节点，失败仅告警。"""
        try:
            ret = cam.MV_CC_SetFloatValue(node, float(value))
            if ret != 0:
                self._warn_node(f"{prefix} set float {node}={value} failed ret=0x{ret:x}")
        except Exception as exc:
            self._warn_node(f"{prefix} set float {node}={value} exception: {exc}")

    def safe_set_bool(self, cam, node: str, value: bool, prefix: str = ""):
        """安全设置 Bool 节点，失败仅告警。"""
        try:
            ret = cam.MV_CC_SetBoolValue(node, bool(value))
            if ret != 0:
                self._warn_node(f"{prefix} set bool {node}={value} failed ret=0x{ret:x}")
        except Exception as exc:
            self._warn_node(f"{prefix} set bool {node}={value} exception: {exc}")

    def log_camera_status(self, cam, prefix: str = ""):
        """打印关键相机节点状态。"""
        rows: List[Tuple[str, str, Any]] = [
            ("TriggerMode", "enum"),
            ("ExposureAuto", "enum"),
            ("ExposureTime", "float"),
            ("GainAuto", "enum"),
            ("Gain", "float"),
            ("AcquisitionFrameRateEnable", "bool"),
            ("AcquisitionFrameRate", "float"),
            ("PixelFormat", "enum"),
            ("Width", "int"),
            ("Height", "int"),
            ("OffsetX", "int"),
            ("OffsetY", "int"),
        ]

        label = prefix.strip() if prefix else "camera"
        print(f"{label} status:")
        for node_name, node_type in rows:
            if node_type == "enum":
                value = self.safe_get_enum(cam, node_name, prefix=label)
            elif node_type == "float":
                value = self.safe_get_float(cam, node_name, prefix=label)
            elif node_type == "int":
                value = self.safe_get_int(cam, node_name, prefix=label)
            else:
                value = self.safe_get_bool(cam, node_name, prefix=label)

            if value is None:
                print(f"  - {node_name}: <unavailable>")
            else:
                print(f"  - {node_name}: {value}")

    # ---------- 像素转换 ----------
    @staticmethod
    def _to_bgr(cam, stOutFrame, data_buf):
        """把 SDK 原始帧缓冲转换为 BGR ndarray。"""
        w = stOutFrame.stFrameInfo.nWidth
        h = stOutFrame.stFrameInfo.nHeight
        size = stOutFrame.stFrameInfo.nFrameLen
        pix = stOutFrame.stFrameInfo.enPixelType

        arr1d = np.frombuffer(data_buf, dtype=np.uint8, count=size)

        if pix == PixelType_Gvsp_Mono8:
            gray = arr1d.reshape(h, w)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if pix == PixelType_Gvsp_BayerRG8:
            raw = arr1d.reshape(h, w)
            return cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)
        if pix == PixelType_Gvsp_BayerBG8:
            raw = arr1d.reshape(h, w)
            return cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)
        if pix == PixelType_Gvsp_BayerGB8:
            raw = arr1d.reshape(h, w)
            return cv2.cvtColor(raw, cv2.COLOR_BayerGB2BGR)
        if pix == PixelType_Gvsp_BayerGR8:
            raw = arr1d.reshape(h, w)
            return cv2.cvtColor(raw, cv2.COLOR_BayerGR2BGR)

        try:
            st = MV_CC_PIXEL_CONVERT_PARAM()
            st.nWidth = w
            st.nHeight = h
            st.pSrcData = cast(data_buf, c_void_p)
            st.nSrcDataLen = size
            st.enSrcPixelType = pix
            st.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            dst_len = w * h * 3
            dst_buf = (c_ubyte * dst_len)()
            st.pDstBuffer = cast(dst_buf, c_void_p)
            st.nDstBufferSize = dst_len
            ret = cam.MV_CC_ConvertPixelType(st)
            if ret == 0:
                return np.frombuffer(dst_buf, dtype=np.uint8, count=dst_len).reshape(h, w, 3)
        except Exception as exc:
            print("ConvertPixelType failed, use gray fallback:", exc)

        gray = arr1d.reshape(h, w)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _extract_frame_meta(st_frame_info) -> FrameMeta:
        """从 SDK 帧头提取时间戳和帧号元信息。"""
        dev_timestamp = (int(st_frame_info.nDevTimeStampHigh) << 32) | int(st_frame_info.nDevTimeStampLow)
        return FrameMeta(
            dev_timestamp_raw=dev_timestamp,
            host_timestamp=int(st_frame_info.nHostTimeStamp),
            frame_num=int(st_frame_info.nFrameNum),
        )

    def _get_image(self, cam, convert_to_bgr: bool = True):
        """抓取单帧，返回 (图像/缓冲, FrameMeta)。"""
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 2000)
        if ret != 0 or not stOutFrame.pBufAddr:
            return None, None

        size = stOutFrame.stFrameInfo.nFrameLen
        data_buf = (c_ubyte * size)()
        cdll.msvcrt.memcpy(byref(data_buf), stOutFrame.pBufAddr, size)
        cam.MV_CC_FreeImageBuffer(stOutFrame)

        frame_meta = self._extract_frame_meta(stOutFrame.stFrameInfo)
        if convert_to_bgr:
            img = self._to_bgr(cam, stOutFrame, data_buf)
        else:
            img = data_buf
        return img, frame_meta

    def _configure_camera(self, cam, prefix: str = ""):
        """配置在线采集参数（含 30fps 相关设置）。"""
        self.safe_set_enum(cam, "TriggerMode", 0, prefix=prefix)
        self.safe_set_enum(cam, "ExposureAuto", 0, prefix=prefix)
        self.safe_set_float(cam, "ExposureTime", self.exposure_time_us, prefix=prefix)
        self.safe_set_enum(cam, "GainAuto", 0, prefix=prefix)
        self.safe_set_float(cam, "Gain", 0.0, prefix=prefix)

        self.safe_set_bool(cam, "AcquisitionFrameRateEnable", True, prefix=prefix)
        self.safe_set_float(cam, "AcquisitionFrameRate", self.target_acquisition_fps, prefix=prefix)

    # ---------- 相机生命周期 ----------
    def open_cameras(self) -> List[MvCamera]:
        """枚举并打开相机，配置参数后开始取流。"""
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0:
            raise RuntimeError(f"enum devices fail! ret[0x{ret:x}]")
        if device_list.nDeviceNum < 2:
            raise RuntimeError(f"need at least 2 cameras, found {device_list.nDeviceNum}")

        cams: List[MvCamera] = []
        try:
            for i in range(device_list.nDeviceNum):
                cam = MvCamera()
                st_device_list = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents

                if cam.MV_CC_CreateHandle(st_device_list) != 0:
                    raise RuntimeError(f"create handle fail for camera index {i}")
                if cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
                    raise RuntimeError(f"open device fail for camera index {i}")

                if self.log_camera_nodes:
                    self.log_camera_status(cam, prefix=f"[camera {i}] before config")

                self._configure_camera(cam, prefix=f"[camera {i}]")

                if self.log_camera_nodes:
                    self.log_camera_status(cam, prefix=f"[camera {i}] after config")

                if cam.MV_CC_StartGrabbing() != 0:
                    raise RuntimeError(f"start grabbing fail for camera index {i}")

                cams.append(cam)
        except Exception:
            self.close_cameras(cams)
            raise

        return cams

    @staticmethod
    def close_cameras(cams: Sequence[MvCamera]):
        """停止取流并关闭相机。"""
        for cam in cams:
            try:
                cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                cam.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                cam.MV_CC_DestroyHandle()
            except Exception:
                pass

    def capture_and_save_frame_pair(
        self,
        frame_id: int,
        left_camera_index: int = 0,
        right_camera_index: int = 1,
        include_images: bool = True,
        save_images: bool = False,
        max_retries: int = 5,
        image_ext: str = "png",
        jpeg_quality: int = 90,
        png_compression: int = 1,
        cams: Optional[Sequence[MvCamera]] = None,
        async_saver: Optional[AsyncImageSaver] = None,
    ) -> FramePacket:
        """采集一对左右帧并生成 FramePacket。"""
        owns_cams = cams is None
        active_cams: Sequence[MvCamera] = self.open_cameras() if owns_cams else cams

        try:
            if left_camera_index == right_camera_index:
                raise ValueError("left_camera_index and right_camera_index must be different.")
            if left_camera_index >= len(active_cams) or right_camera_index >= len(active_cams):
                raise IndexError("camera index out of range.")

            left_img = None
            right_img = None
            left_meta = None
            right_meta = None
            for _ in range(max_retries):
                left_img, left_meta = self._get_image(active_cams[left_camera_index], convert_to_bgr=True)
                right_img, right_meta = self._get_image(active_cams[right_camera_index], convert_to_bgr=True)
                if left_img is not None and right_img is not None and left_meta is not None and right_meta is not None:
                    break

            if left_img is None or right_img is None or left_meta is None or right_meta is None:
                raise RuntimeError("failed to capture a valid left-right frame pair.")

            left_filename, right_filename, left_path, right_path = self._build_pair_paths(
                frame_id=frame_id,
                left_host_timestamp=left_meta.host_timestamp,
                right_host_timestamp=right_meta.host_timestamp,
                image_ext=image_ext,
                create_dirs=save_images,
            )

            if save_images:
                save_task = PairSaveTask(
                    frame_id=frame_id,
                    left_path=left_path,
                    right_path=right_path,
                    left_image=left_img,
                    right_image=right_img,
                    image_ext=self._normalize_image_ext(image_ext),
                    jpeg_quality=jpeg_quality,
                    png_compression=png_compression,
                )
                if async_saver is None:
                    self._write_pair_to_paths(save_task)
                    self._track_and_enforce_pair_retention(
                        frame_id=frame_id,
                        left_path=left_path,
                        right_path=right_path,
                    )
                else:
                    async_saver.submit(save_task)

            packet = FramePacket(
                frame_id=frame_id,
                left_host_timestamp=left_meta.host_timestamp,
                right_host_timestamp=right_meta.host_timestamp,
                left_dev_timestamp_raw=left_meta.dev_timestamp_raw,
                right_dev_timestamp_raw=right_meta.dev_timestamp_raw,
                left_frame_num=left_meta.frame_num,
                right_frame_num=right_meta.frame_num,
                left_filename=left_filename,
                right_filename=right_filename,
                left_path=left_path,
                right_path=right_path,
                left_image=left_img if include_images else None,
                right_image=right_img if include_images else None,
            )
            return packet
        finally:
            if owns_cams:
                self.close_cameras(active_cams)


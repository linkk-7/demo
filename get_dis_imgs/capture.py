import os
import sys
import threading
from collections import deque
from ctypes import *
from datetime import datetime
from queue import Queue
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from MvImport.MvCameraControl_class import *
from MvImport.CameraParams_header import *
from MvImport.CameraParams_const import *
from MvImport.MvErrorDefine_const import *
from MvImport.PixelType_header import *

from .models import FrameMeta, FramePacket, RetentionRecord

# 新增统一命名保存：L_时间戳_帧号.png / R_时间戳_帧号.png
# （保存在 save_folder_base/left、save_folder_base/right）

class CameraController:
    def __init__(self, save_folder_base: str, keep_last_n_pairs: Optional[int] = None):
        self.save_folder_base = save_folder_base
        self.g_bExit = False
        self.queues = []          # each camera: Queue of (frame_bgr, frame_meta)
        self.latest_frames = {}   # index -> (frame_bgr, frame_meta)
        self.save_every_ms = 0    # =0 means save every frame; >0 throttled by milliseconds
        self._last_save_ts = {}   # index -> datetime for throttle

        self.keep_last_n_pairs = keep_last_n_pairs if keep_last_n_pairs and keep_last_n_pairs > 0 else None
        self._retention_records: Deque[RetentionRecord] = deque()

    # ---------- FS utils ----------
    @staticmethod
    def _ensure_dir(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def _safe_remove(path: str):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    def _save_image(self, image, index, timestamp_token):
        folder_path = os.path.join(self.save_folder_base, f"Camera_{index}")
        self._ensure_dir(folder_path)
        file_path = os.path.join(folder_path, f"{timestamp_token}.jpg")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(file_path, image)
        return file_path

    @staticmethod
    def _format_pair_filename(prefix: str, host_timestamp: int, frame_id: int, image_ext: str = "png") -> str:
        return f"{prefix}_{host_timestamp}_{frame_id:06d}.{image_ext}"

    def _save_pair_images(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        frame_id: int,
        left_host_timestamp: int,
        right_host_timestamp: int,
        image_ext: str = "png",
    ) -> Tuple[str, str, str, str]:
        left_dir = os.path.join(self.save_folder_base, "left")
        right_dir = os.path.join(self.save_folder_base, "right")
        self._ensure_dir(left_dir)
        self._ensure_dir(right_dir)

        left_filename = self._format_pair_filename("L", left_host_timestamp, frame_id, image_ext)
        right_filename = self._format_pair_filename("R", right_host_timestamp, frame_id, image_ext)
        left_path = os.path.join(left_dir, left_filename)
        right_path = os.path.join(right_dir, right_filename)

        ok_left = cv2.imwrite(left_path, left_image)
        ok_right = cv2.imwrite(right_path, right_image)
        if not ok_left or not ok_right:
            if ok_left:
                self._safe_remove(left_path)
            if ok_right:
                self._safe_remove(right_path)
            raise IOError("Failed to save one or both paired images.")

        return left_filename, right_filename, left_path, right_path

###########################新增：最近 N 对保留策略（超限后按左右“一对”删除最旧）##############################
    def _track_and_enforce_pair_retention(self, frame_id: int, left_path: str, right_path: str):
        if self.keep_last_n_pairs is None:
            return

        self._retention_records.append(
            RetentionRecord(frame_id=frame_id, left_path=left_path, right_path=right_path)
        )
        while len(self._retention_records) > self.keep_last_n_pairs:
            old_record = self._retention_records.popleft()
            self._safe_remove(old_record.left_path)
            self._safe_remove(old_record.right_path)

    # ---------- pixel conversion ----------
    @staticmethod
    def _to_bgr(cam, stOutFrame, data_buf):
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
        except Exception as e:
            print("ConvertPixelType failed, use gray fallback:", e)

        gray = arr1d.reshape(h, w)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ---------- grab one image ----------
    @staticmethod
    def _extract_frame_meta(st_frame_info) -> FrameMeta:
        dev_timestamp = (int(st_frame_info.nDevTimeStampHigh) << 32) | int(st_frame_info.nDevTimeStampLow)
        return FrameMeta(
            dev_timestamp_raw=dev_timestamp,                   # 同时保留下设备时间戳（从第一）
            host_timestamp=int(st_frame_info.nHostTimeStamp),  # 使用主机时间戳进行命名
            frame_num=int(st_frame_info.nFrameNum),
        )

################# 从sdk取图 #######################
    def _get_image(self, cam):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 2000)
        if ret != 0 or not stOutFrame.pBufAddr:
            return None, None

        size = stOutFrame.stFrameInfo.nFrameLen
        data_buf = (c_ubyte * size)()
        cdll.msvcrt.memcpy(byref(data_buf), stOutFrame.pBufAddr, size)
        cam.MV_CC_FreeImageBuffer(stOutFrame)

        img = self._to_bgr(cam, stOutFrame, data_buf)
        frame_meta = self._extract_frame_meta(stOutFrame.stFrameInfo)
        return img, frame_meta

    def _configure_camera(self, cam):
        cam.MV_CC_SetEnumValue("TriggerMode", 0)      # Off
        cam.MV_CC_SetEnumValue("ExposureAuto", 0)     # Off
        cam.MV_CC_SetFloatValue("ExposureTime", 8000)
        cam.MV_CC_SetEnumValue("GainAuto", 0)
        cam.MV_CC_SetFloatValue("Gain", 0.0)

    def open_cameras(self) -> List[MvCamera]:
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

                self._configure_camera(cam)

                if cam.MV_CC_StartGrabbing() != 0:
                    raise RuntimeError(f"start grabbing fail for camera index {i}")

                cams.append(cam)
        except Exception:
            self.close_cameras(cams)
            raise

        return cams

    @staticmethod
    def close_cameras(cams: Sequence[MvCamera]):
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

############################新增函数：保存FramePacket信息##############################
    def capture_and_save_frame_pair(
        self,
        frame_id: int,
        left_camera_index: int = 0,
        right_camera_index: int = 1,
        include_images: bool = False,
        max_retries: int = 5,
        image_ext: str = "png",
        cams: Optional[Sequence[MvCamera]] = None,
    ) -> FramePacket:
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
                left_img, left_meta = self._get_image(active_cams[left_camera_index])
                right_img, right_meta = self._get_image(active_cams[right_camera_index])
                if left_img is not None and right_img is not None and left_meta is not None and right_meta is not None:
                    break

            if left_img is None or right_img is None or left_meta is None or right_meta is None:
                raise RuntimeError("failed to capture a valid left-right frame pair.")

            left_filename, right_filename, left_path, right_path = self._save_pair_images(
                left_img,
                right_img,
                frame_id,
                left_meta.host_timestamp,
                right_meta.host_timestamp,
                image_ext=image_ext,
            )

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
            self._track_and_enforce_pair_retention(
                frame_id=packet.frame_id,
                left_path=packet.left_path,
                right_path=packet.right_path,
            )
            return packet
        finally:
            if owns_cams:
                self.close_cameras(active_cams)

    # ---------- worker: grab only, push queue ----------
    def _grab_worker(self, cam, index, q: Queue):
        while not self.g_bExit:
            img, frame_meta = self._get_image(cam)
            if img is None or frame_meta is None:
                continue
            if not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    pass
            q.put((img, frame_meta))

    # ---------- main: enumerate, grab, show, save ----------
    def get_dis_imgs(self):
        self.g_bExit = False

        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            sys.exit(1)
        if deviceList.nDeviceNum == 0:
            print("find no device!")
            sys.exit(1)
        print("Find %d devices!" % deviceList.nDeviceNum)

        cams = []
        for i in range(deviceList.nDeviceNum):
            cam = MvCamera()
            stDeviceList = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents

            if cam.MV_CC_CreateHandle(stDeviceList) != 0:
                print("create handle fail")
                sys.exit(1)
            if cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
                print("open device fail")
                sys.exit(1)

            self._configure_camera(cam)

            if cam.MV_CC_StartGrabbing() != 0:
                print("start grabbing fail")
                sys.exit(1)

            cams.append(cam)

        print(f"connected {len(cams)} cameras")

        self.queues = []
        threads = []
        for i, cam in enumerate(cams):
            q = Queue(maxsize=1)
            self.queues.append(q)
            t = threading.Thread(target=self._grab_worker, args=(cam, i, q), daemon=True)
            t.start()
            threads.append(t)
            self._last_save_ts[i] = None

        for i in range(len(cams)):
            cv2.namedWindow(f"Camera {i}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"Camera {i}", 800, 800)

        print("press 'q' to exit.")

        while True:
            updated = False
            for i, q in enumerate(self.queues):
                if not q.empty():
                    frame, frame_meta = q.get()
                    self.latest_frames[i] = (frame, frame_meta)
                    updated = True

                    cv2.imshow(f"Camera {i}", frame)

                    do_save = True
                    if self.save_every_ms > 0:
                        last = self._last_save_ts[i]
                        now = datetime.now()
                        if last and (now - last).total_seconds() * 1000.0 < self.save_every_ms:
                            do_save = False
                        else:
                            self._last_save_ts[i] = now
                    if do_save:
                        try:
                            self._save_image(frame, i, frame_meta.host_timestamp)
                        except Exception as e:
                            print(f"[WARN] save Camera {i} failed: {e}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.g_bExit = True
                break

            if not updated:
                cv2.waitKey(1)

        for t in threads:
            t.join()
        for cam in cams:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()

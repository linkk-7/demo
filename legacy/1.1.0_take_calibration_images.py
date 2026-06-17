# -*- coding: utf-8 -*-
import sys
import os
import threading
from ctypes import *
from datetime import datetime
from queue import Queue

import numpy as np
import cv2
import cv2.aruco as aruco

# === SDK 路径 ===
try:
    sys.path.append(r"E:\研究生\西安碑林\Superpoint双目位移监测代码_原始版0606\MvImport")
    from MvImport.MvCameraControl_class import *
    from MvImport.MvErrorDefine_const import *
except Exception as e:
    print("MvCameraControl_class.py 文件不存在或导入失败：", e)
    raise

class CameraController:
    def __init__(self, save_folder_base):
        self.save_folder_base = save_folder_base  # 图像保存的基础路径
        self.g_bExit = False

        # 当前帧缓存（显示用）：index -> (frame_bgr, timestamp_str)
        self.frames = {}
        # 采集线程队列（每相机一个）
        self.frame_queues = []

        # 保存逻辑
        self.save_count = 0
        self.left_saved_images = []   # index==1  左相机
        self.right_saved_images = []  # index==0  右相机
        self.confirm_mode = False

        # ArUco/ChArUco 资源  4×4 的 50 个标记字典
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        # squaresX=8, squaresY=7, squareLength=15, markerLength=11（单位自定，标定时一致即可）
        self.board = aruco.CharucoBoard((8, 7), 100, 75, self.dictionary)
        # 创建ChArUco标定板：8列7行棋盘格，大方格边长15，ArUco标记边长11，4×4 的 50 个标记字典

    # ---------------------- 文件与显示辅助 ----------------------
    @staticmethod
    def _ensure_dir(path):  # 确保路径存在
        if not os.path.exists(path):
            os.makedirs(path)

    def save_image(self, image, index, timestamp):
        folder_path = os.path.join(self.save_folder_base, f"Camera_{index}")
        self._ensure_dir(folder_path)
        file_path = os.path.join(folder_path, f"{timestamp}.bmp")
        cv2.imwrite(file_path, image)
        print(f"已保存图像到: {file_path}")
        if index == 1:
            self.right_saved_images.append(file_path)
        elif index == 0:
            self.left_saved_images.append(file_path)
        return file_path

    # ---------------------- 解码：由帧头自适应 ----------------------
    @staticmethod
    def _convert_to_bgr(cam, stOutFrame, data_buf):
        w = stOutFrame.stFrameInfo.nWidth
        h = stOutFrame.stFrameInfo.nHeight
        size = stOutFrame.stFrameInfo.nFrameLen
        pix = stOutFrame.stFrameInfo.enPixelType
        # 获取图像的宽度、高度、数据长度和像素格式

        arr1d = np.frombuffer(data_buf, dtype=np.uint8, count=size)

        # Mono8
        if pix == PixelType_Gvsp_Mono8:
            gray = arr1d.reshape(h, w)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 常见 Bayer8
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

        # 兜底：SDK 转 BGR8
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
            print("ConvertPixelType 失败，按灰度兜底：", e)

        gray = arr1d.reshape(h, w)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ---------------------- 取帧（无 GUI） ----------------------
    def get_image(self, cam):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 2000)
        if ret != 0 or not stOutFrame.pBufAddr:
            return None, None

        size = stOutFrame.stFrameInfo.nFrameLen
        data_buf = (c_ubyte * size)()
        cdll.msvcrt.memcpy(byref(data_buf), stOutFrame.pBufAddr, size)
        cam.MV_CC_FreeImageBuffer(stOutFrame)

        img = self._convert_to_bgr(cam, stOutFrame, data_buf)
        ts = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-1]
        return img, ts

    def grab_worker(self, cam, index, q: Queue):
        while not self.g_bExit:
            img, ts = self.get_image(cam)
            if img is None:
                continue
            if not q.empty():
                try:
                    q.get_nowait()
                except:
                    pass
            q.put((img, ts))

    # ---------------------- 评估与保存 ----------------------
    def calculate_reprojection_error(self, images):
        """
        对单相机计算 ChArUco 标定的 RMS 重投影误差。
        返回 (rms_error, camera_matrix) 或 (None, None)
        """
        all_corners = []
        all_ids = []
        imsize = None

        # === 新增：配置检测参数 ===
        aruco_params = aruco.DetectorParameters()
        # 允许检测非常小的标记 (例如 0.5% 甚至更小，根据实际距离调整)
        aruco_params.minMarkerPerimeterRate = 0.02
        # 如果光照不均匀，可以尝试调整二值化窗口（可选）
        # aruco_params.adaptiveThreshWinSizeMin = 3
        # aruco_params.adaptiveThreshWinSizeMax = 23

        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imsize = gray.shape[::-1]

            corners, ids, _ = aruco.detectMarkers(gray, self.dictionary, parameters=aruco_params)  # 检测 ArUco 方块
            # print(f"   [Debug] 图片: {os.path.basename(img_path)}")
            # print(f"   [Debug] 原始标记(Ids)数量: {0 if ids is None else len(ids)}")
            if ids is None or len(ids) == 0:
                continue

            ok, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board)
            # print(f"   [Debug] 插值后棋盘角点(Corners)数量: {0 if charuco_corners is None else len(charuco_corners)}")
            if ok and charuco_corners is not None and charuco_ids is not None:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

        if len(all_corners) == 0 or imsize is None:
            return None, None

        try:
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                all_corners, all_ids, self.board, imsize, None, None)
            return rms, camera_matrix
        except Exception as e:
            print("标定计算错误：", e)
            return None, None
        # 使用ChArUco角点进行相机标定，返回重投影误差和相机内参矩阵

    def evaluate_current_frames(self):
        """评估当前帧的标定质量并打印 RMS 误差"""
        # 将当前帧临时落盘用于评估
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-1]
        temp_folder = os.path.join(self.save_folder_base, "temp_eval")
        self._ensure_dir(temp_folder)

        temp_files = []
        for index, (frame, _) in self.frames.items():
            if frame is not None:
                temp_path = os.path.join(temp_folder, f"temp_{index}_{timestamp}.bmp")
                cv2.imwrite(temp_path, frame)
                temp_files.append((index, temp_path))

        # 组装各相机的评估集合（历史+当前）
        left_files = list(self.left_saved_images)
        right_files = list(self.right_saved_images)
        for idx, p in temp_files:
            if idx == 1:
                right_files.append(p)
            elif idx == 0:
                left_files.append(p)

        print("计算重投影误差中...")
        left_error = right_error = None

        if len(left_files) > 0:
            left_error, _ = self.calculate_reprojection_error([p if isinstance(p, str) else p[1]
                                                               for p in left_files])
            print(f"左相机重投影误差: {left_error}")
        if len(right_files) > 0:
            right_error, _ = self.calculate_reprojection_error([p if isinstance(p, str) else p[1]
                                                                for p in right_files])
            print(f"右相机重投影误差: {right_error}")

        msg = (f"左相机误差: {left_error:.6f}" if left_error is not None else "左相机误差: 无法计算") + \
              (f", 右相机误差: {right_error:.6f}" if right_error is not None else ", 右相机误差: 无法计算")
        print("=== 重投影误差 ===")
        print(msg)
        print("评估完毕：再次按 's' 保存当前帧，或按 'x' 取消。")

        # 清理临时文件
        for _, p in temp_files:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass

    def save_current_frames(self):
        """保存当前各相机图像"""
        if not self.frames:
            print("未找到任何相机图像")
            return

        saved = 0
        for index, (frame, ts) in self.frames.items():
            if frame is None:
                continue
            self.save_image(frame, index, ts)
            saved += 1

        if saved > 0:
            self.save_count += 1
            print(f"已保存 {saved} 个相机图像；累计保存 {self.save_count} 组。")
        else:
            print("没有保存任何图像，所有帧都为空")

    # ---------------------- 主流程：取流 + 显示 + 交互 ----------------------
    def get_stereo_images(self):
        self.g_bExit = False

        print("=== 双目相机标定图像获取 ===")
        print("按 's' 键评估当前图像质量；评估后再次按 's' 保存；按 'x' 取消；按 'q' 退出。")

        self.get_dis_imgs()

    def get_dis_imgs(self):
        # === 新增：配置检测参数 ===
        aruco_params = aruco.DetectorParameters()
        # 允许检测非常小的标记 (例如 0.5% 甚至更小，根据实际距离调整)
        aruco_params.minMarkerPerimeterRate = 0.02
        # 如果光照不均匀，可以尝试调整二值化窗口（可选）
        # aruco_params.adaptiveThreshWinSizeMin = 3
        # aruco_params.adaptiveThreshWinSizeMax = 23

        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret); sys.exit(1)
        if deviceList.nDeviceNum == 0:
            print("find no device!"); sys.exit(1)
        print("Find %d devices!" % deviceList.nDeviceNum)

        cams = []
        for i in range(deviceList.nDeviceNum):
            cam = MvCamera()
            stDeviceList = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents

            if cam.MV_CC_CreateHandle(stDeviceList) != 0:
                print("create handle fail"); sys.exit(1)
            if cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
                print("open device fail"); sys.exit(1)

            # 保险设置（可按需调整）
            cam.MV_CC_SetEnumValue("TriggerMode", 0)      # Off
            cam.MV_CC_SetEnumValue("ExposureAuto", 0)     # Off
            cam.MV_CC_SetFloatValue("ExposureTime", 11000) # 8ms 起步
            cam.MV_CC_SetEnumValue("GainAuto", 0)         # Off
            cam.MV_CC_SetFloatValue("Gain", 8000)
            # 如需先验证通道，可强制 Mono8，更稳：
            # cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)

            if cam.MV_CC_StartGrabbing() != 0:
                print("start grabbing fail"); sys.exit(1)

            cams.append(cam)

        print(f"成功连接 {len(cams)} 个相机")

        # 启动采集线程 & 队列
        threads = []
        self.frame_queues = []
        for i, cam in enumerate(cams):
            q = Queue(maxsize=1)
            self.frame_queues.append(q)
            t = threading.Thread(target=self.grab_worker, args=(cam, i, q), daemon=True)
            t.start()
            threads.append(t)

        # 主线程创建窗口（只创建一次）
        for i in range(len(cams)):
            cv2.namedWindow(f"Camera {i}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"Camera {i}", 600, 600)

        # 显示循环 + 键盘交互（只在主线程）
        while True:
            # 刷新图像
            for i, q in enumerate(self.frame_queues):
                if not q.empty():
                    frame, ts = q.get()
                    self.frames[i] = (frame, ts)

                    # 叠加可视化（ArUco/ChArUco）
                    display = frame.copy()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = aruco.detectMarkers(gray, self.dictionary, parameters=aruco_params)
                    if ids is not None and len(ids) > 0:
                        aruco.drawDetectedMarkers(display, corners, ids)
                        ok, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                            corners, ids, gray, self.board)
                        if ok and charuco_corners is not None and charuco_ids is not None:
                            aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
                            cv2.putText(display, f"Detected {len(charuco_corners)} corners",
                                        (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,255,0), 5, cv2.LINE_AA)

                    saved_count = len(self.right_saved_images) if i == 0 else len(self.left_saved_images)
                    cv2.putText(display, f"Camera {i} | Saved: {saved_count}",
                                (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255,0,0), 5, cv2.LINE_AA)

                    if self.confirm_mode:
                        cv2.putText(display, "Press 's' to SAVE, 'x' to CANCEL",
                                    (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,0,255), 5, cv2.LINE_AA)

                    cv2.imshow(f"Camera {i}", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.g_bExit = True
                break
            elif key == ord('s') and not self.confirm_mode:
                # 进入确认模式并评估
                self.confirm_mode = True
                self.evaluate_current_frames()
            elif key == ord('s') and self.confirm_mode:
                # 确认保存
                self.save_current_frames()
                self.confirm_mode = False
            elif key == ord('x') and self.confirm_mode:
                print("取消保存")
                self.confirm_mode = False

        # 收尾
        for t in threads:
            t.join()

        for cam in cams:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()

        cv2.destroyAllWindows()

# 入口
if __name__ == "__main__":
    # controller = CameraController(r"new_data1\cab")
    save_path = os.path.join("new_data5", "cab")
    controller = CameraController(save_path)
    controller.get_stereo_images()

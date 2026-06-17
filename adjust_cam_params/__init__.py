# 测试，调焦距（修正版：主线程显示 + 自适应像素格式）
# 主线程          ←→      多个采集线程（每个相机一个）
#    │                        │
#    │                  只负责疯狂取帧 → 放进 Queue（长度最大1）
#    │                        │
#    ├──────→ 主线程每帧从每个 Queue 取出最新图像 → cv2.imshow 显示
#    │                        │
#    └──────→ 按 's' 时，把当前正在显示的那一帧保存到硬盘


import sys
import threading
from ctypes import *
import numpy as np
import cv2
import os
from datetime import datetime
from queue import Queue

# 你的 SDK 路径
sys.path.append(r"E:\研究生\西安碑林\Superpoint双目位移监测代码_原始版0606\MvImport")
from MvCameraControl_class import *
from MvErrorDefine_const import *

g_bExit = False
frame_queues = []  # 每个相机一个队列

# ---------------------- 工具函数 ----------------------
# 确保保存路径存在
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 保存图像
def save_image(image, index, timestamp):
    folder_path = rf'D:\碑林资料\Superpoint双目位移监测代码_原始版0606\all_test\hktest\Camera_{index}'
    ensure_dir(folder_path)
    # 把第 index 个相机的图像保存为：Camera_0/0_142530.123.png 这样的格式。
    file_path = os.path.join(folder_path, f"{index}_{timestamp}.png")
    cv2.imwrite(file_path, image)

def _convert_to_bgr(cam, stOutFrame, data_buf):
    """
    根据帧头信息把原始缓冲转换为 BGR (HxWx3, uint8)。
    data_buf: ctypes (c_ubyte * n)() 缓冲已拷贝

    不能直接 imshow，必须转成 OpenCV 能识别的 BGR8（uint8 三通道）
    """
    w = stOutFrame.stFrameInfo.nWidth
    h = stOutFrame.stFrameInfo.nHeight
    size = stOutFrame.stFrameInfo.nFrameLen
    pix = stOutFrame.stFrameInfo.enPixelType

    arr1d = np.frombuffer(data_buf, dtype=np.uint8, count=size)

    # 常见 8bit 单通道格式
    if pix == PixelType_Gvsp_Mono8:
        gray = arr1d.reshape(h, w)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 常见 Bayer 8bit
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

    # 兜底：用 SDK 转 BGR8
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
        print("ConvertPixelType fallback, err:", e)

    # 实在不识别：按灰度显示兜底
    gray = arr1d.reshape(h, w)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ---------------------- 取帧函数（不做显示） ----------------------
def get_image(cam):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 2000)  # 超时稍长一点  阻塞最多 2 秒取一帧
    if ret != 0 or not stOutFrame.pBufAddr:
        # 打印一次即可，避免刷屏
        # print("no data[0x%x]" % ret)
        return None, None

    # 复制底层缓冲到用户态缓冲，然后立即释放底层帧
    size = stOutFrame.stFrameInfo.nFrameLen
    data_buf = (c_ubyte * size)()
    cdll.msvcrt.memcpy(byref(data_buf), stOutFrame.pBufAddr, size)
    cam.MV_CC_FreeImageBuffer(stOutFrame)  # 必须立即释放

    img = _convert_to_bgr(cam, stOutFrame, data_buf)

    ts = datetime.now().strftime("%H%M%S.%f")[:-1]
    # 调试打印：确认在稳定取帧
    # print(f"get one frame: W[{stOutFrame.stFrameInfo.nWidth}], H[{stOutFrame.stFrameInfo.nHeight}], "
    #       f"Pix[{stOutFrame.stFrameInfo.enPixelType}], FrameNum[{stOutFrame.stFrameInfo.nFrameNum}]")
    return img, ts

# ---------------------- 采集线程：只入队 ----------------------
def capture(cam, index, q: Queue):
    global g_bExit
    while not g_bExit:
        img, ts = get_image(cam)
        if img is None:
            continue
        # 丢弃旧帧，保持实时
        if not q.empty():
            try:
                q.get_nowait()  # 扔掉上一帧
            except:
                pass
        q.put(img)  # 放入最新帧

# ---------------------- 主流程 ----------------------
# 整个程序的入口
def adjust_cam():
    global g_bExit

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
        # 创建句柄 → 打开设备 → 设置参数 → 开始采集
        stDeviceList = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret); sys.exit(1)

        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret); sys.exit(1)

        # —— 保险设置：防止被 MVS 改过导致无帧或过曝 —— #
        cam.MV_CC_SetEnumValue("TriggerMode", 0)      # Off      # 连续模式（不是外触发）
        cam.MV_CC_SetEnumValue("ExposureAuto", 0)     # Off      # 关闭自动曝光
        cam.MV_CC_SetFloatValue("ExposureTime", 10000) # 8ms 起步  # 手动曝光 8000μs = 8ms
        cam.MV_CC_SetEnumValue("GainAuto", 1)         # Off      # 关闭自动增益
        cam.MV_CC_SetFloatValue("Gain", 5000)
        # 若只为验证通路，先强制 Mono8 更稳：确认通了再注释掉
        # cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("start grabbing fail! ret[0x%x]" % ret); sys.exit(1)

        cams.append(cam)

    print(len(cams))

    # 启动每个相机的队列和采集线程
    threads = []
    for i, cam in enumerate(cams):
        q = Queue(maxsize=1)
        frame_queues.append(q)
        t = threading.Thread(target=capture, args=(cam, i, q), daemon=True)
        t.start()
        threads.append(t)

    # 只在主线程创建窗口（一次）
    for i in range(len(cams)):
        cv2.namedWindow(f"Camera {i}", cv2.WINDOW_NORMAL)

    print("Press 'q' to quit, 's' to save current frames.")
    while True:
        # 刷新显示
        for i, q in enumerate(frame_queues):
            if not q.empty():
                frame = q.get()
                cv2.imshow(f"Camera {i}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            g_bExit = True
            break
        elif key == ord('s'):
            ts = datetime.now().strftime("%H%M%S.%f")[:-1]
            for i, q in enumerate(frame_queues):
                if not q.empty():
                    img = q.queue[-1]
                    save_image(img, i, ts)

    # 收尾
    for t in threads:
        t.join()

    for cam in cams:
        ret = cam.MV_CC_StopGrabbing()
        if ret != 0:
            print("stop grabbing fail! ret[0x%x]" % ret)
        ret = cam.MV_CC_CloseDevice()
        if ret != 0:
            print("close device fail! ret[0x%x]" % ret)
        ret = cam.MV_CC_DestroyHandle()
        if ret != 0:
            print("destroy handle fail! ret[0x%x]" % ret)

    cv2.destroyAllWindows()

# 如果你希望像原来一样 from xxx import adjust_cam 直接调用，这里不加 __main__ 判断

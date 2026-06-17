import cv2
import numpy as np
import threading
import queue
import time
import os
class StereoCapture:
    def __init__(self, left_device=2, right_device=1):
        self.frameQueue1 = queue.Queue(maxsize=1)
        self.frameQueue2 = queue.Queue(maxsize=1)
        self.cap1 = cv2.VideoCapture(left_device)
        self.cap2 = cv2.VideoCapture(right_device)
        # 可以添加更多初始化代码

    def capture_frames(self, cap, frameQueue):

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if not frameQueue.empty():
                try:
                    frameQueue.get_nowait()  # 如果队列非空，则丢弃旧帧
                except queue.Empty:
                    pass
            frameQueue.put((ret,frame))

    def start(self, left_file, right_file):
        if not os.path.exists(left_file):
            os.makedirs(left_file)
        if not os.path.exists(right_file):
            os.makedirs(right_file)
        cap1 = self.cap1
        cap2 = self.cap2

        # 设置摄像头参数（略）
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)  # 设置图像宽度
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)  # 设置图像高度
        cap1.set(cv2.CAP_PROP_FPS, 60)  # 设置帧率
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)  # 设置图像宽度
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)  # 设置图像高度
        cap2.set(cv2.CAP_PROP_FPS, 60)  # 设置帧率

        # 启动两个线程分别读取帧
        threading.Thread(target=self.capture_frames, args=(cap1, self.frameQueue1), daemon=True).start()
        threading.Thread(target=self.capture_frames, args=(cap2, self.frameQueue2), daemon=True).start()

        i = 1
        last_save_time = time.time()
        while True:
            if not self.frameQueue1.empty() and not self.frameQueue2.empty():
                ret1,frame1 = self.frameQueue1.get()
                ret2,frame2 = self.frameQueue2.get()

                # 处理帧
                if not ret1 or not ret2:
                    print('摄像头未正确打开！')
                    break
                left_img = frame1
                right_img = frame2
                combined_frame = np.hstack((frame1, frame2))
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.imshow('frame', combined_frame)
                current_time = time.time()
                if cv2.waitKey(1) == ord('s'):
                    left_filename = left_file + '\\' + str(i) + '.bmp'
                    right_filename = right_file + '\\' + str(i) + '.bmp'
                    cv2.imwrite(right_filename, right_img)
                    cv2.imwrite(left_filename, left_img)
                    print(f'保存第{i}张棋盘格图像')
                    i += 1
                    last_save_time = current_time
                elif cv2.waitKey(1) == ord('q'):
                    print('程序已正常退出')
                    break

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
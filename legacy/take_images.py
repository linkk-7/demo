# -*- coding: utf-8 -*-
import os
import argparse
import signal
from get_dis_imgs_jetson import CameraController


controller = None

def handle_sigint(signum, frame):
    global controller
    print("\n[INFO] 收到退出信号，准备停止采集...")
    if controller is not None:
        controller.g_bExit = True


def main():
    global controller

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="图像保存根目录")
    parser.add_argument("--save_every_ms", type=int, default=100, help="保存间隔(ms)，0表示每帧都保存")
    parser.add_argument("--display", type=int, default=0, help="1=显示窗口，0=不显示")
    parser.add_argument("--exposure_time", type=float, default=8000.0, help="曝光时间(us)")
    parser.add_argument("--gain", type=float, default=0.0, help="增益")
    parser.add_argument("--mono", type=int, default=0, help="1=强制Mono8")
    parser.add_argument("--jpg_quality", type=int, default=95, help="jpg质量")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    controller = CameraController(
        save_folder_base=args.save_dir,
        save_every_ms=args.save_every_ms,
        display=bool(args.display),
        exposure_time=args.exposure_time,
        gain=args.gain,
        pixel_format_mono=bool(args.mono),
        jpg_quality=args.jpg_quality,
    )
    controller.run()


if __name__ == "__main__":
    main()
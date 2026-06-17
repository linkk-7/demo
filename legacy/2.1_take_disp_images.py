######################## 采集入口脚本（双模式） #################################

import os

from get_dis_imgs import continuous_capture_disk, continuous_capture_online

# 模式切换：
# - online：不落盘，FramePacket 直接携带左右图像数组，适合正式在线处理
# - disk：保存 JPG，并循环覆盖最近 N 对，适合调试和离线留痕
MODE = "online"  # 可改为 "disk"

# 采集输出根目录（disk 模式会在该目录下写入 left/right）
img_save_path = os.path.join("input", "cab")

if MODE == "online":
    continuous_capture_online(
        save_folder_base=img_save_path,
    )
elif MODE == "disk":
    continuous_capture_disk(
        save_folder_base=img_save_path,
        keep_last_n_pairs=500,
        jpeg_quality=90,
        use_async_saver=True,   # 采集与存图解耦
    )
else:
    raise ValueError("MODE must be 'online' or 'disk'")

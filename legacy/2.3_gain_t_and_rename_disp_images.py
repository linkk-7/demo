"""
Legacy / Offline 兼容脚本：时间数组导出 + 顺序重命名

说明：
1. 本脚本不再属于当前新主流程。
2. 当前主流程已经在采集阶段生成标准命名与 FramePacket 元信息，
   不再依赖“从文件名再解析时间”或“重命名为 1l/1r”。
3. 本脚本仅建议用于历史旧数据目录的离线兼容处理。
"""

from get_t import get_t, rename


# legacy 示例路径（按需改成你的旧数据目录）
# left_path = r"E:\shuangmu shuju\20250930\1\left_rec"
# right_path = r"E:\shuangmu shuju\20250930\1\right_rec"
left_path = r"new_data5\cab1\left_rec"
right_path = r"new_data5\cab1\right_rec"


if __name__ == "__main__":
    print("[LEGACY] 2.3_gain_t_and_rename_disp_images.py 仅用于历史旧数据离线兼容处理。")
    print("[LEGACY] 当前新主流程已不依赖本脚本。")

    # 旧式：从文件名解析时间并导出 t_abs / t_rel 等
    get_t(left_path)
    get_t(right_path)

    # 旧式：顺序重命名为 1l.jpg / 1r.jpg ...
    rename(left_path, "l")
    rename(right_path, "r")

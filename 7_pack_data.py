import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any


def load_cal3d_result(npy_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npy_path, allow_pickle=True).item()

    required_keys = ["X", "Y", "Z"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"结果文件中缺少键: {key}")

    X = np.asarray(data["X"]).reshape(-1)
    Y = np.asarray(data["Y"]).reshape(-1)
    Z = np.asarray(data["Z"]).reshape(-1)

    # 优先用绝对时间上传
    if "T_abs" in data:
        T = np.asarray(data["T_abs"]).reshape(-1)
    elif "T" in data:
        T = np.asarray(data["T"]).reshape(-1)
    else:
        raise KeyError("结果文件中缺少时间键: T_abs 或 T")

    if len(X) == 0:
        raise ValueError("结果文件为空，没有可用数据。")

    if not (len(X) == len(Y) == len(Z) == len(T)):
        raise ValueError("X/Y/Z/T 长度不一致，无法打包。")

    return {"X": X, "Y": Y, "Z": Z, "T": T}


def build_point_record(
    left_id: int,
    right_id: int,
    channel: int,
    px_dx: float,
    px_dy: float,
    px_dz: float,
    disp_x: float,
    disp_y: float,
    disp_z: float,
    match_conf: float,
    track_conf: float,
) -> List[float]:
    """
    你双目版单点记录结构:
    [左点ID, 右点ID, 通道, x像素移动, y像素移动, z像素移动,
     x位移, y位移, z位移, 匹配置信度, 跟踪置信度]
    """
    return [
        int(left_id),
        int(right_id),
        int(channel),
        float(px_dx),
        float(px_dy),
        float(px_dz),
        float(disp_x),
        float(disp_y),
        float(disp_z),
        float(match_conf),
        float(track_conf),
    ]


def build_tracking_result_frames(
    result: Dict[str, np.ndarray],
    left_id: int,
    right_id: int,
    channel: int = 1,
    px_dx: float = 0.0,
    px_dy: float = 0.0,
    px_dz: float = 0.0,
    match_conf: float = 1.0,
    track_conf: float = 1.0,
    status: int = 1,
) -> List[Dict[str, Any]]:
    """
    最外层是 list，每个元素是一帧 {"t","s","p"}。
    """
    X = result["X"]
    Y = result["Y"]
    Z = result["Z"]
    T = result["T"]

    frames: List[Dict[str, Any]] = []

    for i in range(len(T)):
        point = build_point_record(
            left_id=left_id,
            right_id=right_id,
            channel=channel,
            px_dx=px_dx,
            px_dy=px_dy,
            px_dz=px_dz,
            disp_x=float(X[i]),
            disp_y=float(Y[i]),
            disp_z=float(Z[i]),
            match_conf=match_conf,
            track_conf=track_conf,
        )

        frame = {
            "t": int(T[i]),   # 写 int
            "s": int(status), # s 由外部传入
            "p": [point]
        }
        frames.append(frame)

    return frames


def save_tracking_result_json(
    frames: List[Dict[str, Any]],
    save_dir: str,
    filename: str = None
) -> str:
    """
    将整段 frames 一次性保存为一个 tracking_result_xxx.json
    """
    os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"displacement_result_{now_str}.json"

    save_path = os.path.join(save_dir, filename)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(frames, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存 displacement_result 到: {save_path}")
    print(f"✅ 共写入帧数: {len(frames)}")
    return save_path


if __name__ == "__main__":
    # ===== 1. 改成实际路径 =====
    npy_path = r"E:\研究生\西安碑林\Superpoint双目位移监测代码_原始版0606\new_data5\cab3\视觉位移计算结果-cal3d-ALIKED.npy"

    # ===== 2. 改成想保存 json 的目录 =====
    save_dir = "output"

    # ===== 3. 改成双目点的 ID =====
    left_id = 354
    right_id = 363

    # ===== 4. 读取结果 =====
    result = load_cal3d_result(npy_path)

    # ===== 5. 生成整段 tracking_result =====
    frames = build_tracking_result_frames(
        result=result,
        left_id=left_id,
        right_id=right_id,
        channel=1,        # 先保持一致
        px_dx=0.0,        # 当前没有真实值，先占位
        px_dy=0.0,
        px_dz=0.0,
        match_conf=1.0,   # 当前没有真实值，先占位
        track_conf=1.0,
        status=1          # s 的生成方式：外部传什么写什么
    )

    # ===== 6. 保存为一个 json =====
    save_tracking_result_json(frames, save_dir)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import correlate

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def moving_average(y, win=9):
    if win <= 1 or len(y) < win:
        return y.copy()
    kernel = np.ones(win) / win
    y_pad = np.pad(y, (win // 2, win // 2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def load_visual_npy(npy_path, direction="X"):
    data = np.load(npy_path, allow_pickle=True).item()
    t = np.asarray(data["T"], dtype=float) / 1000.0   # ms -> s
    y = np.asarray(data[direction], dtype=float)
    return t, y


def load_laser_xlsx(xlsx_path, sheet_name="Sheet1", time_col="time[s]", disp_col="AI1-1-01[mm]"):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[disp_col], errors="coerce").to_numpy()
    valid = np.isfinite(t) & np.isfinite(y)
    return t[valid], y[valid]


def remove_offset(y):
    return y - y[0]


def detect_active_segment(t, y, smooth_win=9, threshold_ratio=0.08, min_points=10):
    """
    自动检测运动有效段：
    用相对初始基线偏移量来找“开始运动”和“结束运动”
    """
    y0 = remove_offset(y)
    ys = moving_average(y0, smooth_win)

    amp = np.max(ys) - np.min(ys)
    if amp < 1e-9:
        return 0, len(y) - 1

    threshold = threshold_ratio * amp
    active = np.abs(ys) >= threshold

    idx = np.where(active)[0]
    if len(idx) == 0:
        return 0, len(y) - 1

    # 找最长连续段
    groups = []
    start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
        else:
            groups.append((start, prev))
            start = k
            prev = k
    groups.append((start, prev))

    # 取最长段
    lengths = [b - a + 1 for a, b in groups]
    best = np.argmax(lengths)
    i0, i1 = groups[best]

    # 扩一点边界
    pad = max(3, (i1 - i0 + 1) // 20)
    i0 = max(0, i0 - pad)
    i1 = min(len(y) - 1, i1 + pad)

    if i1 - i0 + 1 < min_points:
        return 0, len(y) - 1

    return i0, i1


def refine_shift_by_local_xcorr(t_ref, y_ref, t_tar, y_tar):
    """
    在局部活动段上进一步用互相关细化时间偏移
    返回：tar 需要整体加上的 shift（秒）
    """
    f_tar = interp1d(t_tar, y_tar, kind="linear", bounds_error=False, fill_value="extrapolate")
    y_tar_on_ref = f_tar(t_ref)

    a = moving_average(y_ref, 7) - np.mean(y_ref)
    b = moving_average(y_tar_on_ref, 7) - np.mean(y_tar_on_ref)

    corr = correlate(a, b, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    best_lag = lags[np.argmax(corr)]

    dt = np.median(np.diff(t_ref))
    shift = -best_lag * dt
    return shift


def compute_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    max_abs = np.max(np.abs(err))
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
    return {
        "MAE_mm": mae,
        "RMSE_mm": rmse,
        "MaxAbsError_mm": max_abs,
        "Correlation": corr
    }, err


def compare_by_active_segment(
    visual_npy_path,
    laser_xlsx_path,
    save_dir,
    direction="X",
    sheet_name="Sheet1",
    time_col="time[s]",
    disp_col="AI1-1-01[mm]",
    sign_visual=1.0,
    sign_laser=1.0,
):
    os.makedirs(save_dir, exist_ok=True)

    # 1) 读数据
    t_vis, y_vis = load_visual_npy(visual_npy_path, direction=direction)
    t_laser, y_laser = load_laser_xlsx(laser_xlsx_path, sheet_name, time_col, disp_col)

    y_vis = sign_visual * remove_offset(y_vis)
    y_laser = sign_laser * remove_offset(y_laser)

    # 2) 分别找运动有效段
    v0, v1 = detect_active_segment(t_vis, y_vis, smooth_win=9, threshold_ratio=0.08)
    l0, l1 = detect_active_segment(t_laser, y_laser, smooth_win=21, threshold_ratio=0.08)

    t_vis_act = t_vis[v0:v1+1]
    y_vis_act = y_vis[v0:v1+1]

    t_laser_act = t_laser[l0:l1+1]
    y_laser_act = y_laser[l0:l1+1]

    # 3) 先按“运动起点”粗对齐
    coarse_shift = t_vis_act[0] - t_laser_act[0]
    t_laser_shifted = t_laser + coarse_shift
    t_laser_act_shifted = t_laser_act + coarse_shift

    # 4) 再在活动段上做局部互相关细化
    fine_shift = refine_shift_by_local_xcorr(t_vis_act, y_vis_act, t_laser_act_shifted, y_laser_act)
    total_shift = coarse_shift + fine_shift
    t_laser_aligned = t_laser + total_shift

    # 5) 只取真正重叠的活动区间
    t_min = max(np.min(t_vis_act), np.min(t_laser_aligned))
    t_max = min(np.max(t_vis_act), np.max(t_laser_aligned))
    if t_max <= t_min:
        raise ValueError("活动段对齐后没有重叠区间，请检查数据。")

    vis_mask = (t_vis >= t_min) & (t_vis <= t_max)
    t_cmp = t_vis[vis_mask]
    y_vis_cmp = y_vis[vis_mask]

    f_laser = interp1d(
        t_laser_aligned, y_laser,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )
    y_laser_cmp = f_laser(t_cmp)

    # 6) 为了只比较相对位移，把重叠段起点再次统一归零
    y_vis_cmp = y_vis_cmp - y_vis_cmp[0]
    y_laser_cmp = y_laser_cmp - y_laser_cmp[0]

    metrics, err = compute_metrics(y_laser_cmp, y_vis_cmp)

    # 7) 保存数据
    out_df = pd.DataFrame({
        "time_s": t_cmp,
        f"visual_{direction}_mm": y_vis_cmp,
        "laser_mm": y_laser_cmp,
        "error_mm": err,
    })
    out_xlsx = os.path.join(save_dir, f"视觉激光对比_活动段对齐_{direction}.xlsx")
    out_df.to_excel(out_xlsx, index=False)

    metrics_df = pd.DataFrame([{
        "direction": direction,
        "visual_active_start_s": t_vis[v0],
        "visual_active_end_s": t_vis[v1],
        "laser_active_start_s": t_laser[l0],
        "laser_active_end_s": t_laser[l1],
        "coarse_shift_s": coarse_shift,
        "fine_shift_s": fine_shift,
        "total_shift_s": total_shift,
        **metrics
    }])
    out_metrics = os.path.join(save_dir, f"视觉激光误差指标_活动段对齐_{direction}.xlsx")
    metrics_df.to_excel(out_metrics, index=False)

    # 8) 画图
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t_vis, y_vis, label="视觉原始位移")
    plt.plot(t_laser, y_laser, label="激光原始位移")
    plt.axvline(t_vis[v0], color="b", linestyle="--", alpha=0.7)
    plt.axvline(t_vis[v1], color="b", linestyle="--", alpha=0.7)
    plt.axvline(t_laser[l0], color="orange", linestyle="--", alpha=0.7)
    plt.axvline(t_laser[l1], color="orange", linestyle="--", alpha=0.7)
    plt.title("原始时程与自动检测到的活动段")
    plt.xlabel("时间 (s)")
    plt.ylabel("位移 (mm)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t_cmp, y_vis_cmp, label="视觉位移（活动段对齐后）", linewidth=1.5)
    plt.plot(t_cmp, y_laser_cmp, label="激光位移（活动段对齐后）", linewidth=1.5)
    plt.title(f"视觉与激光位移对比（{direction}方向）")
    plt.xlabel("时间 (s)")
    plt.ylabel("位移 (mm)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t_cmp, err, label="误差 = 视觉 - 激光", linewidth=1.2)
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.title(
        f"误差曲线 | MAE={metrics['MAE_mm']:.3f} mm, "
        f"RMSE={metrics['RMSE_mm']:.3f} mm, "
        f"MaxAbs={metrics['MaxAbsError_mm']:.3f} mm"
    )
    plt.xlabel("时间 (s)")
    plt.ylabel("误差 (mm)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    out_fig = os.path.join(save_dir, f"视觉激光对比_活动段对齐_{direction}.png")
    plt.savefig(out_fig, dpi=200)
    plt.show()

    print("比较完成。")
    print(f"方向: {direction}")
    print(f"视觉活动段: {t_vis[v0]:.4f}s ~ {t_vis[v1]:.4f}s")
    print(f"激光活动段: {t_laser[l0]:.4f}s ~ {t_laser[l1]:.4f}s")
    print(f"粗对齐时间偏移: {coarse_shift:.6f} s")
    print(f"细化时间偏移: {fine_shift:.6f} s")
    print(f"总时间偏移: {total_shift:.6f} s")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    print(f"对齐数据已保存到: {out_xlsx}")
    print(f"误差指标已保存到: {out_metrics}")
    print(f"图像已保存到: {out_fig}")


if __name__ == "__main__":
    root_path = r"new_data5\cab3"

    visual_npy_path = os.path.join(root_path, "视觉位移计算结果-cal3d-ALIKED.npy")
    laser_xlsx_path = r"new_data5/truth/test3.xlsx"   # 如果 test1.xlsx 不在 cab1，请改成实际路径

    compare_by_active_segment(
        visual_npy_path=visual_npy_path,
        laser_xlsx_path=laser_xlsx_path,
        save_dir=root_path,
        direction="Z",
        sheet_name="Sheet1",
        time_col="time[s]",
        disp_col="AI1-1-01[mm]",
        sign_visual=1.0,
        sign_laser=1.0,
    )
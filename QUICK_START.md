# 快速使用说明

本文档用于现场快速运行当前同步版双目位移监测系统。

## 1. 当前默认模式

当前 `run_sync_monitoring.py` 默认配置：

```python
RESET_ROI = True
MAX_FRAMES = 500
TRACK_TOP_K = 8
TEMP_CALIBRATION_MODE = True
TEMP_BASELINE_MM = 70.0
SEND_DRY_RUN = False
VISUALIZE_TRACKING = True
```

含义：

- 每次启动都会重新框选 ROI；
- 默认最多跑 500 帧；
- 默认使用 8 个点做追踪和位移融合；
- 当前使用 70mm 手量基线 + 旧内参的临时标定演示模式；
- 默认真实 TCP 发送平台；
- 默认打开追踪可视化窗口。

## 2. 运行真实相机监测

```powershell
D:\Miniconda3\envs\pytorch\python.exe run_sync_monitoring.py
```

启动后流程：

1. 连接双目相机；
2. 采集并校正第一帧；
3. 弹出 ROI 窗口；
4. 框选石碑/结构上纹理清楚的区域；
5. 按 `Enter` 或空格确认；
6. 系统开始首帧匹配、后续追踪、位移计算和平台发送。

可视化窗口按键：

```text
q 或 Esc：停止监测
```

## 3. 平台发送配置

真实发送读取：

```text
config/socket.ini
```

需要确认：

```ini
[server]
host = 平台地址
port = 平台端口
imei = 设备编号
```

当前真实发送格式：

```text
RG{imei}MONIX
UD{imei}MONIX+{"1":{"t":...,"s":1,"p":[...]}}
```

## 4. 只演示平台发送

不接相机、不跑算法，只发送演示位移：

```powershell
D:\Miniconda3\envs\pytorch\python.exe send_data_demo.py
```

只打印、不联网：

```powershell
D:\Miniconda3\envs\pytorch\python.exe send_data_demo.py --dry-run --count 5
```

## 5. 临时标定说明

当前默认：

```python
TEMP_CALIBRATION_MODE = True
TEMP_BASELINE_MM = 70.0
```

临时模式使用：

- 手量基线约 `70mm`；
- `new_data5/cab/calibration_1.npy` 里的旧左相机内参；
- `abs(right_x - left_x)` 作为视差。

这个模式只用于演示采集、追踪、位移 payload 和平台发送闭环，不作为最终真实测量精度依据。

正式双目标定完成后，准备：

```text
new_data5/cab/R.npy
new_data5/cab/T.npy
new_data5/cab/calibration_1.npy
new_data5/cab/calibration_2.npy
```

然后将：

```python
TEMP_CALIBRATION_MODE = False
```

## 6. 常用调整

只本地调试、不发送平台：

```python
SEND_DRY_RUN = True
```

长期连续运行：

```python
MAX_FRAMES = None
```

图像偏暗：

```python
CAMERA_EXPOSURE_TIME_US = 20000.0
```

追踪太慢：

```python
TRACK_TOP_K = 5
```

## 7. 常见问题

没有 ROI 匹配点：

- ROI 需要框到纹理明显区域；
- 检查画面是否太暗、过曝或虚焦；
- 检查左右相机是否同步采到同一场景。

平台收不到数据：

1. 先运行 `send_data_demo.py` 验证纯发送；
2. 检查 `config/socket.ini`；
3. 检查网络和端口；
4. 确认 `SEND_DRY_RUN = False`。

位移数值不准：

- 当前是临时标定演示模式；
- 正式测量必须完成现场双目标定。

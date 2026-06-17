# 同步版双目位移监测流程说明

本文档对应当前 `run_sync_monitoring.py` 的实际实现。当前版本优先用于现场演示和链路联调：相机在线采集、立体校正、ROI 内首帧匹配、旧版增强追踪、临时标定三维位移、平台 TCP 发送。

重要说明：当前默认启用了临时标定模式，使用约 70mm 手量基线和旧内参计算演示位移。该模式用于验证“采集、追踪、位移 payload、平台发送”链路，不等同于最终真实测量标定。

## 1. 运行入口

```powershell
D:\Miniconda3\envs\pytorch\python.exe run_sync_monitoring.py
```

脚本默认会：

- 连接真实双目相机；
- 每次启动重新框选 ROI；
- 打开追踪可视化窗口；
- 默认真实 TCP 发送平台数据；
- 使用临时 70mm 基线计算演示三维位移。

纯平台发送演示入口：

```powershell
D:\Miniconda3\envs\pytorch\python.exe send_data_demo.py
```

只打印、不联网：

```powershell
D:\Miniconda3\envs\pytorch\python.exe send_data_demo.py --dry-run --count 5
```

## 2. 当前主流程

```text
continuous_capture_online
-> on_frame(frame_packet)
-> StereoRectifier.rectify_frame_packet(..., save_images=False)
-> left_rect_image / right_rect_image
-> 每次启动重新选择 ROI
-> match_initial_candidates(...)
-> 旧版 SuperPoint + LightGlue 首帧左右匹配
-> 只保留左图点落在 ROI 内且置信度较高的 matches
-> 初始化左右 SyncPointTracker
-> 后续帧左右 tracker 同步 update
-> LightGlue 追踪；失败则模板匹配；再失败则轨迹预测
-> 得到 left_xy / right_xy
-> 临时标定模式下用 70mm 手量基线 + 旧左相机内参 + 绝对视差计算 xyz
-> displacement = current_xyz - reference_xyz
-> 多点离群过滤后取平均位移
-> 生成 11 字段 payload
-> UD{imei}MONIX+json TCP 发送
```

## 3. ROI 选择

当前配置：

```python
RESET_ROI = True
```

因此每次启动 `run_sync_monitoring.py` 都会重新弹出 ROI 窗口，不再优先读取旧的 `runtime_state/roi_config.json`。

ROI 窗口会自动缩小大图显示，默认最大显示 `1280x720`，但保存的 ROI 坐标会映射回原始图坐标。

操作：

```text
拖框选择 ROI
Enter 或空格确认
c 取消
```

## 4. 首帧匹配逻辑

首帧检测已按旧版本核心逻辑对齐：

```text
SuperPoint(max_num_keypoints=1000)
-> LightGlue(features="superpoint")
-> rbd 去 batch
-> 读取 matches
-> 只额外保留左图点落在 ROI 内且 score >= min_confidence 的 match
-> 按 LightGlue score 从高到低排序
```

当前首帧不按 dy、视差范围做额外候选过滤，`min_confidence=0.2` 保持默认匹配阈值。日志中会看到：

```text
legacy SuperPoint+LightGlue selected N ROI matches
```

## 5. 后续追踪逻辑

`online_tracking/sync_point_tracker.py` 当前迁移了旧版增强追踪思路：

1. 优先用 SuperPoint + LightGlue 做前后帧特征匹配；
2. 如果当前特征点 ID 匹配不上，用上一成功点附近的模板匹配兜底；
3. 如果模板匹配也失败，用历史轨迹预测兜底。

主流程会打印当前追踪方式：

```text
method=(feature,feature)
method=(template,feature)
method=(prediction,template)
```

含义：

- `feature`：LightGlue 前后帧匹配成功；
- `template`：LightGlue 失败后模板匹配成功；
- `prediction`：特征和模板都失败，用轨迹预测。

## 6. 标定参数和三维计算

### 6.1 当前默认：临时标定演示模式

当前默认：

```python
TEMP_CALIBRATION_MODE = True
TEMP_BASELINE_MM = 70.0
CALIBRATION_FOLDER = r"new_data5\cab"
```

临时三维计算位于：

```text
online_tracking/displacement_utils.py::compute_xyz_from_stereo_points_temp_calibration
```

当前临时模式的参数来源：

```text
基线 baseline：手量约 70mm，即 TEMP_BASELINE_MM = 70.0
左相机内参：读取 new_data5/cab/calibration_1.npy 中的 cameraMatrix
右相机内参：临时三维公式不使用，但文件加载结构里仍保留
R.npy/T.npy：临时三维公式不使用真实 R/T
Q.npy：临时三维公式不使用 Q
```

临时公式：

```python
disparity = abs(right_x - left_x)
z = fx * baseline_mm / disparity
x = (left_x - cx) * z / fx
y = (left_y - cy) * z / fy
xyz = [x, y, z]
```

也就是说，临时模式只使用：

```text
left_xy
right_xy
旧左相机内参 fx/fy/cx/cy
手量基线 70mm
```

使用 `abs(right_x - left_x)` 是为了在没有可靠外参/左右方向未最终确认时避免 `Z` 因视差符号变成负数。这样可以让演示链路稳定跑通，但它牺牲了严格的双目标定几何含义。

启动时会打印：

```text
[calibration][temp] enabled: baseline=70.0mm, old intrinsics, absolute disparity. For demo only, not final measurement calibration.
```

### 6.2 当前临时模式的限制

当前输出的 `xyz` 和 `displacement` 可以用于平台链路演示，但不应作为最终毫米级真实测量结果。原因：

1. 基线是手量约值，不是标定优化结果；
2. 旋转矩阵暂按理想平行相机处理，没有真实外参修正；
3. 只使用旧左相机内参，若当前相机/分辨率/校正方式变化，尺度会有偏差；
4. 使用绝对视差会丢失左右方向符号信息；
5. 当前数值主要用于验证在线闭环和平台接收能力。

推荐对甲方说明：

```text
当前为临时标定演示模式，使用 70mm 手量基线和旧内参，
用于验证 Jetson 在线采集、追踪、位移数据打包和平台上传闭环。
正式测量精度需要现场完成双目标定后替换真实标定参数。
```

### 6.3 切回正式标定模式

正式双目标定完成后，需要准备或确认：

```text
new_data5/cab/R.npy
new_data5/cab/T.npy
new_data5/cab/calibration_1.npy
new_data5/cab/calibration_2.npy
可选：new_data5/cab/Q.npy
```

然后将：

```python
TEMP_CALIBRATION_MODE = False
```

即可切回正式三维计算链路：

```text
compute_xyz_from_stereo_points(...)
```

正式模式会使用旧脚本兼容公式和真实标定文件，适合后续真实测量联调。

## 7. 多点位移融合

当前默认：

```python
TRACK_TOP_K = 8
DISPLACEMENT_OUTLIER_FLOOR_MM = 5.0
```

每帧会收集多个有效点的位移，先按位移分布过滤离群点，再对保留点取平均。若发生过滤，日志会显示：

```text
[disp][filter] frame_id=..., kept=..., dropped=...
```

## 8. 可视化窗口

当前默认：

```python
VISUALIZE_TRACKING = True
VISUALIZATION_MAX_WIDTH = 1600
VISUALIZATION_TEXT_SCALE = 0.75
```

窗口名：

```text
sync monitoring tracking
```

窗口显示内容：

- 左侧为左校正图；
- 右侧为右校正图；
- 彩色圆圈为当前追踪点；
- 彩色连线表示当前左右点对；
- 底部左侧显示 `frame_id`、状态、有效点数、位移和发送状态。

按键：

```text
q 或 Esc 停止监测
```

若 Jetson 没有桌面环境，OpenCV 窗口可能打开失败，脚本会打印 `[visualize][warn]` 并自动关闭可视化。

## 9. 发送协议

点记录复用 `7_pack_data.py::build_point_record` 的 11 字段结构：

```text
[左点ID, 右点ID, 通道, x像素移动, y像素移动, z像素移动,
 x位移, y位移, z位移, 匹配置信度, 跟踪置信度]
```

当前发送使用稳定点 ID，不再把 SuperPoint 当前帧索引当作平台点 ID：

```text
point_id=0 -> left_id=1, right_id=2
point_id=1 -> left_id=3, right_id=4
```

帧结构：

```json
{
  "t": 1781628694281,
  "s": 1,
  "p": [[1, 2, 1, 0.0, 0.0, 0.0, 1.2, 0.3, -0.1, 1.0, 0.9]]
}
```

TCP 发送字符串：

```text
UD{imei}MONIX+{"1":{"t":...,"s":1,"p":[...]}}
```

发送前会先注册：

```text
RG{imei}MONIX
```

实际 socket 包使用 `utils.byte_utils.get_length_prefix_bytes(...)` 添加 4 字节长度前缀。

默认：

```python
SEND_DRY_RUN = False
```

也就是默认真实发送。若只想本地调试，将其改为：

```python
SEND_DRY_RUN = True
```

## 10. 关键配置

`run_sync_monitoring.py` 顶部常量：

```python
CALIBRATION_FOLDER = r"new_data5\cab"
ROI_CONFIG_PATH = os.path.join("runtime_state", "roi_config.json")
RESET_ROI = True
MAX_FRAMES = 100
TRACK_TOP_K = 8
TEMP_CALIBRATION_MODE = True
TEMP_BASELINE_MM = 70.0
CAMERA_EXPOSURE_TIME_US = 16000.0
SEND_DRY_RUN = False
VISUALIZE_TRACKING = True
VISUALIZATION_MAX_WIDTH = 1600
SOCKET_CONFIG_PATH = os.path.join("config", "socket.ini")
SEND_CHANNEL_ID = "1"
```

长期连续监测时可改为：

```python
MAX_FRAMES = None
```

## 11. 常见问题

### 11.1 首帧没有 ROI matches

检查：

- ROI 是否框到有纹理的区域；
- 图像是否太暗、过曝或虚焦；
- 左右图是否来自同一时刻；
- LightGlue/SuperPoint 是否正常导入。

### 11.2 Z 或位移很离谱

当前是临时标定模式，只适合演示链路。正式测量必须重新做双目标定，然后关闭：

```python
TEMP_CALIBRATION_MODE = False
```

### 11.3 平台收不到数据

按顺序检查：

1. 先运行 `send_data_demo.py` 验证纯发送；
2. 检查 `config/socket.ini` 的 `host`、`port`、`imei`；
3. 检查 Jetson 网络、防火墙、服务器端口；
4. 确认日志里出现 `RG{imei}MONIX` 注册；
5. 确认 `SEND_DRY_RUN=False`；
6. 确认平台要求的通道号是否为当前 `SEND_CHANNEL_ID="1"`。

### 11.4 画面追踪跳点

当前已按旧版增强逻辑迁移：LightGlue 失败后会模板匹配，再失败会预测。但如果 ROI 纹理弱、画面抖动大或光照变化大，仍可能跳点。可尝试：

- 缩小 ROI 到纹理明显区域；
- 降低曝光过曝风险；
- 将 `TRACK_TOP_K` 调小到 `5`；
- 正式标定后再验证真实位移。

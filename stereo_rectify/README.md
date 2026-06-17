# stereo_rectify 立体校正模块说明

## 1. 模块定位
本模块负责：
- 加载双目标定参数；
- 对左右图像执行立体校正；
- 在在线模式中输出 `RectifiedPacket`；
- 不负责图像采集、跟踪、位移计算。

## 2. 推荐在线用法
当前主流程推荐使用：
- `StereoRectifier`
- `StereoRectifier.rectify_frame_packet(...)`

示例：
```python
from stereo_rectify import StereoRectifier

rectifier = StereoRectifier(calibration_folder="camera_calibration/your_case")
rectified_packet = rectifier.rectify_frame_packet(
    frame_packet=frame_packet,
    save_images=False,
)
```

说明：
- 优先使用 `frame_packet.left_image/right_image`；
- 不依赖本地图像路径；
- `save_images=False` 时不落盘；
- 输出 `left_rect_image/right_rect_image` 供后续模块直接使用。

## 3. RectifiedPacket 字段说明
`RectifiedPacket` 主要字段：
- `frame_id`
- `left_rect_image/right_rect_image`
- `left_rect_path/right_rect_path`
- `q_matrix`
- `calibration_tag`

## 4. Q 矩阵说明
- `Q` 在 `StereoRectifier.prepare(...)` 内通过 `stereoRectify` 计算得到；
- 同一标定参数、同一图像尺寸下会复用已缓存的 map 和 `Q`；
- 可通过 `rectifier.q_matrix` 获取当前缓存的 `Q`；
- `return_q_matrix=True` 时，`RectifiedPacket.q_matrix` 会带出该矩阵。

## 5. 离线批处理接口
- `stereo_rectification(...)` 是 legacy/offline 批处理接口；
- 适合历史目录图像的批量校正；
- 不推荐作为在线主流程入口。

## 6. 注意事项
- 当前不保证校正效果正确，效果主要取决于标定参数质量；
- 输入图像尺寸应与标定条件和实际使用条件一致；
- 在线模式优先走内存图像链路；
- 仅当输入 `packet` 不含图像数组时，才会回退到路径读图。

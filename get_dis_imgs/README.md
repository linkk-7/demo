# get_dis_imgs 采集模块说明

## 1. 模块定位
`get_dis_imgs` 只负责双目相机采集并输出 `FramePacket`。

本模块负责：
- 海康 SDK 采集左右图像；
- BGR 转换（供 OpenCV 直接处理）；
- 生成含时间戳/帧号/图像的 `FramePacket`。

本模块不负责：
- 立体校正、跟踪、位移计算；
- 服务器上传。

## 2. 两种运行模式
### 2.1 在线采集模式（推荐）
入口：`continuous_capture_online(...)`

特点：
- 默认不落盘；
- `FramePacket.left_image/right_image` 为有效图像数组；
- 可直接传给后续模块处理；
- 该模式下采集 + BGR 可接近 30 pair/s。

示例：
```python
from get_dis_imgs import continuous_capture_online

continuous_capture_online(
    save_folder_base="input/cab_online",
    on_frame=None,
)
```

### 2.2 存盘循环覆盖模式
入口：`continuous_capture_disk(...)`

特点：
- 连续保存 JPG；
- 默认 `jpeg_quality=90`；
- `keep_last_n_pairs` 控制只保留最近 N 对；
- 适合调试留痕、离线检查。

示例：
```python
from get_dis_imgs import continuous_capture_disk

continuous_capture_disk(
    save_folder_base="input/cab",
    keep_last_n_pairs=1000,
    jpeg_quality=90,
    on_frame=None,
)
```

## 3. FramePacket 字段说明
核心字段包括：
- `frame_id`
- `left_host_timestamp/right_host_timestamp`
- `left_dev_timestamp_raw/right_dev_timestamp_raw`
- `left_frame_num/right_frame_num`
- `left_filename/right_filename`
- `left_path/right_path`
- `left_image/right_image`

说明：
- 在线模式：`left_image/right_image` 有效；`left_path/right_path` 为潜在路径字符串（默认不写盘）。
- 存盘模式：`left_path/right_path` 为真实保存路径；默认可不在 packet 中保留图像数组。

## 4. 与后续模块关系
在线模式下，`FramePacket` 可直接传给立体校正层，优先使用内存图像数组处理。

## 5. 注意事项
- 在线模式不要长期缓存大量 `FramePacket`，用完及时释放；
- 存盘模式帧率通常低于在线模式（编码和磁盘 I/O 会带来开销）；
- 不建议把 PNG 作为连续在线保存格式；
- 若后续接服务器，建议从内存图像编码并异步发送。


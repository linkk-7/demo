# get_dis_imgs 采集层设计说明

## 1. 模块定位
`get_dis_imgs/` 是当前工程的图像采集层，负责把双目相机的原始帧稳定采集下来，并以“左右成对 + 帧级元信息”的形式对外提供。

当前采集层**负责**：
- 调用海康 SDK 取帧（`MV_CC_GetImageBuffer`）
- 解析单帧元信息（设备时间戳、主机时间戳、帧号）
- 生成一对左右图并落盘
- 按 `FramePacket` 返回当前帧对的数据与元信息
- 可选执行“最近 N 对”保留策略（retention）

当前采集层**不负责**：
- 立体校正
- 特征跟踪
- 位移计算
- `get_t / rename / 2.3_*` 后处理流程
- 左右设备时间戳跨相机物理对时判定

补充：`2.3_gain_t_and_rename_disp_images.py` 与 `get_t` 模块已降级为
legacy/offline 兼容工具，不再属于当前主流程步骤。

---

## 2. 目录与文件说明
当前 `get_dis_imgs/` 主要文件：

- `models.py`
  - 定义采集层核心数据结构：`FrameMeta`、`FramePacket`、`RetentionRecord`
- `capture.py`
  - 采集核心实现：SDK 取图、元信息提取、成对保存、retention、兼容旧连续显示采集
- `api.py`
  - 对外统一 API：`capture_and_save_frame_pair(...)`、`continuous_capture(...)`
- `__init__.py`
  - 包级导出入口，便于顶层脚本直接 `from get_dis_imgs import ...`

与顶层入口关系：
- `2.1_take_disp_images.py` 当前通过 `continuous_capture(...)` 调用采集层。
- `2.1_take_disp_images.py` 当前未固定 `start_frame_id=1`，默认由采集层自动推断起始编号。
- 顶层脚本不再直接拼接底层 SDK 细节。

---

## 3. 当前采集流程说明
### 3.1 单帧获取
在 `capture.py` 的 `_get_image(cam)` 中：
1. 调用 SDK：`cam.MV_CC_GetImageBuffer(stOutFrame, 2000)`
2. 复制底层缓冲区数据到用户侧
3. 释放 SDK 缓冲：`MV_CC_FreeImageBuffer`
4. 像素格式转 BGR：`_to_bgr(...)`
5. 解析帧元信息：`_extract_frame_meta(stOutFrame.stFrameInfo)`
6. 返回 `(image, frame_meta)`

### 3.2 FrameMeta 生成
`_extract_frame_meta(...)` 中从 `stFrameInfo` 提取：
- `dev_timestamp_raw = (nDevTimeStampHigh << 32) | nDevTimeStampLow`
- `host_timestamp = nHostTimeStamp`
- `frame_num = nFrameNum`

### 3.3 左右成对
`capture_and_save_frame_pair(...)` 中一次循环分别抓取左、右两帧；当两侧都拿到有效图像和元信息时，视为当前一对。

> 当前“一对”的主标识是共同 `frame_id`（由调用方推进），而不是共享某个统一时间戳。

### 3.4 图像保存
`_save_pair_images(...)` 把左右图分别保存到：
- `save_folder_base/left/`
- `save_folder_base/right/`

文件名使用各自 `host_timestamp` + 共同 `frame_id`。

### 3.5 返回 FramePacket
每成功保存一对后，返回一个 `FramePacket`，包含：
- 配对标识（`frame_id`）
- 左右 host/dev/frame_num 元信息
- 左右文件名与路径
- 可选图像数组

### 3.6 连续采集
`continuous_capture(...)` 内部循环调用 `capture_and_save_frame_pair(...)`，逐帧保存图像对并更新 `frame_id`，可设置 `max_frames`，并支持 `Ctrl+C` 中断。

- 当 `start_frame_id` 显式传入时：使用该值作为起点（超出范围会归一化到有效区间）。
- 当 `start_frame_id` 未传入时：自动扫描缓存目录（`left/`、`right/`）推断起始 `frame_id`。

### 3.7 frame_id 管理逻辑
当前 `frame_id` 设计与行为如下：

- `frame_id` 的职责：
  - 表示左右图属于同一对
  - 作为程序内部帧对编号，供日志、`FramePacket`、下游模块引用
- `frame_id` 不是时间戳
- `frame_id` 不是 retention 删除依据

当前实现采用有限范围循环编号（默认 `1~999999`）：

1. 运行中每成功保存一对图，`frame_id` 递增。
2. 当超过上限时回绕到 `1`。
3. 启动时不再仅用 `max+1` 直接作为起点，而是：
   - 扫描 `left/` 和 `right/` 文件名
   - 提取当前目录已占用的 `frame_id`，构造 `occupied_ids`
   - 候选起点先取 `max_frame_id + 1`（超上限回绕到 `1`）
   - 从候选起点循环查找第一个未被 `occupied_ids` 占用的编号，作为新的 `start_frame_id`

这样即使目录中同时存在高位编号和低位编号（说明曾发生回绕），重启后也不会直接撞上当前仍存在的文件名。

示例：

- 若当前目录最大 `frame_id = 487`，下次启动起始编号为 `488`。
- 若目录中同时存在 `999700~999999` 和 `1~100`，下次启动起始编号为 `101`（避开已占用的 `1~100`）。

---

## 4. 数据结构说明
## 4.1 FrameMeta
```python
@dataclass(slots=True)
class FrameMeta:
    dev_timestamp_raw: int
    host_timestamp: int
    frame_num: int
```
字段含义：
- `dev_timestamp_raw`：设备原始时间戳（高低位拼接后的原始值）
- `host_timestamp`：SDK 帧信息中的主机时间戳（`nHostTimeStamp`）
- `frame_num`：SDK 帧号（`nFrameNum`）

## 4.2 FramePacket
```python
@dataclass(slots=True)
class FramePacket:
    frame_id: int
    left_host_timestamp: int
    right_host_timestamp: int
    left_dev_timestamp_raw: int
    right_dev_timestamp_raw: int
    left_frame_num: int
    right_frame_num: int
    left_filename: str
    right_filename: str
    left_path: str
    right_path: str
    left_image: Optional[np.ndarray] = None
    right_image: Optional[np.ndarray] = None
```
字段含义：
- `frame_id`：当前左右图对的配对 ID
- `left_host_timestamp / right_host_timestamp`：左右各自主机时间戳
- `left_dev_timestamp_raw / right_dev_timestamp_raw`：左右各自设备原始时间戳
- `left_frame_num / right_frame_num`：左右各自 SDK 帧号
- `left_filename / right_filename`：落盘文件名
- `left_path / right_path`：落盘完整路径
- `left_image / right_image`：可选图像数组（由 `include_images` 控制）

## 4.3 RetentionRecord
```python
@dataclass(slots=True)
class RetentionRecord:
    frame_id: int
    left_path: str
    right_path: str
```
用途：
- retention 队列只保存“删除旧文件所需最小信息”，避免持有大对象。

---

## 5. 时间戳设计说明
### 5.1 `dev_timestamp_raw` 是什么
- 来自设备侧时间戳字段：`nDevTimeStampHigh/nDevTimeStampLow`
- 在代码中仅做“原始值拼接”，不做物理单位换算

### 5.2 `host_timestamp` 是什么
- 来自 SDK 帧信息字段：`nHostTimeStamp`
- 是当前命名策略优先使用的时间戳

### 5.3 为什么当前命名使用 `host_timestamp`
- 该值直接来自 SDK 帧信息，便于跨流程读取与记录
- 作为“文件名时间标签”更直观，避免把设备原始计数值直接暴露为唯一命名依据

### 5.4 为什么仍保留 `dev_timestamp_raw`
- 设备原始时间信息是底层关键元数据
- 后续若需要更严谨时序分析/对时策略，可直接使用该原始字段

### 5.5 为什么当前不引入 `pair_delta_ms`
- 当前尚未确认左右设备时间戳是否可直接跨相机比较
- 在该前提不明确时，不应在采集层计算并输出可能误导的差值

### 5.6 为什么不生成统一 pair 主时间戳
- 采集层保留左右各自原始时间，不提前“合并”成单一时间
- 需要单一时间的下游模块应按自身规则派生，避免采集层提前做不可逆语义决策

---

## 6. 文件命名规则
当前成对保存命名格式：
- `L_{left_host_timestamp}_{frame_id:06d}.png`
- `R_{right_host_timestamp}_{frame_id:06d}.png`

示例：
- `L_1713812345678_000123.png`
- `R_1713812345692_000123.png`

说明：
- 左右图各自使用自己的 `host_timestamp`
- 共同 `frame_id` 表示“属于同一对”

---

## 7. 图像保存与 retention 机制
### 7.1 保存目录结构
成对接口默认保存到：
- `save_folder_base/left/*.png`
- `save_folder_base/right/*.png`

> 兼容方法 `CameraController.get_dis_imgs()` 仍会保存到 `Camera_0/Camera_1`，主要用于旧流程。

### 7.2 `keep_last_n_pairs` 作用
- 设置为正整数时，只保留最近 N 对图像
- 超出后按“左右一对”删除最旧记录

### 7.3 为什么 retention 不保存整个 `FramePacket`
- `FramePacket` 可能带 `left_image/right_image` 大数组
- retention 仅需删除路径信息，保存轻量 `RetentionRecord` 更安全

### 7.4 `keep_last_n_pairs=None` 时为什么不会无限增长 metadata
- `_track_and_enforce_pair_retention(...)` 在 `None` 时直接返回，不 append
- 因此 retention 元数据不会持续增长

### 7.5 `include_images=True` 的内存行为
- `FramePacket` 会携带当前帧图像数组
- retention 队列不再持有 `FramePacket`，不会因 retention 长期缓存图像
- 是否持续占用内存取决于调用方是否自行保存历史 `FramePacket`

### 7.6 retention 与 frame_id 的关系
- retention 只负责控制“最近 N 对图像”的保留数量。
- retention 删除依据是保留队列中的最旧记录及其路径，不依赖 `frame_id` 的大小关系。
- 因此即使 `frame_id` 回绕，retention 行为也不受影响。

---

## 8. 对外接口说明
## 8.1 `capture_and_save_frame_pair(...)`
位置：`get_dis_imgs.api.capture_and_save_frame_pair`

主要参数：
- `save_folder_base: str` 保存根目录
- `frame_id: int` 当前帧对 ID
- `keep_last_n_pairs: Optional[int]` 最近 N 对保留上限
- `include_images: bool` 是否在返回包中包含图像数组
- `left_camera_index / right_camera_index` 左右相机索引

返回值：
- `FramePacket`

示例：
```python
from get_dis_imgs import capture_and_save_frame_pair

packet = capture_and_save_frame_pair(
    save_folder_base="new_data5/cab4_v",
    frame_id=123,
    keep_last_n_pairs=500,
    include_images=False,
)
print(packet.left_path, packet.right_path)
```

## 8.2 `continuous_capture(...)`
位置：`get_dis_imgs.api.continuous_capture`

主要参数：
- `save_folder_base: str` 保存根目录
- `start_frame_id: Optional[int]` 起始帧对 ID（默认 `None`）
- `keep_last_n_pairs: Optional[int]` 最近 N 对保留上限
- `include_images: bool` 是否在返回包中包含图像数组
- `max_frames: Optional[int]` 最大采集对数（`None` 表示持续）
- `left_camera_index / right_camera_index` 左右相机索引
- `frame_id_max: int` `frame_id` 上限（默认 `999999`）

`start_frame_id` 行为说明：
- 显式传入：使用传入值作为起点（并归一化到合法范围）。
- 未传入（默认）：自动扫描缓存目录推断起点，并避开当前已占用编号。

返回值：
- `Optional[FramePacket]`（最后一对，若无采集成功则可能为 `None`）

示例：
```python
from get_dis_imgs import continuous_capture

last_packet = continuous_capture(
    save_folder_base="new_data5/cab4_v",
    keep_last_n_pairs=500,
)
```

---

## 9. 当前采集层已经保证的内容
- 能持续采集并保存左右图
- 能通过 `frame_id` 形成“左右一对”
- 每对都能返回 `FramePacket`
- 能保留 `host_timestamp / dev_timestamp_raw / frame_num` 元信息
- 能按最近 N 对保留并删除最旧图像对
- 顶层脚本可通过统一 API 调用（如 `2.1_take_disp_images.py`）
- `frame_id` 在回绕后重启时，能通过“占用集合避让”逻辑避免直接撞上当前目录中的已有文件名

---

## 10. 当前采集层尚未保证的内容（边界）
- 左右相机严格硬件同步：当前未保证
- 左右 `dev_timestamp_raw` 是否可直接比较：当前未确认
- `dev_timestamp_raw` 的物理单位：当前仓库未明确给出
- 下游校正/跟踪/位移模块与采集层的完整在线打通：当前未完成
- 采集层当前不输出 `pair_delta_ms`

---

## 11. 后续模块对接建议
### 11.1 立体校正模块如何使用 `FramePacket`
- 推荐以 `left_path/right_path` 为主输入，按 `frame_id` 组织结果
- 如要保留时序信息，可同步记录 `left_host_timestamp/right_host_timestamp`

### 11.2 下游优先使用 `path` 还是 `image`
- 长时间运行与解耦场景：优先 `path`
- 低延迟、同进程流水线：可选 `include_images=True` 直接传数组
- 若使用 `include_images=True`，建议下游及时释放或不长期缓存

### 11.3 采集层更适合作为哪种输入源
- 当前更适合作为“文件驱动型流水线”的统一输入源（稳定、可追溯）
- 后续若推进在线化，可继续复用 `FramePacket` 作为跨模块标准载体

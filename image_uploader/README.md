# image_uploader 说明

## 1. 模块定位
`image_uploader` 是独立上传模块：把内存图像（`np.ndarray`）编码为 JPG 后，通过 TCP 发送到平台。  
不改采集、校正、跟踪、位移计算核心逻辑。

## 2. 图片上传（9812）
- 传输方式：TCP socket，连接 `t4.tncet.com:9812`
- 协议头：`IMAGE-JPG&&`
- 内容格式：  
  `IMAGE-JPG&&{authCode}&&{sensor_param_id}&&{timestamp}&&{bad}&&{displacement_value}&&` + `jpg_base64`
- 编码：`cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])`
- 默认 JPG 质量：`20`
- 图片数据：JPG buffer 经过 `base64.b64encode(...)`
- 发送：`get_length_prefix_bytes(payload)` 后 `sendall`

说明：长度前缀当前复用项目已有实现（4 字节大端）。是否与平台最终一致，需联调确认。

最小单图发送脚本：`send_one_image_to_platform.py`。测试 authCode 上传次数有限，脚本默认
`CONFIRM_SEND = False`，不会自动发送。使用前必须手动填写 `SENSOR_PARAM_ID` 和
`IMAGE_PATH`，确认无误后再把 `CONFIRM_SEND` 改为 `True`，脚本只会发送一张图一次。
不要用目录循环、左右图自动发送或批量上传测试 `xian_beilin_test`。

## 3. 9879 同步（按文档 3.4 代码流程）
`config_receiver.py` 通过 TCP 连接 `t4.tncet.com:9879`。平台确认该端口连接建立后需要：

1. 立即发送注册包：`RG{imei}MONIX`
2. 定时发送心跳包：`HB{imei}MONIX`
3. `imei` 与系统上采集箱的 IMEI 码保持一致

注册包和心跳包的发送方式参考旧入口 `8_tcp_client.py`：先 UTF-8 编码，再调用
`get_length_prefix_bytes(...)` 加 4 字节大端长度前缀，最后 `sendall(...)`。

当前测试脚本默认 IMEI：`F44EB4DF6F99`，默认心跳间隔：`15` 秒。

收到平台下发数据后，按截图实现这条解析链路：

1. `handle_recv_messages(self, data: bytes)`
2. `message = data.decode("utf-8")`（为兼容现场，还支持“4字节大端长度前缀 + JSON”fallback）
3. `message_dict = json.loads(message)`
4. `recv_data = RecvData.model_validate_json(message)`
5. `if "type" in message_dict and message_dict["type"] == "SYNC_DATA":`
6. `sync_ipc_data(recv_data)`

`sync_ipc_data(origin_data)` 内部流程：
1. 判空：`origin_data` 或 `origin_data.sync_detail`
2. `data = origin_data.sync_detail`
3. 读取 `cameraList` 并 `upsert_camera`
4. 读取 `param`
5. 遍历 `param.items()`
6. `param_data = {k: v for k, v in value.items() if k != "camera_ids"}`
7. `old_main_param = find_by_id(value["id"])`
8. 有则 `update_param_by_recv(param_data)`，无则 `save(MonitorParam.model_validate(param_data))`

`MonitorParam` 中保留字段语义（含 `sensor_param_id`）。

9879 配置同步不是 9812 图片发送的必要条件。如果 9879 不通，可以从平台或人工记录中手动
填写 `sensor_param_id`，再测试 9812 图片上传。

## 4. 当前保存结果
同步后会在内存里提供：
- `camera_list`
- `params_by_id`
- `sensor_param_ids`（`param_id -> sensor_param_id`）
- `camera_ids_by_param_id`（用于人工确认映射）

注意：`config_receiver` 不自动推断 left/right。  
左图/右图 `sensor_param_id` 需要根据 `name/type/camera_ids` 或平台明确规则确认。

## 5. 9879 联调脚本
脚本：`test_config_receiver_9802.py`

运行：
```bash
python test_config_receiver_9802.py
```

脚本会：
- 连接 `t4.tncet.com:9879`
- 发送带长度前缀的注册包 `RG{imei}MONIX`
- 定时发送带长度前缀的心跳包 `HB{imei}MONIX`
- 循环接收并调用 `handle_recv_messages(data)`
- 打印消息预览
- 打印 `param_id -> sensor_param_id`
- 打印每个 param 的 `name/type/camera_ids/sensor_param_id` 摘要
- 成功后保存 `sensor_param_ids_from_9879.json`

## 6. 连上但无数据时请确认
1. IMEI 是否与平台系统上采集箱 IMEI 一致
2. `type` 实际值是否真是 `SYNC_DATA`
3. 连接方向是否确为“客户端主动连平台 9879”
4. 是否带长度前缀/分包规则
5. 是否需要平台端手动触发同步

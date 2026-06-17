"""9879 端口传感器配置同步接收模块。

本模块按接口文档第 3.4 节中的流程处理平台下发的同步数据：
1) handle_recv_messages(data: bytes)
2) message = data.decode("utf-8")
3) message_dict = json.loads(message)
4) recv_data = RecvData.model_validate_json(message)
5) if type == SYNC_DATA: sync_ipc_data(recv_data)
"""

from __future__ import annotations

import json
import socket
import threading
import time
import warnings
from typing import Any, Dict, Optional, Tuple

from utils.byte_utils import get_length_prefix_bytes

from .models import ImageUploadConfig


class RecvData:
    """接收数据模型，保留与 `model_validate_json` 一致的调用形式。"""

    def __init__(self, type: Optional[str] = None, sync_detail: Optional[Dict[str, Any]] = None):
        self.type = type
        self.sync_detail = sync_detail

    @classmethod
    def model_validate_json(cls, message: str) -> "RecvData":
        obj = json.loads(message)  # 把字符串变成 Python 对象
        if not isinstance(obj, dict):
            raise ValueError("RecvData.model_validate_json 需要 JSON 对象。")
        sync_detail = obj.get("sync_detail")
        if sync_detail is not None and not isinstance(sync_detail, dict):
            raise ValueError("sync_detail 存在时必须是字典。")
        return cls(type=obj.get("type"), sync_detail=sync_detail)


class MonitorParam:
    """
    监测参数

    对应主要字段：
    - id: Optional[int]
    - last_time: Optional[int]
    - work_status: int
    - sensor_param_id: Optional[int]
    """

    def __init__(
        self,
        id: Optional[int] = None, # 主键
        last_time: Optional[int] = None,
        work_status: int = 0,
        sensor_param_id: Optional[int] = None,  # 传感器参数id
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.last_time = last_time
        self.work_status = work_status
        self.sensor_param_id = sensor_param_id
        self.extra = extra or {}

    @classmethod
    # 把 param_data 字典，整理成 MonitorParam 对象
    def model_validate(cls, param_data: Dict[str, Any]) -> "MonitorParam":
        if not isinstance(param_data, dict):
            raise ValueError("MonitorParam.model_validate 需要字典输入。")

        def _to_int_or_none(v: Any) -> Optional[int]:
            if v is None:
                return None
            try:
                return int(v)
            except Exception:
                return None

        known_keys = {"id", "last_time", "work_status", "sensor_param_id"}
        extra = {k: v for k, v in param_data.items() if k not in known_keys}
        return cls(
            id=_to_int_or_none(param_data.get("id")),
            last_time=_to_int_or_none(param_data.get("last_time")),
            work_status=int(param_data.get("work_status", 0) or 0),
            sensor_param_id=_to_int_or_none(param_data.get("sensor_param_id")),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]: # 把对象再转回字典，方便存储和输出
        data = {
            "id": self.id,
            "last_time": self.last_time,
            "work_status": self.work_status,
            "sensor_param_id": self.sensor_param_id,
        }
        data.update(self.extra)
        return data


class _InMemoryCameraDao:
    """内存版 camera DAO，提供 upsert 语义，便于联调时保存同步结果。"""

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}

    def _camera_key(self, camera: Dict[str, Any]) -> str:
        for k in ("id", "camera_id", "cameraId", "name"):
            if k in camera and camera[k] is not None:
                return str(camera[k])
        return str(len(self._items))

    def upsert_camera(self, camera: Dict[str, Any]) -> None:
        key = self._camera_key(camera)
        self._items[key] = dict(camera)

    def list_all(self) -> list:
        return list(self._items.values())


class _InMemoryMonitorParamDao:
    """内存版 monitor param DAO，提供查询、更新、保存语义。"""

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}

    def find_by_id(self, param_id: Any) -> Optional[Dict[str, Any]]:
        if param_id is None:
            return None
        return self._items.get(str(param_id))

    def update_param_by_recv(self, param_data: Dict[str, Any]) -> None:
        param_id = param_data.get("id")
        if param_id is None:
            return
        self._items[str(param_id)] = dict(param_data)

    def save(self, monitor_param: MonitorParam) -> None:
        d = monitor_param.to_dict()
        param_id = d.get("id")
        if param_id is None:
            raise ValueError("MonitorParam.save 需要非空 id。")
        self._items[str(param_id)] = d

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._items)


class SensorParamConfigReceiver:
    """9879 同步消息接收器。"""

    def __init__(
        self,
        config: Optional[ImageUploadConfig] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        imei: Optional[str] = None,
        heartbeat_interval_sec: Optional[float] = None,
        sync_data_type: str = "SYNC_DATA",
    ) -> None:
        if config is None:
            self.config = ImageUploadConfig(
                host=host or "t4.tncet.com",
                config_port=int(port) if port is not None else 9879,
                imei=imei or "F44EB4DF6F99",
                heartbeat_interval_sec=(
                    float(heartbeat_interval_sec)
                    if heartbeat_interval_sec is not None
                    else 15.0
                ),
                auth_code="",
            )
        else:
            self.config = config
            if host is not None:
                self.config.host = host
            if port is not None:
                self.config.config_port = int(port)
            if imei is not None:
                self.config.imei = imei
            if heartbeat_interval_sec is not None:
                self.config.heartbeat_interval_sec = float(heartbeat_interval_sec)

        self.sync_data_type = sync_data_type

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # 调试缓存：保存最近一次收到和解析出的消息，便于联调排查
        self.latest_raw_message: Optional[bytes] = None
        self.latest_text_message: Optional[str] = None
        self.latest_json_message: Optional[Dict[str, Any]] = None
        self.latest_message_type: Optional[str] = None
        self.latest_extract_mode: Optional[str] = None
        self.latest_receive_status: Optional[str] = None
        self.latest_sent_control_packet: Optional[str] = None
        self.latest_heartbeat_time: Optional[float] = None

        # 对外暴露的解析结果
        self.camera_list: list = []
        self.params_by_id: Dict[str, Dict[str, Any]] = {}
        self.sensor_param_ids: Dict[str, str] = {}
        self.camera_ids_by_param_id: Dict[str, Any] = {}

        # 可选的手动配置，当前不自动推断左右相机
        if config is not None:
            self.left_sensor_param_id: Optional[str] = config.left_sensor_param_id
            self.right_sensor_param_id: Optional[str] = config.right_sensor_param_id
        else:
            self.left_sensor_param_id = None
            self.right_sensor_param_id = None

        # 内存 DAO：保留截图里的 upsert/update/save 处理语义
        self.cameraDao = _InMemoryCameraDao()
        self.monitorParamDao = _InMemoryMonitorParamDao()

    def connect(self) -> None:
        """主动通过 TCP 连接平台配置同步端口。"""
        self.close()
        sock: Optional[socket.socket] = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(float(self.config.connect_timeout))
            sock.connect((self.config.host, int(self.config.config_port)))
            sock.settimeout(1.0)
            self._sock = sock
            self.latest_receive_status = "connected"
            self.send_registration_packet()
        except OSError as exc:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
            raise ConnectionError(
                f"连接配置同步服务器失败 {self.config.host}:{self.config.config_port}: {exc}"
            ) from exc

    def _send_control_packet(self, packet: str) -> None:
        """发送平台要求的控制包，如注册包和心跳包。"""
        if self._sock is None:
            raise ConnectionError("TCP 未连接，无法发送控制包。")
        try:
            payload = packet.encode("utf-8")
            self._sock.sendall(get_length_prefix_bytes(payload))
            self.latest_sent_control_packet = packet
        except OSError as exc:
            self.latest_receive_status = f"send_error: {exc}"
            self.close()
            raise ConnectionError(f"发送控制包失败: {packet!r}: {exc}") from exc

    def build_registration_packet(self) -> str:
        """构造注册包：RG{imei}MONIX。"""
        imei = str(self.config.imei).strip()
        if not imei:
            raise ValueError("imei 不能为空，无法构造注册包。")
        return f"RG{imei}MONIX"

    def build_heartbeat_packet(self) -> str:
        """构造心跳包：HB{imei}MONIX。"""
        imei = str(self.config.imei).strip()
        if not imei:
            raise ValueError("imei 不能为空，无法构造心跳包。")
        return f"HB{imei}MONIX"

    def send_registration_packet(self) -> None:
        """连接建立后发送注册包。"""
        packet = self.build_registration_packet()
        self._send_control_packet(packet)
        # 注册成功后从当前时刻开始计算心跳间隔，避免刚注册就立刻发送心跳。
        self.latest_heartbeat_time = time.monotonic()

    def send_heartbeat_packet(self) -> None:
        """发送心跳包。"""
        packet = self.build_heartbeat_packet()
        self._send_control_packet(packet)
        self.latest_heartbeat_time = time.monotonic()

    def send_heartbeat_if_due(self) -> bool:
        """到达心跳间隔时发送心跳包，已发送返回 True。"""
        if self._sock is None:
            return False
        interval = float(self.config.heartbeat_interval_sec)
        if interval <= 0:
            return False
        now = time.monotonic()
        if self.latest_heartbeat_time is None or now - self.latest_heartbeat_time >= interval:
            self.send_heartbeat_packet()
            return True
        return False

    def receive_once(self, timeout_sec: Optional[float] = None) -> Optional[bytes]:
        """接收一次平台下发的数据；超时或无数据时返回 None。"""
        if self._sock is None:
            self.connect()

        assert self._sock is not None
        old_timeout = self._sock.gettimeout()
        try:
            if timeout_sec is not None:
                self._sock.settimeout(float(timeout_sec))
            data = self._sock.recv(8192)
            if not data:
                self.latest_receive_status = "closed"
                self.close()
                return None
            self.latest_receive_status = "received"
            return data
        except socket.timeout:
            self.latest_receive_status = "timeout"
            return None
        except OSError as exc:
            self.latest_receive_status = f"socket_error: {exc}"
            self.close()
            return None
        finally:
            if self._sock is not None:
                self._sock.settimeout(old_timeout)

    def close(self) -> None:
        """关闭 TCP 连接。"""
        if self._sock is None:
            return
        try:
            self._sock.close()
        finally:
            self._sock = None

    def start(self) -> None:
        """启动后台线程，持续接收同步消息。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.connect()
        self._thread = threading.Thread(target=self._run, name="SensorParamConfigReceiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止后台接收线程并关闭连接。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.close()

    def _run(self) -> None:
        """后台接收循环。"""
        while not self._stop_event.is_set():
            if self._sock is None:
                try:
                    self.connect()
                except Exception:
                    continue

            try:
                self.send_heartbeat_if_due()
                data = self._sock.recv(8192)  # type: ignore[union-attr]
                if not data:
                    self.close()
                    continue
                self.handle_recv_messages(data)
            except socket.timeout:
                continue
            except OSError:
                self.close()
                continue

    def _warn(self, msg: str) -> None:
        warnings.warn(f"[SensorParamConfigReceiver] {msg}", RuntimeWarning, stacklevel=2)

    def _extract_json_text(self, data: bytes) -> Tuple[Optional[str], str]:
        """从字节数据中提取 JSON 字符串。

        1) 优先按 UTF-8 文本解析
        2) 兼容 4 字节大端长度前缀 + JSON 的现场格式
        """
        #  UTF-8 文本路径
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="ignore")
            self._warn("原始数据 UTF-8 解码失败，已使用 errors='ignore' 兼容处理。")

        if text.strip():
            try:
                json.loads(text)
                return text, "raw"
            except Exception:
                pass

        # 4 字节大端长度前缀兼容路径
        if len(data) >= 5:
            declared_len = int.from_bytes(data[:4], byteorder="big", signed=False)
            available = len(data) - 4
            if 0 < declared_len <= available:
                payload = data[4 : 4 + declared_len]
                try:
                    payload_text = payload.decode("utf-8")
                except UnicodeDecodeError:
                    payload_text = payload.decode("utf-8", errors="ignore")
                    self._warn(
                        "长度前缀载荷 UTF-8 解码失败，已使用 errors='ignore' 兼容处理。"
                    )
                if payload_text.strip():
                    try:
                        json.loads(payload_text)
                        return payload_text, "len4be"
                    except Exception:
                        pass
        return None, "none"

    def handle_recv_messages(self, data: bytes) -> None:
        """处理平台下发的字节消息。

        处理流程与接口文档保持一致。
        """
        if data is None or not isinstance(data, (bytes, bytearray)):
            self._warn("handle_recv_messages 需要 bytes 类型输入。")
            return

        raw = bytes(data)
        with self._lock:
            self.latest_raw_message = raw

        message, extract_mode = self._extract_json_text(raw)
        if message is None:
            with self._lock:
                self.latest_text_message = raw.decode("utf-8", errors="ignore")
                self.latest_extract_mode = "none"
            self._warn(
                "未能从消息中提取有效 JSON（原始文本和 4 字节长度前缀兼容路径均失败）。"
            )
            return

        with self._lock:
            self.latest_text_message = message
            self.latest_extract_mode = extract_mode

        print(f"接收到来自服务器的消息: {message}")
        try:
            message_dict: Dict[str, Any] = json.loads(message)
            recv_data = RecvData.model_validate_json(message)

            with self._lock:
                self.latest_json_message = message_dict
                self.latest_message_type = str(message_dict.get("type"))

            if "type" in message_dict and message_dict["type"] == self.sync_data_type:
                self.sync_ipc_data(recv_data)
        except Exception as exc:
            self._warn(f"handle_recv_messages 解析失败: {exc}")

    def sync_ipc_data(self, origin_data: RecvData) -> None:
        """同步 IPC 数据。

        将 `origin_data.sync_detail` 中的平台同步数据写入内存缓存。
        """
        if origin_data is None or origin_data.sync_detail is None:
            print("origin_data或sync_detail属性不存在")
            return

        data = origin_data.sync_detail
        if not isinstance(data, dict):
            self._warn("sync_detail 不是字典。")
            return

        # 更新 camera
        camera_list = data.get("cameraList")
        if not isinstance(camera_list, list):
            self._warn("sync_detail.cameraList 缺失或不是列表。")
            camera_list = []
        for camera in camera_list:
            if isinstance(camera, dict):
                self.cameraDao.upsert_camera(camera)
            else:
                self._warn("camera 条目不是字典，已跳过。")

        # 更新 param
        param = data.get("param")
        if not isinstance(param, dict):
            self._warn("sync_detail.param 缺失或不是字典。")
            return

        camera_ids_by_param_id: Dict[str, Any] = {}
        for key, value in param.items():
            if not isinstance(value, dict):
                self._warn(f"param[{key!r}] 不是字典，已跳过。")
                continue

            # 此处 key 为 param 的 id，value 为 param 的数据
            param_data = {k: v for k, v in value.items() if k != "camera_ids"}
            old_main_param = self.monitorParamDao.find_by_id(value.get("id"))
            if old_main_param:
                # 存在则更新
                self.monitorParamDao.update_param_by_recv(param_data)
            else:
                # 不存在则插入
                self.monitorParamDao.save(MonitorParam.model_validate(param_data))

            pid = value.get("id", key)
            camera_ids_by_param_id[str(pid)] = value.get("camera_ids")

        # 从 DAO/cache 刷新对外暴露的解析状态
        with self._lock:
            self.camera_list = self.cameraDao.list_all()
            self.params_by_id = self.monitorParamDao.list_all()
            self.camera_ids_by_param_id = camera_ids_by_param_id
            self.sensor_param_ids = {}
            for pid, p in self.params_by_id.items():
                sid = p.get("sensor_param_id")
                if sid is not None and str(sid) != "":
                    self.sensor_param_ids[str(pid)] = str(sid)

    def get_sensor_ids(self) -> Dict[str, Optional[str]]:
        with self._lock:
            return {
                "left_sensor_param_id": self.left_sensor_param_id,
                "right_sensor_param_id": self.right_sensor_param_id,
            }

    def get_sync_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "latest_message_type": self.latest_message_type,
                "latest_extract_mode": self.latest_extract_mode,
                "camera_list": list(self.camera_list),
                "params_by_id": dict(self.params_by_id),
                "sensor_param_ids": dict(self.sensor_param_ids),
                "camera_ids_by_param_id": dict(self.camera_ids_by_param_id),
                "left_sensor_param_id": self.left_sensor_param_id,
                "right_sensor_param_id": self.right_sensor_param_id,
            }

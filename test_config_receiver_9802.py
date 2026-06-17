"""9879 自动同步联调脚本：接收并解析 SYNC_DATA 的 sensor_param_id。

运行：
    python test_config_receiver_9802.py
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

from image_uploader.config_receiver import SensorParamConfigReceiver


HOST = "t4.tncet.com"
PORT = 9879
IMEI = "F44EB4DF6F99"
HEARTBEAT_INTERVAL_SEC = 15
SYNC_DATA_TYPE = "SYNC_DATA"
RECEIVE_TIMEOUT_SEC = 60
OUTPUT_JSON = "sensor_param_ids_from_9879.json"


def _print_param_summary(
    params_by_id: Dict[str, Dict[str, Any]],
    camera_ids_by_param_id: Dict[str, Any],
) -> None:
    print("\n[params_by_id summary]")
    for param_id, data in params_by_id.items():
        name = data.get("name")
        ptype = data.get("type")
        sensor_param_id = data.get("sensor_param_id")
        camera_ids = camera_ids_by_param_id.get(param_id)
        print(
            f"- param_id={param_id}, name={name}, type={ptype}, "
            f"camera_ids={camera_ids}, sensor_param_id={sensor_param_id}"
        )


def main() -> None:
    receiver = SensorParamConfigReceiver(
        host=HOST,
        port=PORT,
        imei=IMEI,
        heartbeat_interval_sec=HEARTBEAT_INTERVAL_SEC,
        sync_data_type=SYNC_DATA_TYPE,
    )

    deadline = time.monotonic() + RECEIVE_TIMEOUT_SEC
    got_sensor_ids = False
    received_packet_count = 0

    try:
        receiver.connect()
        print(f"[info] connected to {HOST}:{PORT}, waiting up to {RECEIVE_TIMEOUT_SEC}s ...")
        print(
            f"[info] sent registration packet with length prefix: "
            f"{receiver.latest_sent_control_packet}"
        )
        print(
            f"[info] heartbeat packet with length prefix: "
            f"HB{IMEI}MONIX every {HEARTBEAT_INTERVAL_SEC}s"
        )

        while time.monotonic() < deadline:
            if receiver.send_heartbeat_if_due():
                print(f"[info] sent heartbeat packet: {receiver.latest_sent_control_packet}")

            data = receiver.receive_once(timeout_sec=1.0)
            if not data:
                if receiver.latest_receive_status == "closed":
                    print("[warn] server closed connection after registration/heartbeat.")
                    break
                if receiver.latest_receive_status and receiver.latest_receive_status.startswith("socket_error"):
                    print(f"[warn] socket error: {receiver.latest_receive_status}")
                    break
                continue

            received_packet_count += 1
            print(f"\n[recv packet #{received_packet_count}] bytes={len(data)}")
            print(f"[raw hex preview <=80 bytes] {data[:80].hex(' ')}")

            receiver.handle_recv_messages(data)

            preview = (receiver.latest_text_message or "")[:500]
            print("\n[recv text preview <=500 chars]")
            print(preview)

            print(f"[extract mode] {receiver.latest_extract_mode}")
            print(f"[message type] {receiver.latest_message_type}")
            print("[current sensor_param_ids]")
            print(receiver.sensor_param_ids)

            if receiver.sensor_param_ids:
                got_sensor_ids = True
                break

        if got_sensor_ids:
            print("\n[success] parsed sensor_param_ids:")
            for param_id, sensor_id in receiver.sensor_param_ids.items():
                print(f"- {param_id} -> {sensor_id}")

            _print_param_summary(
                receiver.params_by_id,
                getattr(receiver, "camera_ids_by_param_id", {}),
            )

            output = {
                "camera_list": receiver.camera_list,
                "params_by_id": receiver.params_by_id,
                "sensor_param_ids": receiver.sensor_param_ids,
                "camera_ids_by_param_id": getattr(receiver, "camera_ids_by_param_id", {}),
            }
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"\n[info] saved parsed result to: {OUTPUT_JSON}")
        else:
            if receiver.latest_receive_status == "closed":
                print(
                    f"已连接 {PORT} 并已发送注册包，但服务端随后关闭连接，"
                    "未下发任何同步数据。请确认 IMEI 是否正确、注册包是否应使用 4 字节大端长度前缀，"
                    "以及平台是否已绑定该采集箱。"
                )
            elif received_packet_count == 0:
                print(
                    f"已连接 {PORT}，但超时期间没有收到任何字节。"
                    "这更像是平台未下发数据、需要先发注册/鉴权消息，或需要平台后台触发同步。"
                )
            else:
                print(
                    f"已连接 {PORT}，共收到 {received_packet_count} 个 TCP 数据包，"
                    "但没有解析到 sensor_param_id。请把上面的 raw/text/extract mode/message type 发给平台核对协议。"
                )
            print(f"[last receive status] {receiver.latest_receive_status}")
    except Exception as exc:
        print(f"[error] {PORT} sync test failed: {type(exc).__name__}: {exc}")
    finally:
        receiver.close()


if __name__ == "__main__":
    main()

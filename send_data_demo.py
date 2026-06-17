"""Send demo displacement data to the platform protocol.

This demo script does not use cameras, calibration files, torch, SuperPoint, or
LightGlue. It only builds virtual displacement frames and prints or sends them
with the existing UD{imei}MONIX+json TCP format.
"""

from __future__ import annotations

import argparse
import configparser
import importlib.util
import json
import math
import os
import select
import socket
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = os.path.join("config", "socket.ini")
DEFAULT_COUNT = 0
DEFAULT_INTERVAL = 1.0
DEFAULT_DX_STEP = 0.2
DEFAULT_SENSOR_ID = "1"
DEFAULT_TIMEOUT = 5.0


def _fallback_length_prefix(payload: bytes) -> bytes:
    return struct.pack("!I", len(payload)) + payload


def _load_length_prefix_func():
    try:
        from utils.byte_utils import get_length_prefix_bytes

        return get_length_prefix_bytes
    except Exception:
        return _fallback_length_prefix


def _load_build_point_record():
    pack_path = Path(__file__).resolve().parent / "7_pack_data.py"
    if not pack_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("legacy_pack_data", pack_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "build_point_record", None)
    except Exception as exc:
        print(f"[warn] failed to import 7_pack_data.py::build_point_record, using fallback: {exc}")
        return None


GET_LENGTH_PREFIX_BYTES = _load_length_prefix_func()
BUILD_POINT_RECORD = _load_build_point_record()


def load_socket_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Optional[Any]]:
    """Load host, port and imei from config/socket.ini if available."""
    config_data: Dict[str, Optional[Any]] = {
        "host": None,
        "port": None,
        "imei": None,
        "heartbeat_time": 15,
        "config_path": config_path,
        "exists": os.path.exists(config_path),
    }
    if not config_data["exists"]:
        return config_data

    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    if parser.has_section("server"):
        if parser.has_option("server", "host"):
            config_data["host"] = parser.get("server", "host").strip() or None
        if parser.has_option("server", "port"):
            try:
                config_data["port"] = parser.getint("server", "port")
            except ValueError:
                config_data["port"] = None
        if parser.has_option("server", "imei"):
            config_data["imei"] = parser.get("server", "imei").strip() or None
    if parser.has_section("time") and parser.has_option("time", "heartbeat_time"):
        try:
            config_data["heartbeat_time"] = parser.getint("time", "heartbeat_time")
        except ValueError:
            config_data["heartbeat_time"] = 15
    return config_data


def _sensor_channel(sensor_id: Optional[str]) -> int:
    if sensor_id is None:
        return int(DEFAULT_SENSOR_ID)
    try:
        return int(sensor_id)
    except ValueError:
        return int(DEFAULT_SENSOR_ID)


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
) -> list[float]:
    """Build the existing 11-field displacement point record."""
    if BUILD_POINT_RECORD is not None:
        return BUILD_POINT_RECORD(
            left_id=left_id,
            right_id=right_id,
            channel=channel,
            px_dx=px_dx,
            px_dy=px_dy,
            px_dz=px_dz,
            disp_x=disp_x,
            disp_y=disp_y,
            disp_z=disp_z,
            match_conf=match_conf,
            track_conf=track_conf,
        )
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


def build_demo_displacement_payload(
    frame_id: int,
    timestamp: int,
    dx_mm: float,
    dy_mm: float,
    dz_mm: float,
    bad: bool = False,
    sensor_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one frame payload in the current displacement protocol shape."""
    channel = _sensor_channel(sensor_id)
    point = build_point_record(
        left_id=frame_id,
        right_id=frame_id,
        channel=channel,
        px_dx=0.0,
        px_dy=0.0,
        px_dz=0.0,
        disp_x=dx_mm,
        disp_y=dy_mm,
        disp_z=dz_mm,
        match_conf=0.0 if bad else 1.0,
        track_conf=0.0 if bad else 1.0,
    )
    return {
        "t": int(timestamp),
        "s": -1 if bad else 1,
        "p": [point],
    }


def build_send_message(payload: Dict[str, Any], imei: str, sensor_id: Optional[str] = None) -> bytes:
    """Build final UD{imei}MONIX+json message bytes."""
    channel_key = str(sensor_id or DEFAULT_SENSOR_ID)
    send_obj = {channel_key: payload}
    json_str = json.dumps(send_obj, ensure_ascii=False, separators=(",", ":"))
    return f"UD{imei}MONIX+{json_str}".encode("utf-8")


class PlatformTcpSender:
    """Persistent sender aligned with 8_tcp_client.py register/send flow."""

    def __init__(
        self,
        host: str,
        port: int,
        imei: str,
        timeout: float = DEFAULT_TIMEOUT,
        heartbeat_interval: float = 15.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.imei = imei
        self.timeout = float(timeout)
        self.heartbeat_interval = float(heartbeat_interval)
        self.last_heartbeat_monotonic = 0.0
        self.sock: Optional[socket.socket] = None

    def connect(self) -> bool:
        if self.sock is not None:
            return True
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            register_message = f"RG{self.imei}MONIX".encode("utf-8")
            sock.sendall(GET_LENGTH_PREFIX_BYTES(register_message))
            self.sock = sock
            self.last_heartbeat_monotonic = time.monotonic()
            print(f"[tcp] connected: {self.host}:{self.port}")
            print(f"[tcp] registered: RG{self.imei}MONIX")
            return True
        except Exception as exc:
            self.close()
            print(f"[tcp] connect/register failed: {exc}")
            return False

    def send(self, message: bytes) -> bool:
        if not self.connect() or self.sock is None:
            return False
        try:
            self.send_heartbeat_if_due()
            self.sock.sendall(GET_LENGTH_PREFIX_BYTES(message))
            print("[tcp] send success")
            self._print_response_if_available()
            return True
        except Exception as exc:
            self.close()
            print(f"[tcp] send failed: {exc}")
            return False

    def send_heartbeat_if_due(self) -> None:
        if self.sock is None or self.heartbeat_interval <= 0:
            return
        now = time.monotonic()
        if self.last_heartbeat_monotonic and now - self.last_heartbeat_monotonic < self.heartbeat_interval:
            return
        heartbeat_message = f"HB{self.imei}MONIX".encode("utf-8")
        self.sock.sendall(GET_LENGTH_PREFIX_BYTES(heartbeat_message))
        self.last_heartbeat_monotonic = now
        print(f"[tcp] heartbeat: HB{self.imei}MONIX")

    def _print_response_if_available(self) -> None:
        if self.sock is None:
            return
        try:
            readable, _, _ = select.select([self.sock], [], [], 0.2)
            if readable:
                response = self.sock.recv(4096)
                if response:
                    print(f"[tcp] server response bytes: {response!r}")
        except Exception as exc:
            print(f"[tcp] response check skipped: {exc}")

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self.sock.close()
        finally:
            self.sock = None


def send_payload_sync(
    message: bytes,
    host: str,
    port: int,
    imei: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """Compatibility helper: connect, register, send one payload, and close."""
    if not imei:
        print("[tcp] send failed: imei is required")
        return False
    sender = PlatformTcpSender(host=host, port=port, imei=imei, timeout=timeout)
    try:
        return sender.send(message)
    finally:
        sender.close()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _missing_required(config: Dict[str, Optional[Any]]) -> list[str]:
    missing = []
    for key in ("host", "port", "imei"):
        if config.get(key) in (None, ""):
            missing.append(key)
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate demo displacement data and dry-run or send it to the platform."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="number of frames to generate; 0 means run continuously")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="seconds between frames")
    parser.add_argument("--send", action="store_true", help="compatibility flag; real sending is now the default")
    parser.add_argument("--dry-run", action="store_true", help="print payloads only and do not connect")
    parser.add_argument("--dx-step", type=float, default=DEFAULT_DX_STEP, help="phase step for the 5-8 mm demo curve")
    parser.add_argument("--imei", type=str, default=None, help="override imei from config/socket.ini")
    parser.add_argument("--host", type=str, default=None, help="override host from config/socket.ini")
    parser.add_argument("--port", type=int, default=None, help="override port from config/socket.ini")
    parser.add_argument("--sensor-id", type=str, default=DEFAULT_SENSOR_ID, help="channel/sensor id")
    parser.add_argument("--bad", action="store_true", help="generate bad status frames")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="socket config path")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="TCP connect/send timeout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_socket_config(args.config)

    print(f"[config] reading: {args.config}")
    if not config["exists"]:
        print(f"[config] file not found: {args.config}")

    if args.host:
        config["host"] = args.host
    if args.port is not None:
        config["port"] = args.port
    if args.imei:
        config["imei"] = args.imei

    dry_run = bool(args.dry_run)

    missing = _missing_required(config)
    if not dry_run and missing:
        print(f"[config] missing required values for real send: {missing}; fallback to dry-run")
        dry_run = True

    if dry_run:
        print("[mode] dry-run: print payloads only, no network send")
    else:
        print(f"[mode] real send: host={config['host']} port={config['port']} imei={config['imei']}")

    total = int(args.count)
    run_forever = total <= 0
    sender: Optional[PlatformTcpSender] = None
    if not dry_run:
        sender = PlatformTcpSender(
            host=str(config["host"]),
            port=int(config["port"]),
            imei=str(config["imei"]),
            timeout=float(args.timeout),
            heartbeat_interval=float(config.get("heartbeat_time") or 15),
        )

    frame_id = 0
    try:
        while run_forever or frame_id < total:
            frame_id += 1
            timestamp = _now_ms()
            phase = float(args.dx_step) * frame_id
            dx_mm = 6.5 + 1.5 * math.sin(phase)
            dy_mm = 6.5 + 1.5 * math.sin(phase + 2.0 * math.pi / 3.0)
            dz_mm = 6.5 + 1.5 * math.sin(phase + 4.0 * math.pi / 3.0)
            payload = build_demo_displacement_payload(
                frame_id=frame_id,
                timestamp=timestamp,
                dx_mm=dx_mm,
                dy_mm=dy_mm,
                dz_mm=dz_mm,
                bad=bool(args.bad),
                sensor_id=args.sensor_id,
            )
            message = build_send_message(
                payload=payload,
                imei=str(config.get("imei") or args.imei or "<IMEI>"),
                sensor_id=args.sensor_id,
            )

            print(
                f"[demo] frame_id={frame_id}, "
                f"dx_mm={dx_mm:.6f}, dy_mm={dy_mm:.6f}, dz_mm={dz_mm:.6f}, bad={bool(args.bad)}"
            )
            print(f"[payload] {json.dumps(payload, ensure_ascii=False)}")
            print(f"[message] {message.decode('utf-8', errors='replace')}")

            if not dry_run and sender is not None:
                total_label = "continuous" if run_forever else str(total)
                print(f"[tcp] sending {frame_id}/{total_label} to {config['host']}:{config['port']} imei={config['imei']}")
                ok = sender.send(message)
                if not ok:
                    print("[tcp] this frame failed; will retry on next frame")

            if float(args.interval) > 0:
                time.sleep(float(args.interval))
    except KeyboardInterrupt:
        print("\n[done] stopped by user")
    finally:
        if sender is not None:
            sender.close()
    print("[done] data sending demo finished")


if __name__ == "__main__":
    main()

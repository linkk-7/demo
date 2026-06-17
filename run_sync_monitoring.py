"""Synchronous online monitoring pipeline.

Flow:
capture one stereo pair -> rectify -> ROI candidate init -> sync track ->
3D displacement -> sync payload send/log.
"""

from __future__ import annotations

import configparser
import importlib.util
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from get_dis_imgs import continuous_capture_online
from online_tracking import (
    CandidatePoint,
    SyncPointTracker,
    compute_xyz_from_stereo_points,
    compute_xyz_from_stereo_points_temp_calibration,
    create_roi_config_from_selection,
    load_roi_config,
    match_initial_candidates,
    save_roi_config,
    validate_roi_config,
)
from online_tracking.sync_point_tracker import _load_lightglue_symbols
from stereo_rectify import StereoRectifier
from utils.byte_utils import get_length_prefix_bytes


# Calibration parameters. Change this if a different calibration set is used.
CALIBRATION_FOLDER = r"new_data5\cab"

SAVE_FOLDER_BASE = os.path.join("input", "cab_online")
ROI_CONFIG_PATH = os.path.join("runtime_state", "roi_config.json")
RESET_ROI = True
MAX_FRAMES: Optional[int] = 500
TRACK_TOP_K = 8
TEMP_CALIBRATION_MODE = True
TEMP_BASELINE_MM = 70.0
STEREO_DISPARITY_MIN = 1.0
STEREO_DISPARITY_MAX = 1200.0
STEREO_Z_MIN_MM = 50.0
STEREO_Z_MAX_MM = 50000.0
DISPLACEMENT_OUTLIER_FLOOR_MM = 5.0

# Camera brightness for the original SDK images. The capture layer default is
# 8000 us; this is intentionally brighter for field monitoring/visualization.
CAMERA_EXPOSURE_TIME_US = 16000.0
CAMERA_TARGET_ACQUISITION_FPS = 30.0
LOG_CAMERA_NODES = True

# Real monitoring defaults to TCP sending. Use True only for local debugging.
SEND_DRY_RUN = False
SEND_INTERVAL_FRAMES = 1
DISPLACEMENT_UNIT = "mm"
VISUALIZE_TRACKING = True
VISUALIZATION_WINDOW_NAME = "sync monitoring tracking"
VISUALIZATION_MAX_WIDTH = 1600
VISUALIZATION_TEXT_SCALE = 0.75
VISUALIZATION_TEXT_THICKNESS = 2
VISUALIZATION_TEXT_LINE_HEIGHT = 28

SOCKET_CONFIG_PATH = os.path.join("config", "socket.ini")
SEND_CHANNEL_ID = "1"
TCP_CONNECT_TIMEOUT_SEC = 5.0
TCP_SEND_TIMEOUT_SEC = 5.0


class StopMonitoring(Exception):
    """Raised inside on_frame to stop the capture loop cleanly."""


@dataclass
class TrackedStereoPoint:
    candidate: CandidatePoint
    left_tracker: SyncPointTracker
    right_tracker: SyncPointTracker
    reference_xyz: np.ndarray
    initial_left_xy: Tuple[float, float]
    initial_right_xy: Tuple[float, float]


def _load_build_point_record():
    pack_path = Path(__file__).resolve().parent / "7_pack_data.py"
    spec = importlib.util.spec_from_file_location("legacy_pack_data", pack_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "build_point_record", None)


_BUILD_POINT_RECORD = _load_build_point_record()


def _build_point_record(
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
) -> List[float]:
    if _BUILD_POINT_RECORD is not None:
        return _BUILD_POINT_RECORD(
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


class DisplacementTcpSender:
    """Small synchronous sender for the existing UD{imei}MONIX+json protocol."""

    def __init__(self, config_path: str = SOCKET_CONFIG_PATH) -> None:
        self.config_path = config_path
        self.sock: Optional[socket.socket] = None
        self.host: Optional[str] = None
        self.port: Optional[int] = None
        self.imei: Optional[str] = None
        self.ready = False
        self._load_config()

    def _load_config(self) -> None:
        if not os.path.exists(self.config_path):
            print(f"[send] missing config: {self.config_path}; fallback to dry_run")
            return
        config = configparser.ConfigParser()
        config.read(self.config_path, encoding="utf-8")
        missing = []
        for key in ("host", "port", "imei"):
            if not config.has_option("server", key) or not config.get("server", key).strip():
                missing.append(f"server.{key}")
        if missing:
            print(f"[send] missing config keys: {missing}; fallback to dry_run")
            return

        self.host = config.get("server", "host")
        self.port = config.getint("server", "port")
        self.imei = config.get("server", "imei")
        self.ready = True

    def connect(self) -> None:
        if self.sock is not None:
            return
        if not self.ready or self.host is None or self.port is None or self.imei is None:
            raise RuntimeError("TCP sender config is incomplete.")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TCP_CONNECT_TIMEOUT_SEC)
        sock.connect((self.host, self.port))
        sock.settimeout(TCP_SEND_TIMEOUT_SEC)
        register_msg = f"RG{self.imei}MONIX".encode("utf-8")
        sock.sendall(get_length_prefix_bytes(register_msg))
        self.sock = sock
        print(f"[send] connected {self.host}:{self.port}, registered imei={self.imei}")

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self.sock.close()
        finally:
            self.sock = None

    def send_frame(self, frame: Dict[str, Any]) -> str:
        if not self.ready or self.imei is None:
            return "dry_run_config_incomplete"
        send_obj = {SEND_CHANNEL_ID: frame}
        json_str = json.dumps(send_obj, ensure_ascii=False, separators=(",", ":"))
        msg = f"UD{self.imei}MONIX+{json_str}".encode("utf-8")
        try:
            self.connect()
            assert self.sock is not None
            self.sock.sendall(get_length_prefix_bytes(msg))
            return "sent"
        except Exception as exc:
            self.close()
            print(f"[send] TCP send failed; fallback payload kept as dry_run: {exc}")
            return "dry_run_send_failed"


_TCP_SENDER: Optional[DisplacementTcpSender] = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _format_vec(value: Optional[Sequence[float]]) -> str:
    if value is None:
        return "None"
    arr = np.asarray(value, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{x:.3f}" for x in arr) + "]"


def _stable_left_id(point_id: int) -> int:
    return int(point_id) * 2 + 1


def _stable_right_id(point_id: int) -> int:
    return int(point_id) * 2 + 2


def _stereo_geometry_error(left_xy: Sequence[float], right_xy: Sequence[float]) -> Optional[str]:
    raw_disparity = float(right_xy[0]) - float(left_xy[0])
    checked_disparity = abs(raw_disparity) if TEMP_CALIBRATION_MODE else raw_disparity
    if checked_disparity < STEREO_DISPARITY_MIN or checked_disparity > STEREO_DISPARITY_MAX:
        return (
            f"disparity={raw_disparity:.3f}, checked={checked_disparity:.3f} outside "
            f"[{STEREO_DISPARITY_MIN:.3f}, {STEREO_DISPARITY_MAX:.3f}]"
        )
    return None


def _xyz_error(xyz: Sequence[float]) -> Optional[str]:
    arr = np.asarray(xyz, dtype=float).reshape(-1)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return f"invalid xyz={arr}"
    z = float(arr[2])
    if z < STEREO_Z_MIN_MM or z > STEREO_Z_MAX_MM:
        return f"z={z:.3f} outside [{STEREO_Z_MIN_MM:.3f}, {STEREO_Z_MAX_MM:.3f}]"
    return None


def _compute_xyz_for_monitoring(
    left_xy: Sequence[float],
    right_xy: Sequence[float],
    q_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    if TEMP_CALIBRATION_MODE:
        return compute_xyz_from_stereo_points_temp_calibration(
            left_xy=left_xy,
            right_xy=right_xy,
            calibration_folder=CALIBRATION_FOLDER,
            baseline_mm=TEMP_BASELINE_MM,
        )
    return compute_xyz_from_stereo_points(
        left_xy,
        right_xy,
        calibration_folder=CALIBRATION_FOLDER,
        q_matrix=q_matrix,
    )


def _robust_average_displacement(
    displacements: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, List[int]]:
    arr = np.asarray(displacements, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected Nx3 displacements, got shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("no valid displacements.")
    if arr.shape[0] == 1:
        return arr[0], [0]

    center = np.median(arr, axis=0)
    distances = np.linalg.norm(arr - center, axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    threshold = max(DISPLACEMENT_OUTLIER_FLOOR_MM, median_distance + 3.0 * max(mad, 1e-6))
    inlier_indices = np.flatnonzero(distances <= threshold).tolist()
    if not inlier_indices:
        inlier_indices = [int(np.argmin(distances))]
    return np.mean(arr[inlier_indices], axis=0), inlier_indices


def _to_bgr_for_display(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    return image.copy()


def _draw_text_lines(canvas: np.ndarray, lines: List[str]) -> None:
    line_h = int(VISUALIZATION_TEXT_LINE_HEIGHT)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(VISUALIZATION_TEXT_SCALE)
    thickness = int(VISUALIZATION_TEXT_THICKNESS)
    text_sizes = [
        cv2.getTextSize(line, font, scale, thickness)[0]
        for line in lines
    ]
    if text_sizes:
        box_w = max(size[0] for size in text_sizes) + 28
        box_h = line_h * len(lines) + 18
        box_x = 8
        box_y = max(8, canvas.shape[0] - box_h - 8)
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
        x = box_x + 10
        y = box_y + line_h
    else:
        x = 18
        y = max(42, canvas.shape[0] - 32)

    for line in lines:
        cv2.putText(
            canvas,
            line,
            (x, y),
            font,
            scale,
            (0, 0, 0),
            thickness + 4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_h


def show_tracking_visualization(
    left_image: np.ndarray,
    right_image: np.ndarray,
    frame_id: int,
    point_logs: List[Dict[str, Any]],
    displacement: Optional[Sequence[float]],
    send_status: str,
    tracking_state: str,
) -> str:
    """Show current tracking overlay.

    Returns:
        "ok": displayed normally
        "quit": user pressed q/Esc
        "error": OpenCV display failed
    """
    left_vis = _to_bgr_for_display(left_image)
    right_vis = _to_bgr_for_display(right_image)
    if left_vis.shape[0] != right_vis.shape[0]:
        target_h = left_vis.shape[0]
        target_w = max(1, int(right_vis.shape[1] * target_h / right_vis.shape[0]))
        right_vis = cv2.resize(right_vis, (target_w, target_h), interpolation=cv2.INTER_AREA)

    canvas = np.hstack([left_vis, right_vis])
    offset_x = left_vis.shape[1]

    colors = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 120, 0),
        (255, 0, 255),
        (80, 180, 255),
    ]
    for idx, log in enumerate(point_logs):
        left_xy = log.get("left_xy")
        right_xy = log.get("right_xy")
        point_id = log.get("point_id", idx)
        if left_xy is None or right_xy is None:
            continue
        color = colors[idx % len(colors)]
        lx, ly = int(round(left_xy[0])), int(round(left_xy[1]))
        rx, ry = int(round(right_xy[0] + offset_x)), int(round(right_xy[1]))
        cv2.circle(canvas, (lx, ly), 8, color, 2)
        cv2.circle(canvas, (rx, ry), 8, color, 2)
        cv2.line(canvas, (lx, ly), (rx, ry), color, 1)
        cv2.putText(
            canvas,
            str(point_id),
            (lx + 8, ly - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    if canvas.shape[1] > VISUALIZATION_MAX_WIDTH:
        scale = VISUALIZATION_MAX_WIDTH / float(canvas.shape[1])
        target_size = (
            int(canvas.shape[1] * scale),
            int(canvas.shape[0] * scale),
        )
        canvas = cv2.resize(canvas, target_size, interpolation=cv2.INTER_AREA)

    _draw_text_lines(
        canvas,
        [
            f"frame_id={frame_id}  state={tracking_state}  tracked={len(point_logs)}",
            f"displacement(mm)={_format_vec(displacement)}",
            f"send_status={send_status}  q/Esc=quit",
        ],
    )

    try:
        cv2.imshow(VISUALIZATION_WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
    except cv2.error as exc:
        print(f"[visualize][warn] OpenCV display failed; visualization disabled: {exc}")
        return "error"

    if key in (ord("q"), 27):
        return "quit"
    return "ok"


def _frame_timestamp_ms(frame_packet) -> int:
    values = []
    for attr in ("left_host_timestamp", "right_host_timestamp"):
        host_ts = getattr(frame_packet, attr, None)
        if host_ts is not None:
            values.append(float(host_ts))
    if values:
        ts = sum(values) / len(values)
        return int(ts if ts > 10_000_000_000 else ts * 1000)
    return _now_ms()


def _build_payload_frame(
    displacement: Optional[Sequence[float]],
    frame_id: int,
    timestamp: int,
    bad: bool = False,
    point_records: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    if point_records is None:
        disp = np.zeros(3, dtype=float) if displacement is None else np.asarray(displacement, dtype=float)
        point_records = [
            _build_point_record(
                left_id=0,
                right_id=0,
                channel=int(SEND_CHANNEL_ID),
                px_dx=0.0,
                px_dy=0.0,
                px_dz=0.0,
                disp_x=float(disp[0]),
                disp_y=float(disp[1]),
                disp_z=float(disp[2]),
                match_conf=0.0 if bad else 1.0,
                track_conf=0.0 if bad else 1.0,
            )
        ]

    return {"t": int(timestamp), "s": -1 if bad else 1, "p": point_records}


def send_displacement_sync(
    displacement,
    frame_id,
    timestamp,
    bad: bool = False,
    dry_run: bool = True,
    point_records: Optional[List[List[float]]] = None,
) -> str:
    """Synchronously send or print one displacement payload."""
    global _TCP_SENDER

    payload_frame = _build_payload_frame(
        displacement=displacement,
        frame_id=int(frame_id),
        timestamp=int(timestamp),
        bad=bad,
        point_records=point_records,
    )
    log_payload = {
        "frame_id": int(frame_id),
        "unit": DISPLACEMENT_UNIT,
        **payload_frame,
    }

    if dry_run:
        print(f"[send][dry_run] {json.dumps(log_payload, ensure_ascii=False)}")
        return "dry_run"

    if _TCP_SENDER is None:
        _TCP_SENDER = DisplacementTcpSender(SOCKET_CONFIG_PATH)
    status = _TCP_SENDER.send_frame(payload_frame)
    if status != "sent":
        print(f"[send][{status}] {json.dumps(log_payload, ensure_ascii=False)}")
    return status


def _load_or_create_roi_config(left_rect_image, right_rect_image, frame_id, calibration_tag):
    height, width = left_rect_image.shape[:2]
    if not RESET_ROI and os.path.exists(ROI_CONFIG_PATH):
        roi_config = load_roi_config(ROI_CONFIG_PATH)
        validation = validate_roi_config(
            roi_config,
            image_width=width,
            image_height=height,
            calibration_tag=calibration_tag,
            calibration_folder=CALIBRATION_FOLDER,
        )
        if validation.ok:
            print(f"[roi] loaded config: {ROI_CONFIG_PATH}")
            print(f"[roi] roi_left={roi_config.roi_left.to_xywh()}")
            return roi_config

        print("[roi] existing config invalid; selecting again")
        for message in validation.messages:
            print(f"[roi] - {message}")

    roi_config = create_roi_config_from_selection(
        left_rect_image=left_rect_image,
        right_rect_image=right_rect_image,
        calibration_tag=calibration_tag,
        calibration_folder=CALIBRATION_FOLDER,
        reference_frame_id=frame_id,
        save_reference_images_flag=True,
    )
    save_roi_config(roi_config, ROI_CONFIG_PATH)
    print(f"[roi] saved config: {ROI_CONFIG_PATH}")
    print(f"[roi] roi_left={roi_config.roi_left.to_xywh()}")
    return roi_config


def _make_shared_feature_models(roi_config):
    LightGlue, SuperPoint, _rbd = _load_lightglue_symbols()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_keypoints = int(roi_config.init_policy.max_keypoints)
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher, device


def _candidate_summary(candidate: CandidatePoint) -> Dict[str, Any]:
    data = candidate.as_dict()
    keep = [
        "point_id",
        "left_idx",
        "right_idx",
        "left_xy",
        "right_xy",
        "score",
        "dy",
        "disparity",
        "in_roi",
        "status",
        "initial_quality_score",
        "message",
    ]
    return {key: data.get(key) for key in keep}


def main() -> None:
    if TEMP_CALIBRATION_MODE:
        print(
            "[calibration][temp] enabled: "
            f"baseline={TEMP_BASELINE_MM:.1f}mm, old intrinsics, absolute disparity. "
            "For demo only, not final measurement calibration."
        )
    rectifier = StereoRectifier(calibration_folder=CALIBRATION_FOLDER)
    state: Dict[str, Any] = {
        "initialized": False,
        "tracked_points": [],
        "frame_count": 0,
        "extractor": None,
        "matcher": None,
        "device": None,
        "visualization_enabled": bool(VISUALIZE_TRACKING),
    }

    def initialize_from_frame(frame_packet, left_rect_image, right_rect_image, rectified_packet) -> bool:
        roi_config = _load_or_create_roi_config(
            left_rect_image=left_rect_image,
            right_rect_image=right_rect_image,
            frame_id=frame_packet.frame_id,
            calibration_tag=rectified_packet.calibration_tag,
        )
        extractor, matcher, device = _make_shared_feature_models(roi_config)
        state["extractor"] = extractor
        state["matcher"] = matcher
        state["device"] = device

        init_result = match_initial_candidates(
            left_img=left_rect_image,
            right_img=right_rect_image,
            roi_config=roi_config,
            extractor=extractor,
            matcher=matcher,
            device=device,
        )
        summary = init_result.as_dict_summary()
        print(
            f"[init] frame_id={frame_packet.frame_id}, "
            f"matches_count={summary['matches_count']}, "
            f"candidates_count={summary['candidates_count']}"
        )
        if init_result.message:
            print(f"[init] message={init_result.message}")
        for candidate in init_result.candidates[:5]:
            print(f"[init][candidate] {_candidate_summary(candidate)}")

        selected = init_result.candidates
        target_track_count = max(1, int(TRACK_TOP_K))
        if not selected:
            print("[init] no valid candidates in ROI; waiting for next frame")
            return False

        tracked_points: List[TrackedStereoPoint] = []
        for candidate in selected:
            if len(tracked_points) >= target_track_count:
                break
            try:
                reference_xyz = _compute_xyz_for_monitoring(
                    candidate.left_xy,
                    candidate.right_xy,
                    q_matrix=rectified_packet.q_matrix,
                )
                xyz_error = _xyz_error(reference_xyz)
                if xyz_error is not None:
                    raise ValueError(f"candidate reference xyz rejected: {xyz_error}")
                left_tracker = SyncPointTracker(
                    extractor=extractor,
                    matcher=matcher,
                    device=device,
                    input_color=roi_config.init_policy.input_color,
                    max_num_keypoints=roi_config.init_policy.max_keypoints,
                )
                right_tracker = SyncPointTracker(
                    extractor=extractor,
                    matcher=matcher,
                    device=device,
                    input_color=roi_config.init_policy.input_color,
                    max_num_keypoints=roi_config.init_policy.max_keypoints,
                )
                left_tracker.init(left_rect_image, candidate.left_idx, frame_id=frame_packet.frame_id)
                right_tracker.init(right_rect_image, candidate.right_idx, frame_id=frame_packet.frame_id)
                candidate.used_for_tracking = True
                candidate.used_for_displacement = True
                tracked_points.append(
                    TrackedStereoPoint(
                        candidate=candidate,
                        left_tracker=left_tracker,
                        right_tracker=right_tracker,
                        reference_xyz=reference_xyz,
                        initial_left_xy=candidate.left_xy,
                        initial_right_xy=candidate.right_xy,
                    )
                )
                print(
                    f"[init][selected] point_id={candidate.point_id}, "
                    f"stable_left_id={_stable_left_id(candidate.point_id)}, "
                    f"stable_right_id={_stable_right_id(candidate.point_id)}, "
                    f"left_idx={candidate.left_idx}, right_idx={candidate.right_idx}, "
                    f"reference_xyz={_format_vec(reference_xyz)}"
                )
            except Exception as exc:
                print(f"[init][warn] skip candidate point_id={candidate.point_id}: {exc}")

        if not tracked_points:
            print("[init] no candidate produced valid reference xyz; waiting for next frame")
            return False

        state["tracked_points"] = tracked_points
        state["initialized"] = True
        zero_disp = np.zeros(3, dtype=float)
        timestamp = _frame_timestamp_ms(frame_packet)
        send_status = send_displacement_sync(
            displacement=zero_disp,
            frame_id=frame_packet.frame_id,
            timestamp=timestamp,
            bad=False,
            dry_run=SEND_DRY_RUN,
        )
        print(
            f"[frame] frame_id={frame_packet.frame_id}, initialized=True, "
            f"tracked_points_count={len(tracked_points)}, displacement={_format_vec(zero_disp)}, "
            f"send_status={send_status}"
        )
        if state["visualization_enabled"]:
            init_logs = [
                {
                    "point_id": tracked.candidate.point_id,
                    "left_xy": tracked.initial_left_xy,
                    "right_xy": tracked.initial_right_xy,
                }
                for tracked in tracked_points
            ]
            visual_status = show_tracking_visualization(
                left_image=left_rect_image,
                right_image=right_rect_image,
                frame_id=frame_packet.frame_id,
                point_logs=init_logs,
                displacement=zero_disp,
                send_status=send_status,
                tracking_state="initialized",
            )
            if visual_status == "quit":
                raise StopMonitoring("stopped from visualization window")
            if visual_status == "error":
                state["visualization_enabled"] = False
        return True

    def on_frame(frame_packet) -> None:
        state["frame_count"] += 1
        rectified_packet = rectifier.rectify_frame_packet(
            frame_packet=frame_packet,
            save_images=False,
            return_q_matrix=True,
        )
        left_rect_image = rectified_packet.left_rect_image
        right_rect_image = rectified_packet.right_rect_image
        if left_rect_image is None or right_rect_image is None:
            raise RuntimeError("RectifiedPacket does not contain rectified images.")

        if not state["initialized"]:
            initialize_from_frame(
                frame_packet=frame_packet,
                left_rect_image=left_rect_image,
                right_rect_image=right_rect_image,
                rectified_packet=rectified_packet,
            )
            return

        timestamp = _frame_timestamp_ms(frame_packet)
        valid_displacements = []
        point_records = []
        valid_logs = []

        for tracked in state["tracked_points"]:
            candidate = tracked.candidate
            left_result = tracked.left_tracker.update(left_rect_image, frame_id=frame_packet.frame_id)
            right_result = tracked.right_tracker.update(right_rect_image, frame_id=frame_packet.frame_id)

            if not left_result.is_valid or not right_result.is_valid:
                print(
                    f"[track][warn] frame_id={frame_packet.frame_id}, point_id={candidate.point_id}, "
                    f"left_status={left_result.status}, right_status={right_result.status}"
                )
                continue

            try:
                geometry_error = _stereo_geometry_error(left_result.xy, right_result.xy)
                if geometry_error is not None:
                    raise ValueError(f"tracked stereo geometry rejected: {geometry_error}")
                xyz = _compute_xyz_for_monitoring(
                    left_result.xy,
                    right_result.xy,
                    q_matrix=rectified_packet.q_matrix,
                )
                xyz_error = _xyz_error(xyz)
                if xyz_error is not None:
                    raise ValueError(f"tracked xyz rejected: {xyz_error}")
                displacement = xyz - tracked.reference_xyz
            except Exception as exc:
                print(f"[disp][warn] frame_id={frame_packet.frame_id}, point_id={candidate.point_id}: {exc}")
                continue

            valid_displacements.append(displacement)
            track_conf_values = [
                value
                for value in (left_result.confidence, right_result.confidence)
                if value is not None
            ]
            track_conf = min(track_conf_values) if track_conf_values else 1.0
            px_dx = float(left_result.xy[0] - tracked.initial_left_xy[0])
            px_dy = float(left_result.xy[1] - tracked.initial_left_xy[1])
            point_records.append(
                _build_point_record(
                    left_id=_stable_left_id(candidate.point_id),
                    right_id=_stable_right_id(candidate.point_id),
                    channel=int(SEND_CHANNEL_ID),
                    px_dx=px_dx,
                    px_dy=px_dy,
                    px_dz=0.0,
                    disp_x=float(displacement[0]),
                    disp_y=float(displacement[1]),
                    disp_z=float(displacement[2]),
                    match_conf=float(candidate.score if candidate.score is not None else 1.0),
                    track_conf=float(track_conf),
                )
            )
            valid_logs.append(
                {
                    "point_id": candidate.point_id,
                    "left_xy": left_result.xy,
                    "right_xy": right_result.xy,
                    "left_idx": left_result.matched_index,
                    "right_idx": right_result.matched_index,
                    "left_method": left_result.method,
                    "right_method": right_result.method,
                    "xyz": xyz,
                    "displacement": displacement,
                }
            )

        if not valid_displacements:
            send_status = "skipped_no_valid_points"
            if state["frame_count"] % max(1, int(SEND_INTERVAL_FRAMES)) == 0:
                send_status = send_displacement_sync(
                    displacement=None,
                    frame_id=frame_packet.frame_id,
                    timestamp=timestamp,
                    bad=True,
                    dry_run=SEND_DRY_RUN,
                    point_records=[],
                )
            print(
                f"[frame] frame_id={frame_packet.frame_id}, tracking=True, "
                f"tracked_points_count=0, left_xy=None, right_xy=None, xyz=None, "
                f"displacement=None, send_status={send_status}"
            )
            if state["visualization_enabled"]:
                visual_status = show_tracking_visualization(
                    left_image=left_rect_image,
                    right_image=right_rect_image,
                    frame_id=frame_packet.frame_id,
                    point_logs=[],
                    displacement=None,
                    send_status=send_status,
                    tracking_state="lost",
                )
                if visual_status == "quit":
                    raise StopMonitoring("stopped from visualization window")
                if visual_status == "error":
                    state["visualization_enabled"] = False
            return

        final_displacement, inlier_indices = _robust_average_displacement(valid_displacements)
        if len(inlier_indices) < len(valid_displacements):
            print(
                f"[disp][filter] frame_id={frame_packet.frame_id}, "
                f"kept={len(inlier_indices)}, dropped={len(valid_displacements) - len(inlier_indices)}"
            )
        valid_displacements = [valid_displacements[idx] for idx in inlier_indices]
        point_records = [point_records[idx] for idx in inlier_indices]
        valid_logs = [valid_logs[idx] for idx in inlier_indices]

        send_status = "not_due"
        if state["frame_count"] % max(1, int(SEND_INTERVAL_FRAMES)) == 0:
            send_status = send_displacement_sync(
                displacement=final_displacement,
                frame_id=frame_packet.frame_id,
                timestamp=timestamp,
                bad=False,
                dry_run=SEND_DRY_RUN,
                point_records=point_records,
            )

        first_log = valid_logs[0]
        stable_ids = [
            (
                _stable_left_id(int(log["point_id"])),
                _stable_right_id(int(log["point_id"])),
            )
            for log in valid_logs
        ]
        print(
            f"[frame] frame_id={frame_packet.frame_id}, tracking=True, "
            f"tracked_points_count={len(valid_displacements)}, "
            f"stable_ids={stable_ids}, "
            f"left_xy={_format_vec(first_log['left_xy'])}, "
            f"right_xy={_format_vec(first_log['right_xy'])}, "
            f"current_idx=({first_log['left_idx']},{first_log['right_idx']}), "
            f"method=({first_log['left_method']},{first_log['right_method']}), "
            f"xyz={_format_vec(first_log['xyz'])}, "
            f"displacement={_format_vec(final_displacement)}, "
            f"send_status={send_status}"
        )
        if state["visualization_enabled"]:
            visual_status = show_tracking_visualization(
                left_image=left_rect_image,
                right_image=right_rect_image,
                frame_id=frame_packet.frame_id,
                point_logs=valid_logs,
                displacement=final_displacement,
                send_status=send_status,
                tracking_state="tracking",
            )
            if visual_status == "quit":
                raise StopMonitoring("stopped from visualization window")
            if visual_status == "error":
                state["visualization_enabled"] = False

    try:
        continuous_capture_online(
            save_folder_base=SAVE_FOLDER_BASE,
            on_frame=on_frame,
            max_frames=MAX_FRAMES,
            log_every_n=30,
            exposure_time_us=CAMERA_EXPOSURE_TIME_US,
            target_acquisition_fps=CAMERA_TARGET_ACQUISITION_FPS,
            log_camera_nodes=LOG_CAMERA_NODES,
        )
    except StopMonitoring as exc:
        print(f"[done] {exc}")
    finally:
        if _TCP_SENDER is not None:
            _TCP_SENDER.close()
        if VISUALIZE_TRACKING:
            try:
                cv2.destroyWindow(VISUALIZATION_WINDOW_NAME)
            except cv2.error:
                pass


if __name__ == "__main__":
    main()

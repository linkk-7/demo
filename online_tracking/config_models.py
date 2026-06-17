"""Configuration models for online ROI-based tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class ValidationResult:
    """Result of validating a runtime ROI configuration."""

    ok: bool
    messages: List[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.ok = False
        self.messages.append(message)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "messages": list(self.messages),
        }


@dataclass
class ROIRegion:
    """Rectangle ROI in xywh format."""

    x: int
    y: int
    w: int
    h: int

    def to_xywh(self) -> List[int]:
        return [int(self.x), int(self.y), int(self.w), int(self.h)]

    def to_xyxy(self) -> List[int]:
        return [int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h)]

    def contains_point(self, x: float, y: float, edge_margin: int = 0) -> bool:
        margin = int(edge_margin)
        return (
            self.x + margin <= x < self.x + self.w - margin
            and self.y + margin <= y < self.y + self.h - margin
        )

    def clip_to_image(self, width: int, height: int) -> "ROIRegion":
        image_w = int(width)
        image_h = int(height)
        x1 = max(0, min(int(self.x), image_w))
        y1 = max(0, min(int(self.y), image_h))
        x2 = max(0, min(int(self.x + self.w), image_w))
        y2 = max(0, min(int(self.y + self.h), image_h))
        return ROIRegion(x=x1, y=y1, w=max(0, x2 - x1), h=max(0, y2 - y1))

    def is_valid(self) -> bool:
        return int(self.w) > 0 and int(self.h) > 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROIRegion":
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            w=int(data["w"]),
            h=int(data["h"]),
        )


@dataclass
class ImageInfo:
    """Image coordinate system and size for ROI coordinates."""

    width: int
    height: int
    coordinate_space: str = "rectified"
    source: str = "left_rect_image"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "coordinate_space": self.coordinate_space,
            "source": self.source,
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageInfo":
        return cls(
            coordinate_space=data.get("coordinate_space", "rectified"),
            source=data.get("source", "left_rect_image"),
            width=int(data["width"]),
            height=int(data["height"]),
        )


@dataclass
class CameraInfo:
    """Camera identity metadata used to detect hardware changes."""

    left_camera_id: Optional[str] = None
    right_camera_id: Optional[str] = None
    left_serial: Optional[str] = None
    right_serial: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "left_camera_id": self.left_camera_id,
            "right_camera_id": self.right_camera_id,
            "left_serial": self.left_serial,
            "right_serial": self.right_serial,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CameraInfo":
        data = data or {}
        return cls(
            left_camera_id=data.get("left_camera_id"),
            right_camera_id=data.get("right_camera_id"),
            left_serial=data.get("left_serial"),
            right_serial=data.get("right_serial"),
        )


@dataclass
class InitPolicy:
    """Feature matching filters used when initializing candidates inside ROI."""

    input_color: str = "bgr"
    max_keypoints: int = 1000
    min_confidence: float = 0.2
    epipolar_y_threshold: float = 3.0
    disparity_min: float = 1.0
    disparity_max: float = 1200.0
    edge_margin: int = 8
    max_candidates: int = 1000
    target_track_points: int = 8

    def as_dict(self) -> Dict[str, Any]:
        return {
            "input_color": self.input_color,
            "max_keypoints": int(self.max_keypoints),
            "min_confidence": float(self.min_confidence),
            "epipolar_y_threshold": float(self.epipolar_y_threshold),
            "disparity_min": float(self.disparity_min),
            "disparity_max": float(self.disparity_max),
            "edge_margin": int(self.edge_margin),
            "max_candidates": int(self.max_candidates),
            "target_track_points": int(self.target_track_points),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "InitPolicy":
        data = data or {}
        return cls(
            input_color=data.get("input_color", "bgr"),
            max_keypoints=int(data.get("max_keypoints", 1000)),
            min_confidence=float(data.get("min_confidence", 0.2)),
            epipolar_y_threshold=float(data.get("epipolar_y_threshold", 3.0)),
            disparity_min=float(data.get("disparity_min", 1.0)),
            disparity_max=float(data.get("disparity_max", 1200.0)),
            edge_margin=int(data.get("edge_margin", 8)),
            max_candidates=int(data.get("max_candidates", 1000)),
            target_track_points=int(data.get("target_track_points", 8)),
        )


@dataclass
class ReferenceInfo:
    """Reference images captured when the ROI was selected."""

    reference_frame_id: Optional[int] = None
    save_reference_images: bool = True
    left_rect_image_path: Optional[str] = None
    right_rect_image_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reference_frame_id": self.reference_frame_id,
            "save_reference_images": bool(self.save_reference_images),
            "left_rect_image_path": self.left_rect_image_path,
            "right_rect_image_path": self.right_rect_image_path,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ReferenceInfo":
        data = data or {}
        return cls(
            reference_frame_id=data.get("reference_frame_id"),
            save_reference_images=bool(data.get("save_reference_images", True)),
            left_rect_image_path=data.get("left_rect_image_path"),
            right_rect_image_path=data.get("right_rect_image_path"),
        )


@dataclass
class ROIConfig:
    """Runtime ROI configuration for online tracking."""

    image: ImageInfo
    roi_left: ROIRegion
    version: int = 1
    created_time: str = field(default_factory=_now_iso)
    updated_time: str = field(default_factory=_now_iso)
    calibration_tag: Optional[str] = None
    calibration_folder: Optional[str] = None
    roi_right: Optional[ROIRegion] = None
    camera: CameraInfo = field(default_factory=CameraInfo)
    init_policy: InitPolicy = field(default_factory=InitPolicy)
    reference: ReferenceInfo = field(default_factory=ReferenceInfo)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "created_time": self.created_time,
            "updated_time": self.updated_time,
            "calibration_tag": self.calibration_tag,
            "calibration_folder": self.calibration_folder,
            "image": self.image.as_dict(),
            "roi_left": self.roi_left.as_dict(),
            "roi_right": self.roi_right.as_dict() if self.roi_right is not None else None,
            "camera": self.camera.as_dict(),
            "init_policy": self.init_policy.as_dict(),
            "reference": self.reference.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROIConfig":
        roi_right_data = data.get("roi_right")
        return cls(
            version=int(data.get("version", 1)),
            created_time=data.get("created_time", _now_iso()),
            updated_time=data.get("updated_time", _now_iso()),
            calibration_tag=data.get("calibration_tag"),
            calibration_folder=data.get("calibration_folder"),
            image=ImageInfo.from_dict(data["image"]),
            roi_left=ROIRegion.from_dict(data["roi_left"]),
            roi_right=ROIRegion.from_dict(roi_right_data) if roi_right_data else None,
            camera=CameraInfo.from_dict(data.get("camera")),
            init_policy=InitPolicy.from_dict(data.get("init_policy")),
            reference=ReferenceInfo.from_dict(data.get("reference")),
        )

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ROIConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def validate_for_image(
        self,
        width: int,
        height: int,
        calibration_tag: Optional[str] = None,
        calibration_folder: Optional[str] = None,
    ) -> ValidationResult:
        result = ValidationResult(ok=True)
        image_width = int(width)
        image_height = int(height)

        if int(self.image.width) != image_width or int(self.image.height) != image_height:
            result.add(
                "image size mismatch: "
                f"config=({self.image.width}, {self.image.height}), "
                f"current=({image_width}, {image_height})"
            )

        if calibration_tag is not None and self.calibration_tag is not None:
            if calibration_tag != self.calibration_tag:
                result.add(
                    "calibration_tag mismatch: "
                    f"config={self.calibration_tag}, current={calibration_tag}"
                )

        if calibration_folder is not None and self.calibration_folder is not None:
            if calibration_folder != self.calibration_folder:
                result.add(
                    "calibration_folder mismatch: "
                    f"config={self.calibration_folder}, current={calibration_folder}"
                )

        self._validate_roi("roi_left", self.roi_left, image_width, image_height, result)
        if self.roi_right is not None:
            self._validate_roi("roi_right", self.roi_right, image_width, image_height, result)

        return result

    @staticmethod
    def _validate_roi(
        name: str,
        roi: ROIRegion,
        width: int,
        height: int,
        result: ValidationResult,
    ) -> None:
        if not roi.is_valid():
            result.add(f"{name} is invalid: w and h must be positive")
            return
        x1, y1, x2, y2 = roi.to_xyxy()
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            result.add(
                f"{name} is outside image bounds: "
                f"roi={roi.to_xywh()}, image=({width}, {height})"
            )

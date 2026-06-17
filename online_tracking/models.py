"""Data models for the online tracking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CandidatePoint:
    """Candidate stereo point selected inside the ROI for later tracking."""

    point_id: int
    left_idx: int
    right_idx: int
    left_xy: Tuple[float, float]
    right_xy: Tuple[float, float]
    score: Optional[float] = None
    dy: Optional[float] = None
    disparity: Optional[float] = None
    in_roi: bool = True
    initial_quality_score: Optional[float] = None
    later_quality_score: Optional[float] = None
    status: str = "candidate"
    used_for_tracking: bool = False
    used_for_displacement: bool = False
    message: str = ""

    @property
    def is_valid_for_stereo(self) -> bool:
        if self.left_xy is None or self.right_xy is None:
            return False
        if self.dy is not None and self.dy < 0:
            return False
        if self.disparity is not None and self.disparity <= 0:
            return False
        return True

    def compute_basic_quality(
        self,
        epipolar_y_threshold: float = 3.0,
        disparity_min: float = 1.0,
        disparity_max: float = 300.0,
    ) -> float:
        """Compute a simple initial quality score in [0, 1]."""
        quality = 1.0
        if self.score is not None:
            quality *= max(0.0, min(1.0, float(self.score)))

        if self.dy is not None:
            threshold = max(float(epipolar_y_threshold), 1e-6)
            quality *= max(0.0, 1.0 - min(float(self.dy), threshold) / threshold)

        if self.disparity is not None:
            disparity = float(self.disparity)
            if disparity < disparity_min or disparity > disparity_max:
                quality = 0.0

        if not self.in_roi:
            quality = 0.0

        self.initial_quality_score = float(quality)
        return self.initial_quality_score

    def as_dict(self) -> Dict[str, Any]:
        return {
            "point_id": int(self.point_id),
            "left_idx": int(self.left_idx),
            "right_idx": int(self.right_idx),
            "left_xy": list(self.left_xy),
            "right_xy": list(self.right_xy),
            "score": self.score,
            "dy": self.dy,
            "disparity": self.disparity,
            "in_roi": self.in_roi,
            "initial_quality_score": self.initial_quality_score,
            "later_quality_score": self.later_quality_score,
            "status": self.status,
            "used_for_tracking": self.used_for_tracking,
            "used_for_displacement": self.used_for_displacement,
            "message": self.message,
            "is_valid_for_stereo": self.is_valid_for_stereo,
        }


@dataclass
class TrackingResult:
    """Single-camera tracking result for one frame."""

    frame_id: Optional[int]
    xy: Optional[Tuple[float, float]]
    status: str
    method: str
    confidence: Optional[float] = None
    matched_index: Optional[int] = None
    occluded: bool = False
    message: str = ""
    history_length: int = 0
    prediction_reliability: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        return self.xy is not None

    @property
    def is_lost(self) -> bool:
        return self.status == "lost"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "xy": list(self.xy) if self.xy is not None else None,
            "status": self.status,
            "method": self.method,
            "confidence": self.confidence,
            "matched_index": self.matched_index,
            "occluded": self.occluded,
            "message": self.message,
            "history_length": self.history_length,
            "prediction_reliability": self.prediction_reliability,
            "is_valid": self.is_valid,
            "is_lost": self.is_lost,
        }


@dataclass
class OcclusionSegment:
    """A consecutive segment tracked by fallback or marked as occluded."""

    start_frame_id: Optional[int]
    end_frame_id: Optional[int] = None
    frame_ids: List[int] = field(default_factory=list)
    reason: str = ""

    def close(self, end_frame_id: Optional[int]) -> None:
        self.end_frame_id = end_frame_id

    def as_dict(self) -> Dict[str, Any]:
        return {
            "start_frame_id": self.start_frame_id,
            "end_frame_id": self.end_frame_id,
            "frame_ids": list(self.frame_ids),
            "reason": self.reason,
        }


@dataclass
class InitialStereoMatchResult:
    """Initial left-right keypoint matching result for the first stereo frame."""

    kpts0: object
    kpts1: object
    matches: object
    scores: Optional[object]
    selected_left_idx: Optional[int] = None
    selected_right_idx: Optional[int] = None
    selected_score: Optional[float] = None
    selected_left_xy: Optional[Tuple[float, float]] = None
    selected_right_xy: Optional[Tuple[float, float]] = None
    message: str = ""
    candidates: List[CandidatePoint] = field(default_factory=list)

    @property
    def has_selection(self) -> bool:
        return self.selected_left_idx is not None and self.selected_right_idx is not None

    @staticmethod
    def _safe_len(value: object) -> Optional[int]:
        try:
            return len(value)  # type: ignore[arg-type]
        except TypeError:
            return None

    def as_dict_summary(self) -> Dict[str, Any]:
        return {
            "matches_count": self._safe_len(self.matches),
            "candidates_count": len(self.candidates),
            "selected_left_idx": self.selected_left_idx,
            "selected_right_idx": self.selected_right_idx,
            "selected_score": self.selected_score,
            "has_selection": self.has_selection,
            "message": self.message,
        }


@dataclass
class OnlineStereoTrackingResult:
    """Synchronized left/right tracking result for one stereo frame."""

    frame_id: Optional[int]
    left: TrackingResult
    right: TrackingResult
    stereo_status: str = ""
    message: str = ""

    @classmethod
    def from_left_right(
        cls,
        frame_id: Optional[int],
        left: TrackingResult,
        right: TrackingResult,
        message: str = "",
    ) -> "OnlineStereoTrackingResult":
        if left.is_valid and right.is_valid:
            stereo_status = "both_valid"
        elif left.is_lost and right.is_lost:
            stereo_status = "both_lost"
        elif left.is_lost:
            stereo_status = "left_lost"
        elif right.is_lost:
            stereo_status = "right_lost"
        else:
            stereo_status = "partial"
        return cls(
            frame_id=frame_id,
            left=left,
            right=right,
            stereo_status=stereo_status,
            message=message,
        )

    @property
    def both_valid(self) -> bool:
        return self.left.is_valid and self.right.is_valid

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
            "stereo_status": self.stereo_status,
            "message": self.message,
            "both_valid": self.both_valid,
        }

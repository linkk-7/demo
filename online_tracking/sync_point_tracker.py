"""Minimal synchronous feature-point tracker for the online pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch

from .image_utils import load_image_array
from .models import TrackingResult


_EXTRACT_CACHE = {}
_EXTRACT_CACHE_ORDER = []
_MATCH_CACHE = {}
_MATCH_CACHE_ORDER = []
_MAX_EXTRACT_CACHE_ITEMS = 8
_MAX_MATCH_CACHE_ITEMS = 8


def _remember_cache(cache: dict, order: list, key: tuple, value: Any, max_items: int) -> Any:
    if key in cache:
        return cache[key]
    cache[key] = value
    order.append(key)
    while len(order) > max_items:
        old_key = order.pop(0)
        cache.pop(old_key, None)
    return value


def _load_lightglue_symbols():
    try:
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
    except ModuleNotFoundError:
        project_root = Path(__file__).resolve().parents[1]
        lightglue_path = project_root / "LightGlue-main"
        if lightglue_path.exists():
            sys.path.append(str(lightglue_path))
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
    return LightGlue, SuperPoint, rbd


def _to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _to_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def _point_xy(kpts: Any, idx: int) -> Tuple[float, float]:
    point = kpts[idx]
    return _to_float(point[0]), _to_float(point[1])


def _keypoint_count(feats: Any) -> int:
    try:
        keypoints = feats["keypoints"]
        if torch.is_tensor(keypoints):
            return int(keypoints.shape[0])
        return len(keypoints)
    except Exception:
        return 0


def _image_cache_key(extractor: Any, image: np.ndarray, input_color: str) -> tuple:
    data_ptr = int(image.__array_interface__["data"][0])
    return (
        id(extractor),
        id(image),
        data_ptr,
        tuple(image.shape),
        str(image.dtype),
        str(input_color),
    )


def _match_cache_key(matcher: Any, feats0_batch: Any, feats1_batch: Any) -> tuple:
    return (id(matcher), id(feats0_batch), id(feats1_batch))


def _as_point_tuple(point: Any) -> Tuple[float, float]:
    return float(point[0]), float(point[1])


def _crop_template(image: np.ndarray, xy: Tuple[float, float], half_size: int):
    h, w = image.shape[:2]
    x = int(round(xy[0]))
    y = int(round(xy[1]))
    x1 = max(0, x - half_size)
    y1 = max(0, y - half_size)
    x2 = min(w - 1, x + half_size)
    y2 = min(h - 1, y + half_size)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return image[y1 : y2 + 1, x1 : x2 + 1], (x1, y1, x2, y2)


def _predict_position(history_xy: list[Tuple[float, float]], method: str, frames_ahead: int) -> Optional[Tuple[float, float]]:
    if not history_xy:
        return None
    trajectory = np.asarray(history_xy, dtype=float)
    n = len(trajectory)
    if n < 2:
        return _as_point_tuple(trajectory[-1])

    if method == "acceleration" and n >= 3:
        velocity = trajectory[-1] - trajectory[-2]
        acceleration = velocity - (trajectory[-2] - trajectory[-3])
        predicted = trajectory[-1] + velocity * frames_ahead + 0.5 * acceleration * frames_ahead**2
    elif method == "average":
        window = min(5, n - 1)
        velocities = [trajectory[-i] - trajectory[-i - 1] for i in range(1, window + 1)]
        predicted = trajectory[-1] + np.mean(velocities, axis=0) * frames_ahead
    else:
        velocity = trajectory[-1] - trajectory[-2]
        predicted = trajectory[-1] + velocity * frames_ahead
    return _as_point_tuple(predicted)


class SyncPointTracker:
    """Track one SuperPoint keypoint from frame to frame with LightGlue."""

    def __init__(
        self,
        extractor=None,
        matcher=None,
        device=None,
        input_color: str = "bgr",
        max_num_keypoints: int = 1000,
        template_size: int = 41,
        search_margin: int = 100,
        template_match_threshold: float = 0.7,
        max_prediction_frames: int = 30,
        prediction_method: str = "velocity",
    ) -> None:
        LightGlue, SuperPoint, rbd = _load_lightglue_symbols()
        torch.set_grad_enabled(False)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_color = input_color
        self.extractor = extractor or SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = matcher or LightGlue(features="superpoint").eval().to(self.device)
        self._rbd = rbd
        self.template_size = int(template_size)
        self.search_margin = int(search_margin)
        self.template_match_threshold = float(template_match_threshold)
        self.max_prediction_frames = int(max_prediction_frames)
        self.prediction_method = prediction_method

        self.last_feats_batch = None
        self.last_feats = None
        self.last_image = None
        self.last_idx: Optional[int] = None
        self.last_xy: Optional[Tuple[float, float]] = None
        self.last_success_feats_batch = None
        self.last_success_feats = None
        self.last_success_image = None
        self.last_success_idx: Optional[int] = None
        self.last_success_xy: Optional[Tuple[float, float]] = None
        self.prediction_frames = 0
        self.history_xy: list[Tuple[float, float]] = []

    def _extract(self, image: np.ndarray):
        cache_key = _image_cache_key(self.extractor, image, self.input_color)
        cached = _EXTRACT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        tensor = load_image_array(image, input_color=self.input_color)
        with torch.no_grad():
            feats_batch = self.extractor.extract(tensor.to(self.device))
        feats = self._rbd(feats_batch)
        return _remember_cache(
            _EXTRACT_CACHE,
            _EXTRACT_CACHE_ORDER,
            cache_key,
            (feats_batch, feats),
            _MAX_EXTRACT_CACHE_ITEMS,
        )

    def init(self, image: np.ndarray, init_idx: int, frame_id=None) -> TrackingResult:
        feats_batch, feats = self._extract(image)
        keypoints = feats["keypoints"]
        idx = int(init_idx)
        keypoint_count = _keypoint_count(feats)
        if keypoint_count == 0:
            raise IndexError("no keypoints found while initializing tracker.")
        if idx < 0 or idx >= keypoint_count:
            raise IndexError(f"init_idx={idx} out of range for {keypoint_count} keypoints.")

        xy = _point_xy(keypoints, idx)
        self.last_feats_batch = feats_batch
        self.last_feats = feats
        self.last_image = image
        self.last_idx = idx
        self.last_xy = xy
        self.last_success_feats_batch = feats_batch
        self.last_success_feats = feats
        self.last_success_image = image
        self.last_success_idx = idx
        self.last_success_xy = xy
        self.prediction_frames = 0
        self.history_xy = [xy]

        return TrackingResult(
            frame_id=frame_id,
            xy=xy,
            status="init",
            method="feature",
            matched_index=idx,
            history_length=len(self.history_xy),
            message="tracker initialized",
        )

    def update(self, image: np.ndarray, frame_id=None) -> TrackingResult:
        if self.last_success_feats_batch is None or self.last_success_idx is None:
            raise RuntimeError("SyncPointTracker.update called before init.")

        current_feats_batch, current_feats = self._extract(image)
        current_count = _keypoint_count(current_feats)
        if current_count == 0:
            fallback = self._update_by_template_or_prediction(image, frame_id, "no keypoints found in current frame")
            if fallback is not None:
                return fallback
            return self._lost_result(frame_id, "no keypoints found in current frame")

        try:
            match_key = _match_cache_key(
                self.matcher,
                self.last_success_feats_batch,
                current_feats_batch,
            )
            cached_matches = _MATCH_CACHE.get(match_key)
            if cached_matches is None:
                with torch.no_grad():
                    matches_batch = self.matcher(
                        {"image0": self.last_success_feats_batch, "image1": current_feats_batch}
                    )
                matches01 = self._rbd(matches_batch)
                cached_matches = _remember_cache(
                    _MATCH_CACHE,
                    _MATCH_CACHE_ORDER,
                    match_key,
                    matches01,
                    _MAX_MATCH_CACHE_ITEMS,
                )
            else:
                matches01 = cached_matches
        except RuntimeError as exc:
            fallback = self._update_by_template_or_prediction(image, frame_id, f"LightGlue matching failed: {exc}")
            if fallback is not None:
                return fallback
            return self._lost_result(frame_id, f"LightGlue matching failed: {exc}")

        matches = matches01.get("matches", None)
        scores = matches01.get("scores", None)

        matched_row = None
        if matches is not None:
            for row in range(len(matches)):
                if _to_int(matches[row][0]) == self.last_success_idx:
                    matched_row = row
                    break

        if matched_row is None:
            fallback = self._update_by_template_or_prediction(
                image,
                frame_id,
                f"last_success_idx={self.last_success_idx} not matched",
            )
            if fallback is not None:
                return fallback
            return self._lost_result(frame_id, f"last_success_idx={self.last_success_idx} not matched")

        new_idx = _to_int(matches[matched_row][1])
        xy = _point_xy(current_feats["keypoints"], new_idx)
        confidence = None
        if scores is not None and matched_row < len(scores):
            confidence = _to_float(scores[matched_row])

        self.last_feats_batch = current_feats_batch
        self.last_feats = current_feats
        self.last_image = image
        self.last_idx = new_idx
        self.last_xy = xy
        self.last_success_feats_batch = current_feats_batch
        self.last_success_feats = current_feats
        self.last_success_image = image
        self.last_success_idx = new_idx
        self.last_success_xy = xy
        self.prediction_frames = 0
        self.history_xy.append(xy)

        return TrackingResult(
            frame_id=frame_id,
            xy=xy,
            status="tracked",
            method="feature",
            confidence=confidence,
            matched_index=new_idx,
            history_length=len(self.history_xy),
            message="tracked by LightGlue",
        )

    def _lost_result(self, frame_id, message: str) -> TrackingResult:
        return TrackingResult(
            frame_id=frame_id,
            xy=None,
            status="lost",
            method="none",
            history_length=len(self.history_xy),
            message=message,
        )

    def _update_by_template_or_prediction(
        self,
        image: np.ndarray,
        frame_id,
        reason: str,
    ) -> Optional[TrackingResult]:
        template_result = self._update_by_template(image, frame_id, reason)
        if template_result is not None:
            return template_result
        return self._update_by_prediction(image, frame_id, reason)

    def _update_by_template(self, image: np.ndarray, frame_id, reason: str) -> Optional[TrackingResult]:
        if self.last_success_image is None or self.last_success_xy is None:
            return None

        half_size = max(1, self.template_size // 2)
        template, _box = _crop_template(self.last_success_image, self.last_success_xy, half_size)
        if template is None or template.size == 0:
            return None

        h, w = image.shape[:2]
        x = int(round(self.last_success_xy[0]))
        y = int(round(self.last_success_xy[1]))
        search_x1 = max(0, x - self.search_margin)
        search_y1 = max(0, y - self.search_margin)
        search_x2 = min(w - 1, x + self.search_margin)
        search_y2 = min(h - 1, y + self.search_margin)
        search_area = image[search_y1 : search_y2 + 1, search_x1 : search_x2 + 1]
        if (
            search_area.size == 0
            or search_area.shape[0] <= template.shape[0]
            or search_area.shape[1] <= template.shape[1]
        ):
            return None

        result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < self.template_match_threshold:
            return None

        match_x = search_x1 + max_loc[0] + template.shape[1] // 2
        match_y = search_y1 + max_loc[1] + template.shape[0] // 2
        xy = (float(match_x), float(match_y))

        self.last_image = image
        self.last_xy = xy
        self.last_success_image = image
        self.last_success_xy = xy
        self.prediction_frames = 0
        self.history_xy.append(xy)

        return TrackingResult(
            frame_id=frame_id,
            xy=xy,
            status="tracked",
            method="template",
            confidence=float(max_val),
            matched_index=self.last_success_idx,
            occluded=True,
            history_length=len(self.history_xy),
            prediction_reliability=1.0,
            message=f"template fallback after {reason}",
        )

    def _update_by_prediction(self, image: np.ndarray, frame_id, reason: str) -> Optional[TrackingResult]:
        if len(self.history_xy) < 2:
            return None
        self.prediction_frames += 1
        predicted = _predict_position(
            self.history_xy,
            method=self.prediction_method,
            frames_ahead=self.prediction_frames,
        )
        if predicted is None:
            return None

        h, w = image.shape[:2]
        x = min(max(float(predicted[0]), 0.0), float(w - 1))
        y = min(max(float(predicted[1]), 0.0), float(h - 1))
        xy = (x, y)
        reliability = max(0.1, 1.0 - (self.prediction_frames / max(1, self.max_prediction_frames)))

        self.last_image = image
        self.last_xy = xy
        self.last_success_xy = xy
        self.history_xy.append(xy)

        return TrackingResult(
            frame_id=frame_id,
            xy=xy,
            status="tracked",
            method="prediction",
            confidence=float(reliability),
            matched_index=self.last_success_idx,
            occluded=True,
            history_length=len(self.history_xy),
            prediction_reliability=float(reliability),
            message=f"prediction fallback after {reason}",
        )

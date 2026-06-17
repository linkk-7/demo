"""Initial ROI candidate matching for online stereo tracking."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch

from .config_models import ROIConfig
from .image_utils import load_image_array
from .models import CandidatePoint, InitialStereoMatchResult


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


def _to_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def _point_xy(kpts: Any, idx: int) -> Tuple[float, float]:
    point = kpts[idx]
    return _to_float(point[0]), _to_float(point[1])


def _score_at(scores: Any, row: int) -> Optional[float]:
    if scores is None:
        return None
    try:
        if len(scores) <= row:
            return None
    except TypeError:
        return None
    return _to_float(scores[row])


def _candidate_from_match(
    row: int,
    matches: Any,
    scores: Any,
    kpts0: Any,
    kpts1: Any,
    roi_config: ROIConfig,
    edge_margin: int = 0,
    status: str = "candidate",
    message: str = "",
) -> CandidatePoint:
    left_idx = _to_int(matches[row][0])
    right_idx = _to_int(matches[row][1])
    left_xy = _point_xy(kpts0, left_idx)
    right_xy = _point_xy(kpts1, right_idx)
    score = _score_at(scores, row)
    dy = abs(left_xy[1] - right_xy[1])
    # The current legacy 3D formula uses right_x - left_x as positive disparity.
    disparity = right_xy[0] - left_xy[0]
    in_roi = roi_config.roi_left.contains_point(
        left_xy[0],
        left_xy[1],
        edge_margin=edge_margin,
    )
    return CandidatePoint(
        point_id=0,
        left_idx=left_idx,
        right_idx=right_idx,
        left_xy=left_xy,
        right_xy=right_xy,
        score=score,
        dy=dy,
        disparity=disparity,
        in_roi=in_roi,
        initial_quality_score=float(score if score is not None else 0.0),
        status=status,
        used_for_tracking=False,
        used_for_displacement=False,
        message=message,
    )


def _safe_keypoint_count(feats: Any) -> int:
    try:
        keypoints = feats["keypoints"]
        if torch.is_tensor(keypoints):
            if keypoints.ndim == 3:
                return int(keypoints.shape[1])
            if keypoints.ndim == 2:
                return int(keypoints.shape[0])
        return len(keypoints)
    except Exception:
        return 0


def _save_candidate_visualization(
    left_img: np.ndarray,
    right_img: np.ndarray,
    candidates: list[CandidatePoint],
    save_path: str,
) -> None:
    left_vis = left_img.copy()
    right_vis = right_img.copy()
    if left_vis.ndim == 2:
        left_vis = cv2.cvtColor(left_vis, cv2.COLOR_GRAY2BGR)
    if right_vis.ndim == 2:
        right_vis = cv2.cvtColor(right_vis, cv2.COLOR_GRAY2BGR)

    if left_vis.shape[0] != right_vis.shape[0]:
        target_h = left_vis.shape[0]
        target_w = int(right_vis.shape[1] * target_h / right_vis.shape[0])
        right_vis = cv2.resize(right_vis, (target_w, target_h))

    canvas = np.hstack([left_vis, right_vis])
    offset_x = left_vis.shape[1]
    for candidate in candidates[:100]:
        lx, ly = candidate.left_xy
        rx, ry = candidate.right_xy
        left_pt = (int(round(lx)), int(round(ly)))
        right_pt = (int(round(rx + offset_x)), int(round(ry)))
        cv2.circle(canvas, left_pt, 3, (0, 255, 0), -1)
        cv2.circle(canvas, right_pt, 3, (0, 255, 0), -1)
        cv2.line(canvas, left_pt, right_pt, (0, 200, 255), 1)

    cv2.imwrite(save_path, canvas)


def match_initial_candidates(
    left_img,
    right_img,
    roi_config: ROIConfig,
    extractor=None,
    matcher=None,
    device=None,
    input_color: Optional[str] = None,
    visualize: bool = False,
    save_path: Optional[str] = None,
) -> InitialStereoMatchResult:
    """Run the legacy first-frame SuperPoint+LightGlue match, limited by ROI.

    This intentionally mirrors ``get_init_id.get_init_id``: extract SuperPoint
    features on the left/right images, run LightGlue once, then use the raw
    match rows. The only online addition is selecting matches whose left point
    falls inside the user-drawn ROI.
    """
    if left_img is None:
        raise ValueError("left_img is None.")
    if right_img is None:
        raise ValueError("right_img is None.")
    if not isinstance(left_img, np.ndarray):
        raise ValueError("left_img must be a numpy.ndarray.")
    if not isinstance(right_img, np.ndarray):
        raise ValueError("right_img must be a numpy.ndarray.")
    if roi_config is None:
        raise ValueError("roi_config is required.")

    LightGlue, SuperPoint, rbd = _load_lightglue_symbols()
    torch.set_grad_enabled(False)

    if device is None:
        active_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        active_device = torch.device(device)

    policy = roi_config.init_policy
    active_input_color = input_color or policy.input_color
    active_extractor = extractor or SuperPoint(max_num_keypoints=policy.max_keypoints).eval().to(active_device)
    active_matcher = matcher or LightGlue(features="superpoint").eval().to(active_device)

    image0 = load_image_array(left_img, input_color=active_input_color)
    image1 = load_image_array(right_img, input_color=active_input_color)

    feats0_batch = active_extractor.extract(image0.to(active_device))
    feats1_batch = active_extractor.extract(image1.to(active_device))
    feats0 = rbd(feats0_batch)
    feats1 = rbd(feats1_batch)
    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]

    count0 = _safe_keypoint_count(feats0_batch)
    count1 = _safe_keypoint_count(feats1_batch)
    if count0 == 0 or count1 == 0:
        return InitialStereoMatchResult(
            kpts0=kpts0,
            kpts1=kpts1,
            matches=torch.empty((0, 2), dtype=torch.long, device=active_device),
            scores=None,
            message=(
                "not enough keypoints for LightGlue matching: "
                f"left={count0}, right={count1}"
            ),
            candidates=[],
        )

    try:
        matches01_batch = active_matcher({"image0": feats0_batch, "image1": feats1_batch})
        matches01 = rbd(matches01_batch)
    except RuntimeError as exc:
        return InitialStereoMatchResult(
            kpts0=kpts0,
            kpts1=kpts1,
            matches=torch.empty((0, 2), dtype=torch.long, device=active_device),
            scores=None,
            message=f"LightGlue matching failed: {exc}",
            candidates=[],
        )

    matches = matches01["matches"]
    scores = matches01.get("scores", None)

    candidates: list[CandidatePoint] = []
    if matches is not None:
        for row in range(len(matches)):
            candidate = _candidate_from_match(
                row=row,
                matches=matches,
                scores=scores,
                kpts0=kpts0,
                kpts1=kpts1,
                roi_config=roi_config,
                edge_margin=0,
                status="legacy_roi_match",
                message="legacy SuperPoint+LightGlue match selected by left ROI only",
            )
            left_xy = candidate.left_xy

            if not roi_config.roi_left.contains_point(
                left_xy[0],
                left_xy[1],
                edge_margin=0,
            ):
                continue

            if candidate.score is not None and candidate.score < policy.min_confidence:
                continue

            candidate.in_roi = True
            candidate.initial_quality_score = float(candidate.score if candidate.score is not None else 0.0)
            candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            candidate.initial_quality_score
            if candidate.initial_quality_score is not None
            else float("-inf")
        ),
        reverse=True,
    )
    candidates = candidates[: policy.max_candidates]
    for point_id, candidate in enumerate(candidates):
        candidate.point_id = point_id

    selected = candidates[0] if candidates else None
    result = InitialStereoMatchResult(
        kpts0=kpts0,
        kpts1=kpts1,
        matches=matches,
        scores=scores,
        selected_left_idx=selected.left_idx if selected is not None else None,
        selected_right_idx=selected.right_idx if selected is not None else None,
        selected_score=selected.score if selected is not None else None,
        selected_left_xy=selected.left_xy if selected is not None else None,
        selected_right_xy=selected.right_xy if selected is not None else None,
        message=(
            f"legacy SuperPoint+LightGlue selected {len(candidates)} ROI matches"
            if candidates
            else "no legacy LightGlue matches found inside ROI"
        ),
        candidates=candidates,
    )

    if visualize and save_path:
        _save_candidate_visualization(left_img, right_img, candidates, save_path)

    return result

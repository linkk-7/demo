"""Online tracking utilities.

This package is intentionally separate from the original path-based
``get_init_id`` and ``track`` modules.
"""

from .image_utils import image_array_to_tensor, load_image_array
from .config_models import (
    CameraInfo,
    ImageInfo,
    InitPolicy,
    ROIConfig,
    ROIRegion,
    ReferenceInfo,
    ValidationResult,
)
from .models import (
    CandidatePoint,
    InitialStereoMatchResult,
    OcclusionSegment,
    OnlineStereoTrackingResult,
    TrackingResult,
)
from .online_init_matcher import match_initial_candidates
from .sync_point_tracker import SyncPointTracker
from .displacement_utils import (
    compute_displacement_from_points,
    compute_xyz_from_stereo_points,
    compute_xyz_from_stereo_points_temp_calibration,
    load_calibration_data,
    median_displacement,
)
from .roi_manager import (
    create_roi_config_from_selection,
    load_roi_config,
    save_roi_config,
    validate_roi_config,
)

__all__ = [
    "CameraInfo",
    "CandidatePoint",
    "InitialStereoMatchResult",
    "ImageInfo",
    "InitPolicy",
    "OcclusionSegment",
    "OnlineStereoTrackingResult",
    "ROIConfig",
    "ROIRegion",
    "ReferenceInfo",
    "SyncPointTracker",
    "TrackingResult",
    "ValidationResult",
    "compute_displacement_from_points",
    "compute_xyz_from_stereo_points",
    "compute_xyz_from_stereo_points_temp_calibration",
    "create_roi_config_from_selection",
    "image_array_to_tensor",
    "load_image_array",
    "load_calibration_data",
    "load_roi_config",
    "match_initial_candidates",
    "median_displacement",
    "save_roi_config",
    "validate_roi_config",
]

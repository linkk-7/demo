import os
from typing import Optional, Tuple

import cv2
import numpy as np

from .models import RectificationParams, RectifiedPacket
from .utils import build_output_path, infer_output_dirs, load_rectification_params, read_image


def _resolve_calibration_paths(
    calibration_folder: Optional[str],
    calibration_file1: Optional[str],
    calibration_file2: Optional[str],
    r: Optional[str],
    t: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if calibration_folder:
        calibration_file1 = calibration_file1 or os.path.join(calibration_folder, "calibration_1.npy")
        calibration_file2 = calibration_file2 or os.path.join(calibration_folder, "calibration_2.npy")
        r = r or os.path.join(calibration_folder, "R.npy")
        t = t or os.path.join(calibration_folder, "T.npy")
    return calibration_file1, calibration_file2, r, t


def _resolve_pair_inputs(
    frame_packet=None,
    left_path: Optional[str] = None,
    right_path: Optional[str] = None,
    left_image: Optional[np.ndarray] = None,
    right_image: Optional[np.ndarray] = None,
    frame_id: Optional[int] = None,
) -> Tuple[int, np.ndarray, np.ndarray, Optional[str], Optional[str]]:
    packet_left_image = getattr(frame_packet, "left_image", None) if frame_packet is not None else None
    packet_right_image = getattr(frame_packet, "right_image", None) if frame_packet is not None else None
    packet_left_path = getattr(frame_packet, "left_path", None) if frame_packet is not None else None
    packet_right_path = getattr(frame_packet, "right_path", None) if frame_packet is not None else None

    resolved_frame_id = frame_id
    if resolved_frame_id is None and frame_packet is not None:
        resolved_frame_id = getattr(frame_packet, "frame_id", None)
    if resolved_frame_id is None:
        raise ValueError("frame_id is required (provide frame_packet.frame_id or frame_id).")

    src_left_image = packet_left_image if packet_left_image is not None else left_image
    src_right_image = packet_right_image if packet_right_image is not None else right_image
    src_left_path = packet_left_path if packet_left_path else left_path
    src_right_path = packet_right_path if packet_right_path else right_path

    if src_left_image is None:
        if not src_left_path:
            raise ValueError("left image source is missing (no in-memory image and no path).")
        src_left_image = read_image(src_left_path, side="left")
    if src_right_image is None:
        if not src_right_path:
            raise ValueError("right image source is missing (no in-memory image and no path).")
        src_right_image = read_image(src_right_path, side="right")

    return int(resolved_frame_id), src_left_image, src_right_image, src_left_path, src_right_path


class StereoRectifier:
    """Reusable runtime rectifier for online processing."""

    def __init__(
        self,
        rectification_params: Optional[RectificationParams] = None,
        calibration_folder: Optional[str] = None,
        calibration_file1: Optional[str] = None,
        calibration_file2: Optional[str] = None,
        r: Optional[str] = None,
        t: Optional[str] = None,
        mode: str = "python",
        d1_matlab: Optional[str] = None,
        d2_matlab: Optional[str] = None,
        calibration_tag: Optional[str] = None,
        rectify_scale: float = 0.0,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
    ):
        calibration_file1, calibration_file2, r, t = _resolve_calibration_paths(
            calibration_folder=calibration_folder,
            calibration_file1=calibration_file1,
            calibration_file2=calibration_file2,
            r=r,
            t=t,
        )

        if rectification_params is None:
            if not calibration_file1 or not calibration_file2 or not r or not t:
                raise ValueError(
                    "StereoRectifier requires rectification_params or calibration file paths."
                )
            rectification_params = load_rectification_params(
                calibration_file1=calibration_file1,
                calibration_file2=calibration_file2,
                r=r,
                t=t,
                mode=mode,
                d1_matlab=d1_matlab,
                d2_matlab=d2_matlab,
                calibration_tag=calibration_tag,
            )

        self.rectification_params = rectification_params
        self.rectify_scale = rectify_scale
        self.interpolation = interpolation
        self.border_mode = border_mode

        self._prepared_image_size: Optional[Tuple[int, int]] = None
        self._map1x: Optional[np.ndarray] = None
        self._map1y: Optional[np.ndarray] = None
        self._map2x: Optional[np.ndarray] = None
        self._map2y: Optional[np.ndarray] = None
        self.q_matrix: Optional[np.ndarray] = None

    def prepare(self, image_size: Tuple[int, int]):
        normalized_size = (int(image_size[0]), int(image_size[1]))
        if self._prepared_image_size == normalized_size and self._map1x is not None:
            return

        params = self.rectification_params
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            params.camera_matrix_1,
            params.dist_coeffs_1,
            params.camera_matrix_2,
            params.dist_coeffs_2,
            normalized_size,
            params.rotation,
            params.translation,
            alpha=self.rectify_scale,
        )
        map1x, map1y = cv2.initUndistortRectifyMap(
            params.camera_matrix_1,
            params.dist_coeffs_1,
            r1,
            p1,
            normalized_size,
            cv2.CV_32FC1,
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            params.camera_matrix_2,
            params.dist_coeffs_2,
            r2,
            p2,
            normalized_size,
            cv2.CV_32FC1,
        )

        self._prepared_image_size = normalized_size
        self._map1x = map1x
        self._map1y = map1y
        self._map2x = map2x
        self._map2y = map2y
        self.q_matrix = q

    def rectify_pair(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if left_image is None or right_image is None:
            raise ValueError("left_image and right_image must not be None.")
        if left_image.shape[:2] != right_image.shape[:2]:
            raise ValueError(
                f"left/right image shape mismatch: {left_image.shape[:2]} vs {right_image.shape[:2]}"
            )

        self.prepare(left_image.shape[1::-1])
        rectified_left = cv2.remap(
            left_image,
            self._map1x,
            self._map1y,
            self.interpolation,
            borderMode=self.border_mode,
        )
        rectified_right = cv2.remap(
            right_image,
            self._map2x,
            self._map2y,
            self.interpolation,
            borderMode=self.border_mode,
        )
        return rectified_left, rectified_right

    def rectify_frame_packet(
        self,
        frame_packet=None,
        left_path: Optional[str] = None,
        right_path: Optional[str] = None,
        left_image: Optional[np.ndarray] = None,
        right_image: Optional[np.ndarray] = None,
        output_folder1: Optional[str] = None,
        output_folder2: Optional[str] = None,
        frame_id: Optional[int] = None,
        include_images: bool = True,
        save_images: bool = False,
        image_ext: str = "jpg",
        return_q_matrix: bool = False,
        append_rect_suffix: bool = True,
    ) -> RectifiedPacket:
        """
        Input priority:
        1) frame_packet.left_image/right_image
        2) explicit left_image/right_image
        3) frame_packet.left_path/right_path
        4) explicit left_path/right_path
        """
        (
            resolved_frame_id,
            src_left_image,
            src_right_image,
            src_left_path,
            src_right_path,
        ) = _resolve_pair_inputs(
            frame_packet=frame_packet,
            left_path=left_path,
            right_path=right_path,
            left_image=left_image,
            right_image=right_image,
            frame_id=frame_id,
        )

        rect_left, rect_right = self.rectify_pair(src_left_image, src_right_image)

        left_rect_path = None
        right_rect_path = None
        if save_images:
            inferred_left_out, inferred_right_out = infer_output_dirs(
                frame_packet=frame_packet,
                left_path=src_left_path,
                right_path=src_right_path,
                output_folder1=output_folder1,
                output_folder2=output_folder2,
            )
            if not inferred_left_out or not inferred_right_out:
                raise ValueError(
                    "save_images=True requires output_folder1/output_folder2, "
                    "or paths that allow legacy output inference."
                )

            os.makedirs(inferred_left_out, exist_ok=True)
            os.makedirs(inferred_right_out, exist_ok=True)
            left_rect_path = build_output_path(
                output_dir=inferred_left_out,
                source_path=src_left_path,
                prefix="L",
                frame_id=resolved_frame_id,
                image_ext=image_ext,
                append_rect_suffix=append_rect_suffix,
            )
            right_rect_path = build_output_path(
                output_dir=inferred_right_out,
                source_path=src_right_path,
                prefix="R",
                frame_id=resolved_frame_id,
                image_ext=image_ext,
                append_rect_suffix=append_rect_suffix,
            )

            ok_l = cv2.imwrite(left_rect_path, rect_left)
            ok_r = cv2.imwrite(right_rect_path, rect_right)
            if not ok_l or not ok_r:
                raise IOError("failed to save rectified image pair.")

        keep_images = include_images or not save_images
        return RectifiedPacket(
            frame_id=resolved_frame_id,
            left_rect_path=left_rect_path,
            right_rect_path=right_rect_path,
            left_rect_image=rect_left if keep_images else None,
            right_rect_image=rect_right if keep_images else None,
            q_matrix=self.q_matrix if return_q_matrix else None,
            calibration_tag=self.rectification_params.calibration_tag,
        )


def rectify_frame_pair(
    calibration_file1: Optional[str] = None,
    calibration_file2: Optional[str] = None,
    r: Optional[str] = None,
    t: Optional[str] = None,
    mode: str = "python",
    frame_packet=None,
    left_path: Optional[str] = None,
    right_path: Optional[str] = None,
    left_image: Optional[np.ndarray] = None,
    right_image: Optional[np.ndarray] = None,
    output_folder1: Optional[str] = None,
    output_folder2: Optional[str] = None,
    frame_id: Optional[int] = None,
    include_images: bool = True,
    save_images: bool = False,
    image_ext: str = "jpg",
    rectify_scale: float = 0.0,
    d1_matlab: Optional[str] = None,
    d2_matlab: Optional[str] = None,
    calibration_tag: Optional[str] = None,
    return_q_matrix: bool = False,
    rectification_params: Optional[RectificationParams] = None,
    append_rect_suffix: bool = True,
    rectifier: Optional[StereoRectifier] = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    calibration_folder: Optional[str] = None,
) -> RectifiedPacket:
    """
    Rectify one left-right frame pair.

    This compatibility wrapper still supports legacy function-style calls.
    New online pipeline should prefer holding a long-lived StereoRectifier instance.
    """
    active_rectifier = rectifier or StereoRectifier(
        rectification_params=rectification_params,
        calibration_folder=calibration_folder,
        calibration_file1=calibration_file1,
        calibration_file2=calibration_file2,
        r=r,
        t=t,
        mode=mode,
        d1_matlab=d1_matlab,
        d2_matlab=d2_matlab,
        calibration_tag=calibration_tag,
        rectify_scale=rectify_scale,
        interpolation=interpolation,
        border_mode=border_mode,
    )
    return active_rectifier.rectify_frame_packet(
        frame_packet=frame_packet,
        left_path=left_path,
        right_path=right_path,
        left_image=left_image,
        right_image=right_image,
        output_folder1=output_folder1,
        output_folder2=output_folder2,
        frame_id=frame_id,
        include_images=include_images,
        save_images=save_images,
        image_ext=image_ext,
        return_q_matrix=return_q_matrix,
        append_rect_suffix=append_rect_suffix,
    )

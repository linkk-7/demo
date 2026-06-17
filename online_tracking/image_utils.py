"""Image array helpers for the online tracking pipeline.

The original LightGlue helpers used by this project load images from paths.
These helpers accept already-captured OpenCV ``np.ndarray`` images and return
``torch.float32`` tensors in ``C,H,W`` format with values in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def _validate_input_color(input_color: str) -> str:
    normalized = str(input_color).lower()
    if normalized not in {"bgr", "rgb", "gray"}:
        raise ValueError("input_color must be one of: 'bgr', 'rgb', 'gray'.")
    return normalized


def _validate_resize(resize: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if resize is None:
        return None
    if not isinstance(resize, (tuple, list)) or len(resize) != 2:
        raise ValueError("resize must be None or a (width, height) tuple.")
    width, height = int(resize[0]), int(resize[1])
    if width <= 0 or height <= 0:
        raise ValueError("resize width and height must be positive.")
    return width, height


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32, copy=False)
    if image_float.size == 0:
        raise ValueError("image must not be empty.")

    if np.issubdtype(image.dtype, np.integer):
        return image_float / 255.0

    finite_values = image_float[np.isfinite(image_float)]
    if finite_values.size == 0:
        raise ValueError("image contains no finite values.")

    if float(finite_values.max()) > 1.0:
        image_float = image_float / 255.0
    return np.clip(image_float, 0.0, 1.0)


def image_array_to_tensor(image: np.ndarray, input_color: str = "bgr") -> torch.Tensor:
    """Convert an OpenCV image array to a LightGlue-style tensor.

    Args:
        image: OpenCV ``np.ndarray`` image. Supported shapes are ``H,W``,
            ``H,W,1`` and ``H,W,3``.
        input_color: Color order of the input array: ``"bgr"``, ``"rgb"``,
            or ``"gray"``. BGR arrays are converted to RGB.

    Returns:
        ``torch.float32`` tensor with shape ``C,H,W`` and values in ``[0, 1]``.
    """
    input_color = _validate_input_color(input_color)

    if image is None:
        raise ValueError("image is None.")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"image must be a numpy.ndarray, got {type(image)}.")
    if image.ndim not in {2, 3}:
        raise ValueError(f"unsupported image dimensions: {image.ndim}.")

    if image.ndim == 2:
        if input_color in {"bgr", "rgb"}:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = image[:, :, None]
    else:
        channels = image.shape[2]
        if channels == 1:
            if input_color in {"bgr", "rgb"}:
                image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
            else:
                image = image
        elif channels == 3:
            if input_color == "bgr":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif input_color == "gray":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, None]
        else:
            raise ValueError(f"unsupported channel count: {channels}.")

    image = _normalize_image(image)
    if image.ndim == 2:
        image = image[:, :, None]

    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return tensor.to(dtype=torch.float32)


def load_image_array(
    image: np.ndarray,
    resize: Optional[Tuple[int, int]] = None,
    input_color: str = "bgr",
) -> torch.Tensor:
    """Load an in-memory image array into a LightGlue-compatible tensor.

    Args:
        image: OpenCV ``np.ndarray`` image.
        resize: Optional ``(width, height)`` tuple passed to ``cv2.resize``.
            ``None`` keeps the original image size.
        input_color: Color order of the input array: ``"bgr"``, ``"rgb"``,
            or ``"gray"``.

    Returns:
        ``torch.float32`` tensor with shape ``C,H,W`` and values in ``[0, 1]``.
    """
    resize = _validate_resize(resize)

    if image is None:
        raise ValueError("image is None.")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"image must be a numpy.ndarray, got {type(image)}.")

    if resize is not None:
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)

    return image_array_to_tensor(image, input_color=input_color)

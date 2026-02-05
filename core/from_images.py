"""
Module: core/from_images.py
Responsibility:
  - Invert 2D orientation fields from color-map images and color bars.
  - Provide config-driven image-to-field conversion utilities.

Public API:
  - FromImagesConfig
  - invert_field_from_images(config)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from matplotlib import image as mpimg
from core.representation import rotation_matrix_z, rotation_matrix_to_quaternion


__all__ = ["FromImagesConfig", "invert_field_from_images"]


@dataclass(frozen=True)
class FromImagesConfig:
    field_image: Path
    cbar_image: Path
    cbar_min: float
    cbar_max: float
    cbar_direction: str
    defect_threshold: int
    color_tolerance: float
    defect_policy: str
    orientation_format: str
    mask_min_area: int
    mask_morph: str
    mask_kernel: int


def _load_rgb(path: Path) -> np.ndarray:
    img = mpimg.imread(path)
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def _smooth_colors(colors: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return colors
    pad = window // 2
    padded = np.pad(colors, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.empty_like(colors, dtype=np.float32)
    for i in range(colors.shape[0]):
        smoothed[i] = np.mean(padded[i : i + window], axis=0)
    return smoothed


def _build_color_map(cbar: np.ndarray, direction: str, vmin: float, vmax: float) -> Tuple[np.ndarray, np.ndarray]:
    if cbar.shape[0] >= cbar.shape[1]:
        center = cbar.shape[1] // 2
        span = min(3, center)
        columns = cbar[:, center - span : center + span + 1, :]
        colors = np.mean(columns, axis=1)
    else:
        center = cbar.shape[0] // 2
        span = min(3, center)
        rows = cbar[center - span : center + span + 1, :, :]
        colors = np.mean(rows, axis=0)
    colors = colors.reshape(-1, 3)
    colors = _smooth_colors(colors, window=5)
    if direction == "top_high":
        values = np.linspace(vmax, vmin, num=len(colors))
    else:
        values = np.linspace(vmin, vmax, num=len(colors))
    return colors.astype(np.float32), values.astype(np.float32)


def _detect_defects(field_rgb: np.ndarray, threshold: int) -> np.ndarray:
    return np.all(field_rgb < threshold, axis=-1)


def _remove_small_zero_components(mask: np.ndarray, min_area: int) -> Tuple[np.ndarray, int]:
    try:
        from scipy.ndimage import label
    except ImportError:
        return mask, 0
    if min_area <= 0:
        return mask, 0
    zero_mask = ~mask
    labeled, count = label(zero_mask)
    if count == 0:
        return mask, 0
    removed_pixels = 0
    for idx in range(1, count + 1):
        component = labeled == idx
        area = int(np.sum(component))
        if area < min_area:
            mask[component] = True
            removed_pixels += area
    return mask, removed_pixels


def _morph_zero_mask(mask: np.ndarray, morph: str, kernel: int) -> np.ndarray:
    try:
        from scipy.ndimage import binary_closing, binary_opening
    except ImportError:
        return mask
    if morph == "none":
        return mask
    structure = np.ones((kernel, kernel), dtype=bool)
    zero_mask = ~mask
    if morph == "open":
        zero_mask = binary_opening(zero_mask, structure=structure)
    elif morph == "close":
        zero_mask = binary_closing(zero_mask, structure=structure)
    return ~zero_mask


def _invert_from_images_impl(config: FromImagesConfig) -> Dict[str, np.ndarray]:
    """Internal implementation for image inversion."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy is required for image inversion (pip install scipy)") from exc

    field_rgb = _load_rgb(config.field_image)
    cbar_rgb = _load_rgb(config.cbar_image)

    colors, values = _build_color_map(cbar_rgb, config.cbar_direction, config.cbar_min, config.cbar_max)
    tree = cKDTree(colors)
    flat_rgb = field_rgb.reshape(-1, 3).astype(np.float32)
    distances, indices = tree.query(flat_rgb, k=1)
    theta = values[indices].reshape(field_rgb.shape[:2])

    defect_mask = _detect_defects(field_rgb, config.defect_threshold)
    far_mask = distances.reshape(field_rgb.shape[:2]) > config.color_tolerance
    mask_raw = ~(defect_mask | far_mask)

    removed_pixels = 0
    mask = mask_raw.copy()
    mask_postprocess_applied = False
    try:
        from scipy import ndimage as _ndimage
        _ = _ndimage
        if config.mask_min_area > 0 or config.mask_morph != "none":
            mask, removed_pixels = _remove_small_zero_components(mask, config.mask_min_area)
            mask = _morph_zero_mask(mask, config.mask_morph, config.mask_kernel)
            mask_postprocess_applied = True
    except ImportError:
        mask = mask_raw

    if config.defect_policy == "nan":
        theta = theta.astype(np.float32)
        theta[~mask] = np.nan
    elif config.defect_policy == "zero":
        theta = theta.astype(np.float32)
        theta[~mask] = 0.0
    else:
        theta = theta.astype(np.float32)
        if np.any(~mask):
            valid_indices = np.argwhere(mask)
            invalid_indices = np.argwhere(~mask)
            if valid_indices.size > 0:
                tree_valid = cKDTree(valid_indices)
                _, nearest = tree_valid.query(invalid_indices, k=1)
                theta[tuple(invalid_indices.T)] = theta[tuple(valid_indices[nearest].T)]

    mask_uint8 = mask.astype(np.uint8)
    result: Dict[str, np.ndarray] = {
        "dim": 2,
        "shape": np.array(theta.shape, dtype=np.int32),
        "mask": mask_uint8,
        "mask_raw": mask_raw.astype(np.uint8),
        "theta_deg": theta,
    }
    result["mask_postprocess_applied"] = np.array(mask_postprocess_applied, dtype=np.bool_)
    result["mask_removed_pixels"] = np.array(removed_pixels, dtype=np.int32)

    if config.orientation_format == "euler":
        euler = np.zeros((*theta.shape, 3), dtype=np.float32)
        euler[..., 0] = theta
        result["euler_deg"] = euler
    elif config.orientation_format == "matrix":
        result["R"] = rotation_matrix_z(theta)
    else:
        result["q"] = rotation_matrix_to_quaternion(rotation_matrix_z(theta))

    return result


def invert_field_from_images(config: FromImagesConfig) -> Dict[str, np.ndarray]:
    return _invert_from_images_impl(config)

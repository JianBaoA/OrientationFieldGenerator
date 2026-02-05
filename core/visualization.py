"""
Module: core/visualization.py
Responsibility:
  - Render orientation field previews (2D/3D slices) and mask images.
  - Provide unified preview entry points for CLI/GUI reuse.

Public API:
  - compute_slice_arrays_from_npz(npz_path, axis, index)
  - render_slice_3d(npz_path, axis, prefix, out_dir, index, ...)
  - render_slices_batch(field_or_npz, axis, indices, out_dir, prefix, ...)
  - render_orthoslices(field_or_npz, index, out_dir, ...)
  - render_field_and_mask_png(theta_2d, mask_2d, out_field_png, out_mask_png, ...)
  - resolve_global_vmin_vmax(field)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator
logger = logging.getLogger(__name__)

__all__ = [
    "compute_slice_arrays_from_npz",
    "render_field_and_mask_png",
    "resolve_global_vmin_vmax",
    "render_slice_3d",
    "render_slices_batch",
    "render_orthoslices",
]

def _save_figure(fig: Figure, path: Path, dpi: int = 150) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path


def theta_deg_from_R(matrix: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]))


def compute_theta_from_R(matrix: np.ndarray) -> np.ndarray:
    return theta_deg_from_R(matrix)


def compute_slice_arrays_from_npz(
    npz_path: str | Path,
    axis: str,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    field = _load_field(Path(npz_path))
    _validate_field_shapes(field)
    mask = field.get("mask")
    if mask is None:
        raise ValueError("mask is required for preview")
    theta = field.get("theta_deg")
    if theta is None and "R" in field:
        if field["R"].ndim >= 5:
            theta = None
        else:
            theta = theta_deg_from_R(field["R"])
    if theta is None:
        if "R" in field:
            theta_slice = _theta_slice_from_R(field["R"], axis, index, "voxel", "avg")
            mask_slice = get_slice(mask, axis, index, "voxel", "avg", is_mask=True)
            theta_slice = theta_slice.astype(np.float32, copy=True)
            theta_slice[mask_slice == 0] = np.nan
            return theta_slice, mask_slice
        raise ValueError("theta_deg or R required for preview")

    if theta.ndim == 2:
        theta_slice = theta
        mask_slice = mask
    else:
        theta_slice = get_slice(theta, axis, index, "voxel", "avg", is_mask=False)
        mask_slice = get_slice(mask, axis, index, "voxel", "avg", is_mask=True)

    theta_slice = theta_slice.astype(np.float32, copy=True)
    theta_slice[mask_slice == 0] = np.nan
    return theta_slice, mask_slice


def render_field_and_mask_png(
    theta_2d: np.ndarray,
    mask_2d: np.ndarray,
    out_field_png: Path,
    out_mask_png: Path,
    vmin: Optional[float],
    vmax: Optional[float],
    style_cfg: Optional[Dict[str, object]] = None,
) -> tuple[Path, Path]:
    cfg = style_cfg or {}
    dpi = int(cfg.get("dpi", 150))
    figsize = cfg.get("figsize", (5.2, 4.2))
    layout_main = cfg.get("layout_main", [0.07, 0.07, 0.74, 0.86])
    layout_side = cfg.get("layout_side", [0.87, 0.1, 0.02, 0.8])

    cb_min, cb_max = _resolve_color_bounds(theta_2d, mask_2d, vmin, vmax)
    cmap = _field_colormap()

    out_field_png.parent.mkdir(parents=True, exist_ok=True)
    out_mask_png.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvas(fig)
    ax_main = fig.add_axes(layout_main)
    ax_side = fig.add_axes(layout_side)
    ax_main.set_axis_off()

    field_display = _prepare_field_display(theta_2d, mask_2d)
    ax_main.imshow(field_display, cmap=cmap, vmin=cb_min, vmax=cb_max, interpolation="nearest")
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=mcolors.Normalize(cb_min, cb_max), cmap=cmap),
        cax=ax_side,
    )
    tick_values = [cb_min, cb_max]
    tick_labels = [f"{cb_min}", f"{cb_max}"]
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.yaxis.set_major_locator(FixedLocator(tick_values))
    cbar.ax.yaxis.set_major_formatter(FixedFormatter(tick_labels))
    cbar.ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=True,
        labelleft=False,
        labelright=True,
        length=3,
        width=1,
        labelsize=9,
    )
    cbar.ax.set_axis_on()
    fig.savefig(out_field_png, dpi=dpi)

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvas(fig)
    ax_main = fig.add_axes(layout_main)
    ax_side = fig.add_axes(layout_side)
    ax_main.set_axis_off()
    mask_display = _prepare_mask_display(mask_2d)
    ax_main.imshow(mask_display, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    legend_handles = [
        mpatches.Patch(color="#000000", label="0"),
        mpatches.Patch(color="#ffffff", label="1"),
    ]
    ax_side.legend(
        handles=legend_handles,
        loc="center",
        frameon=True,
        fontsize=8,
    )
    ax_side.set_axis_off()
    fig.savefig(out_mask_png, dpi=dpi)
    return out_field_png, out_mask_png


def resolve_global_vmin_vmax(field: Dict[str, np.ndarray]) -> Tuple[Optional[float], Optional[float]]:
    meta = field.get("meta", {})
    cb_min = meta.get("cb_min")
    cb_max = meta.get("cb_max")
    if cb_min is not None and cb_max is not None:
        return float(cb_min), float(cb_max)

    mask = field.get("mask")
    theta = field.get("theta_deg")
    if theta is None and "R" in field:
        if field["R"].ndim <= 4:
            theta = theta_deg_from_R(field["R"])
    if theta is None or mask is None:
        return None, None
    valid = theta[mask == 1]
    valid = valid[~np.isnan(valid)]
    if valid.size == 0:
        return -1.0, 1.0
    return float(np.floor(np.nanmin(valid))), float(np.ceil(np.nanmax(valid)))


def _resolve_color_bounds(
    theta: np.ndarray, mask: np.ndarray, vmin: Optional[float], vmax: Optional[float]
) -> Tuple[float, float]:
    if vmin is not None and vmax is not None:
        cb_min = float(np.floor(vmin))
        cb_max = float(np.ceil(vmax))
    else:
        valid = theta[mask == 1]
        valid = valid[~np.isnan(valid)]
        if valid.size == 0:
            return -1.0, 1.0
        cb_min = float(np.floor(np.nanmin(valid)))
        cb_max = float(np.ceil(np.nanmax(valid)))
    if cb_min == cb_max:
        cb_max = cb_min + 1.0
    return cb_min, cb_max


def _field_colormap() -> mcolors.LinearSegmentedColormap:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_white_blue", ["#0000ff", "#ffffff", "#ff0000"]
    )
    cmap.set_bad("black")
    return cmap


def _create_layout(figsize: Tuple[float, float] = (5.2, 4.2)) -> tuple[Figure, Axes, Axes]:
    fig = Figure(figsize=figsize)
    FigureCanvas(fig)
    ax_main = fig.add_axes([0.07, 0.07, 0.74, 0.86])
    ax_side = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    ax_main.set_axis_off()
    return fig, ax_main, ax_side


def _imshow_aligned(
    ax: Axes,
    data: np.ndarray,
    cmap: mcolors.Colormap,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )


def slice_3d(arr: np.ndarray, axis: str, index: int) -> np.ndarray:
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError("axis must be x|y|z")
    axis_index = axis_map[axis]
    if axis_index >= arr.ndim:
        raise ValueError(f"slice axis {axis} out of bounds for array ndim={arr.ndim}")
    length = arr.shape[axis_index]
    if not (0 <= index < length):
        raise IndexError(f"slice index out of range: axis={axis} index={index} length={length}")
    if axis == "x":
        result = arr[index, :, :]
    elif axis == "y":
        result = arr[:, index, :]
    else:
        result = arr[:, :, index]
    if result.ndim < 2:
        raise ValueError("slice_3d result must be at least 2D")
    return result


def get_slice(
    data: np.ndarray,
    axis: str,
    index: int,
    slice_mode: str = "voxel",
    face_policy: str = "avg",
    is_mask: bool = False,
) -> np.ndarray:
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError("axis must be x|y|z")
    axis_index = axis_map[axis]
    if axis_index >= data.ndim:
        raise ValueError(f"slice axis {axis} out of bounds for array ndim={data.ndim}")
    length = data.shape[axis_index]
    if slice_mode == "voxel":
        if not (0 <= index < length):
            raise IndexError(f"slice index out of range: axis={axis} index={index} length={length}")
        return slice_3d(data, axis, index)
    if not (1 <= index < length):
        raise IndexError(f"slice index out of range: axis={axis} index={index} length={length}")
    left = slice_3d(data, axis, index - 1)
    right = slice_3d(data, axis, index)
    if is_mask:
        return (left == 1) & (right == 1)
    if face_policy == "left":
        return left
    if face_policy == "right":
        return right
    return 0.5 * (left + right)


def _field_shape(field: Dict[str, np.ndarray]) -> Tuple[int, ...]:
    shape = field.get("shape")
    if shape is None:
        return tuple(int(value) for value in field["mask"].shape)
    if isinstance(shape, np.ndarray):
        return tuple(int(value) for value in shape.tolist())
    return tuple(int(value) for value in shape)


def _validate_field_shapes(field: Dict[str, np.ndarray]) -> None:
    shape = _field_shape(field)
    mask = field.get("mask")
    if mask is not None and tuple(mask.shape) != shape:
        raise ValueError(f"mask shape {mask.shape} does not match field shape {shape}")
    theta = field.get("theta_deg")
    if theta is not None:
        if theta.ndim < 2:
            raise ValueError("theta_deg must be at least 2D")
        if tuple(theta.shape[: len(shape)]) != shape:
            raise ValueError(f"theta_deg shape {theta.shape} does not match field shape {shape}")
    matrix = field.get("R")
    if matrix is not None:
        if tuple(matrix.shape[: len(shape)]) != shape:
            raise ValueError(f"R shape {matrix.shape} does not match field shape {shape}")
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f"R must have trailing (3,3) rotation matrix dims, got {matrix.shape[-2:]}")


def _theta_slice_from_R(
    matrix: np.ndarray,
    axis: str,
    index: int,
    slice_mode: str,
    face_policy: str,
) -> np.ndarray:
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError("axis must be x|y|z")
    axis_index = axis_map[axis]
    if axis_index >= matrix.ndim:
        raise ValueError(f"slice axis {axis} out of bounds for array ndim={matrix.ndim}")
    length = matrix.shape[axis_index]
    if slice_mode == "voxel":
        if not (0 <= index < length):
            raise IndexError(f"slice index out of range: axis={axis} index={index} length={length}")
        matrix_slice = slice_3d(matrix, axis, index)
        return theta_deg_from_R(matrix_slice)
    if not (1 <= index < length):
        raise IndexError(f"slice index out of range: axis={axis} index={index} length={length}")
    left = theta_deg_from_R(slice_3d(matrix, axis, index - 1))
    right = theta_deg_from_R(slice_3d(matrix, axis, index))
    if face_policy == "left":
        return left
    if face_policy == "right":
        return right
    return 0.5 * (left + right)


def _warn_on_slice_diagnostics(
    theta_slice: np.ndarray,
    mask_slice: np.ndarray,
    axis: str,
    index: int,
) -> None:
    issues = []
    if theta_slice.ndim != 2:
        issues.append("theta_not_2d")
    if mask_slice.ndim != 2:
        issues.append("mask_not_2d")
    if theta_slice.shape != mask_slice.shape:
        issues.append("shape_mismatch")

    mask_sum = int(np.sum(mask_slice)) if mask_slice.size else 0
    finite_count = int(np.isfinite(theta_slice).sum()) if theta_slice.size else 0
    safe_min = float(np.nanmin(theta_slice)) if finite_count else "nan"
    safe_max = float(np.nanmax(theta_slice)) if finite_count else "nan"

    if mask_sum == 0:
        issues.append("mask_sum_zero")
    if finite_count == 0:
        issues.append("theta_all_nonfinite")
    if finite_count and safe_min == safe_max:
        issues.append("theta_constant")

    if issues:
        logger.warning(
            "slice diagnostics axis=%s index=%d issue=%s mask_sum=%d finite=%d theta_min=%s theta_max=%s",
            axis,
            index,
            ",".join(issues),
            mask_sum,
            finite_count,
            safe_min,
            safe_max,
        )


def _prepare_field_display(theta_2d: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    field = theta_2d.astype(np.float32)
    field[mask_2d == 0] = np.nan
    return np.pad(field, pad_width=1, mode="constant", constant_values=np.nan)


def _prepare_mask_display(mask_2d: np.ndarray) -> np.ndarray:
    mask_disp = (mask_2d > 0).astype(np.float32)
    return np.pad(mask_disp, pad_width=1, mode="constant", constant_values=0.0)


def render_field_and_mask(
    theta_2d: np.ndarray,
    mask_2d: np.ndarray,
    prefix: Path,
    vmin: Optional[float],
    vmax: Optional[float],
    dpi: int = 150,
    output_mode: str = "both",
) -> Tuple[Optional[Path], Optional[Path]]:
    cb_min, cb_max = _resolve_color_bounds(theta_2d, mask_2d, vmin, vmax)
    field_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    if output_mode in {"both", "field"}:
        field_path = prefix.with_name(prefix.name + "_field.png")
        field_display = _prepare_field_display(theta_2d, mask_2d)
        cmap = _field_colormap()
        fig, ax, ax_side = _create_layout()
        _imshow_aligned(ax, field_display, cmap=cmap, vmin=cb_min, vmax=cb_max)
        cbar = fig.colorbar(
            cm.ScalarMappable(norm=mcolors.Normalize(cb_min, cb_max), cmap=cmap),
            cax=ax_side,
        )
        tick_values = [cb_min, cb_max]
        tick_labels = [f"{cb_min}", f"{cb_max}"]
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.yaxis.set_major_locator(FixedLocator(tick_values))
        cbar.ax.yaxis.set_major_formatter(FixedFormatter(tick_labels))
        cbar.ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=True,
            labelleft=False,
            labelright=True,
            length=3,
            width=1,
            labelsize=9,
        )
        cbar.ax.set_axis_on()
        _save_figure(fig, field_path, dpi=dpi)
    if output_mode in {"both", "mask"}:
        mask_path = prefix.with_name(prefix.name + "_mask.png")
        mask_display = _prepare_mask_display(mask_2d)
        fig, ax, ax_side = _create_layout()
        _imshow_aligned(ax, mask_display, cmap="gray", vmin=0.0, vmax=1.0)
        legend_handles = [
            mpatches.Patch(color="#000000", label="0"),
            mpatches.Patch(color="#ffffff", label="1"),
        ]
        ax_side.legend(
            handles=legend_handles,
            loc="center",
            frameon=True,
            fontsize=8,
        )
        ax_side.set_axis_off()
        _save_figure(fig, mask_path, dpi=dpi)
    return field_path, mask_path


def _load_field(field_or_npz: Dict[str, np.ndarray] | Path) -> Dict[str, np.ndarray]:
    if isinstance(field_or_npz, Path):
        from core.io import read_npz

        return read_npz(field_or_npz)
    return field_or_npz


def preview_npz(
    npz_path: str | Path,
    prefix: str | Path,
    *,
    slice_axis: str = "z",
    slice_index: str = "mid",
    output_mode: str = "both",
    slice_mode: str = "voxel",
    face_policy: str = "avg",
) -> None:
    field = _load_field(Path(npz_path))
    _validate_field_shapes(field)
    vmin, vmax = resolve_global_vmin_vmax(field)
    mask = field.get("mask")
    if mask is None:
        raise ValueError("mask is required for preview")
    theta = field.get("theta_deg")
    matrix = field.get("R")
    if theta is None:
        if matrix is None:
            raise ValueError("theta_deg or R required for preview")
        if matrix.ndim <= 4:
            theta = theta_deg_from_R(matrix)

    if theta is not None and theta.ndim == 2:
        theta_slice = theta
        mask_slice = mask
    else:
        if slice_index == "mid":
            axis_map = {"x": 0, "y": 1, "z": 2}
            axis_index = axis_map.get(slice_axis, 2)
            length_source = theta if theta is not None else matrix
            index_value = length_source.shape[axis_index] // 2
        else:
            index_value = int(slice_index)
        if theta is None:
            theta_slice = _theta_slice_from_R(matrix, slice_axis, index_value, slice_mode, face_policy)
        else:
            theta_slice = get_slice(theta, slice_axis, index_value, slice_mode, face_policy, is_mask=False)
        mask_slice = get_slice(mask, slice_axis, index_value, slice_mode, face_policy, is_mask=True)
        _warn_on_slice_diagnostics(theta_slice, mask_slice, slice_axis, index_value)

    prefix_path = Path(prefix).with_suffix("")
    render_field_and_mask(theta_slice, mask_slice, prefix_path, vmin, vmax, output_mode=output_mode)


def render_slice_3d(
    npz_path: Path,
    axis: str,
    prefix: str,
    out_dir: Path,
    index: int,
    slice_mode: str = "voxel",
    face_policy: str = "avg",
) -> tuple[Path, Path]:
    field = _load_field(npz_path)
    _validate_field_shapes(field)
    out_dir.mkdir(parents=True, exist_ok=True)
    vmin, vmax = resolve_global_vmin_vmax(field)
    mask = field.get("mask")
    if mask is None:
        raise ValueError("mask is required for preview")
    theta = field.get("theta_deg")
    if theta is None:
        if "R" not in field:
            raise ValueError("theta_deg or R required for preview")
        theta_slice = _theta_slice_from_R(field["R"], axis, index, slice_mode, face_policy)
    else:
        theta_slice = get_slice(theta, axis, index, slice_mode, face_policy, is_mask=False)
    mask_slice = get_slice(mask, axis, index, slice_mode, face_policy, is_mask=True)
    _warn_on_slice_diagnostics(theta_slice, mask_slice, axis, index)
    prefix_path = out_dir / prefix
    field_path, mask_path = render_field_and_mask(theta_slice, mask_slice, prefix_path, vmin, vmax, output_mode="both")
    return field_path, mask_path


def _collect_slice_values(
    theta: np.ndarray,
    mask: np.ndarray,
    slices: list[tuple[str, int, str, str]],
) -> np.ndarray:
    values = []
    for axis, index, slice_mode, face_policy in slices:
        theta_slice = get_slice(theta, axis, index, slice_mode, face_policy, is_mask=False)
        mask_slice = get_slice(mask, axis, index, slice_mode, face_policy, is_mask=True)
        valid = theta_slice[mask_slice == 1]
        valid = valid[~np.isnan(valid)]
        if valid.size:
            values.append(valid.ravel())
    if not values:
        return np.array([], dtype=np.float32)
    return np.concatenate(values)


def _global_vmin_vmax_for_slices(
    theta: np.ndarray,
    mask: np.ndarray,
    slices: list[tuple[str, int, str, str]],
) -> Tuple[float, float]:
    values = _collect_slice_values(theta, mask, slices)
    if values.size == 0:
        return -1.0, 1.0
    vmin = float(np.floor(np.nanmin(values)))
    vmax = float(np.ceil(np.nanmax(values)))
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def render_slices_batch(
    field_or_npz: Dict[str, np.ndarray] | Path,
    axis: str,
    indices: list[int],
    out_dir: Path,
    prefix: str,
    slice_mode: str = "voxel",
    face_policy: str = "avg",
    dpi: int = 150,
    output_mode: str = "both",
) -> None:
    field = _load_field(field_or_npz)
    _validate_field_shapes(field)
    mask = field.get("mask")
    if mask is None:
        raise ValueError("mask is required for preview")
    theta = field.get("theta_deg")
    if theta is None and "R" in field:
        theta = theta_deg_from_R(field["R"])
    if theta is None:
        raise ValueError("theta_deg or R required for preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    slices = [(axis, idx, slice_mode, face_policy) for idx in indices]
    vmin, vmax = _global_vmin_vmax_for_slices(theta, mask, slices)
    last_stats: Optional[tuple[float, float, float]] = None
    repeated = 0
    for idx in indices:
        theta_slice = get_slice(theta, axis, idx, slice_mode, face_policy, is_mask=False)
        mask_slice = get_slice(mask, axis, idx, slice_mode, face_policy, is_mask=True)
        _warn_on_slice_diagnostics(theta_slice, mask_slice, axis, idx)
        valid = theta_slice[mask_slice == 1]
        valid = valid[~np.isnan(valid)]
        nanmin = float(np.nanmin(valid)) if valid.size else float("nan")
        nanmax = float(np.nanmax(valid)) if valid.size else float("nan")
        mask_mean = float(np.mean(mask_slice))
        stats = (nanmin, nanmax, mask_mean)
        if last_stats is not None and stats == last_stats:
            repeated += 1
            if repeated >= 1:
                print(f"WARNING: slice stats repeated at {axis}{idx:03d}: {stats}")
        else:
            repeated = 0
        last_stats = stats
        output_path = out_dir / f"{prefix}_{axis}{idx:03d}"
        render_field_and_mask(theta_slice, mask_slice, output_path, vmin, vmax, dpi=dpi, output_mode=output_mode)


def render_orthoslices(
    field_or_npz: Dict[str, np.ndarray] | Path,
    index: int | str,
    out_dir: Path,
    dpi: int = 150,
) -> None:
    field = _load_field(field_or_npz)
    _validate_field_shapes(field)
    mask = field.get("mask")
    if mask is None:
        raise ValueError("mask is required for preview")
    theta = field.get("theta_deg")
    if theta is None and "R" in field:
        theta = theta_deg_from_R(field["R"])
    if theta is None:
        raise ValueError("theta_deg or R required for preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    axis_map = {"x": 0, "y": 1, "z": 2}
    slices = []
    for axis, axis_index in axis_map.items():
        if index == "mid":
            idx = theta.shape[axis_index] // 2
        else:
            idx = int(index)
        slices.append((axis, idx, "voxel", "avg"))
    vmin, vmax = _global_vmin_vmax_for_slices(theta, mask, slices)
    for axis, idx, slice_mode, face_policy in slices:
        theta_slice = get_slice(theta, axis, idx, slice_mode, face_policy, is_mask=False)
        mask_slice = get_slice(mask, axis, idx, slice_mode, face_policy, is_mask=True)
        _warn_on_slice_diagnostics(theta_slice, mask_slice, axis, idx)
        prefix = out_dir / axis
        render_field_and_mask(theta_slice, mask_slice, prefix, vmin, vmax, dpi=dpi, output_mode="both")

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple

import click
import numpy as np

from core.generator import RandomFieldConfig, generate_random_field
from core.io import get_output_root, read_npz, write_npz
from core.visualization import (
    render_field_and_mask_png,
    resolve_global_vmin_vmax,
    render_orthoslices,
    render_slices_batch,
)
from core.export import export_npz
from core.from_images import FromImagesConfig, invert_field_from_images


MESSAGES = {
    "zh": {
        "random_done": "随机取向场生成完成: {output}",
        "images_done": "云图反演完成: {output}",
        "preview_done": "预览图已保存: {preview}",
        "stats": "theta 统计(min/max/mean/std): {stats}",
    },
    "en": {
        "random_done": "Random orientation field generated: {output}",
        "images_done": "Field inverted from images: {output}",
        "preview_done": "Preview saved: {preview}",
        "stats": "theta stats (min/max/mean/std): {stats}",
    },
}


def _parse_shape(shape: str, dim: int) -> Tuple[int, ...]:
    parts = [int(value) for value in shape.split(",")]
    if len(parts) != dim:
        raise click.BadParameter("shape length must match dim")
    return tuple(parts)


def _mask_ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask.astype(bool)))


def _sample_orthogonality(matrix: np.ndarray, count: int = 10) -> float:
    flat = matrix.reshape(-1, 3, 3)
    count = min(count, flat.shape[0])
    indices = np.random.default_rng().choice(flat.shape[0], size=count, replace=False)
    sample = flat[indices]
    diffs = sample @ np.transpose(sample, (0, 2, 1)) - np.eye(3)
    return float(np.max(np.abs(diffs)))


def _sample_quat_norm(quat: np.ndarray, count: int = 10) -> float:
    flat = quat.reshape(-1, 4)
    count = min(count, flat.shape[0])
    indices = np.random.default_rng().choice(flat.shape[0], size=count, replace=False)
    norms = np.linalg.norm(flat[indices], axis=1)
    return float(np.max(np.abs(norms - 1.0)))


def _mask_diagnostics(mask: np.ndarray) -> dict:
    unique_vals = np.unique(mask)
    return {
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "min": int(mask.min()),
        "max": int(mask.max()),
        "unique": unique_vals[:10].tolist(),
    }


def _format_shape(shape: Tuple[int, ...]) -> str:
    return "(" + ",".join(str(value) for value in shape) + ")"


def _range_bound(value: float, is_min: bool) -> float:
    if float(value).is_integer():
        return float(value)
    return float(np.floor(value) if is_min else np.ceil(value))


def _theta_from_field(field: dict) -> Optional[np.ndarray]:
    theta = field.get("theta_deg")
    if theta is not None:
        return theta
    if "R" in field:
        return _theta_deg_from_R(field["R"])
    if "euler_deg" in field:
        return field["euler_deg"][..., 0]
    return None


def _theta_deg_from_R(matrix: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]))


def _get_slice(
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
        return _slice_3d(data, axis, index)
    if not (1 <= index < length):
        raise IndexError(f"slice index out of range: axis={axis} index={index} length={length}")
    left = _slice_3d(data, axis, index - 1)
    right = _slice_3d(data, axis, index)
    if is_mask:
        return (left == 1) & (right == 1)
    if face_policy == "left":
        return left
    if face_policy == "right":
        return right
    return 0.5 * (left + right)


def _slice_3d(arr: np.ndarray, axis: str, index: int) -> np.ndarray:
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
        matrix_slice = _slice_3d(matrix, axis, index)
        return _theta_deg_from_R(matrix_slice)
    if not (1 <= index < length):
        raise IndexError(f"slice index out of range: axis={axis} index={index} length={length}")
    left = _theta_deg_from_R(_slice_3d(matrix, axis, index - 1))
    right = _theta_deg_from_R(_slice_3d(matrix, axis, index))
    if face_policy == "left":
        return left
    if face_policy == "right":
        return right
    return 0.5 * (left + right)


def _render_preview_png(
    theta_slice: np.ndarray,
    mask_slice: np.ndarray,
    prefix: Path,
    vmin: Optional[float],
    vmax: Optional[float],
    output_mode: str = "both",
) -> None:
    field_path = prefix.with_name(prefix.name + "_field.png")
    mask_path = prefix.with_name(prefix.name + "_mask.png")
    render_field_and_mask_png(theta_slice, mask_slice, field_path, mask_path, vmin, vmax, None)
    if output_mode == "field" and mask_path.exists():
        mask_path.unlink()
    elif output_mode == "mask" and field_path.exists():
        field_path.unlink()


def _summarize_random(meta_min: dict, output_path: Path) -> str:
    seed_label = meta_min.get("seed_user")
    seed_entropy = meta_min.get("seed_entropy")
    seed_text = seed_label if seed_label is not None else (seed_entropy if seed_entropy is not None else "auto")
    return (
        "OK random output={output} dim={dim} shape={shape} grains={grains} "
        "mask_ratio={mask_ratio:.3f} gb_max={gb_max:.3f} hard_ok={hard_ok} seed={seed}"
    ).format(
        output=output_path,
        dim=meta_min["dim"],
        shape=_format_shape(tuple(meta_min["shape"])),
        grains=meta_min["grain_count"],
        mask_ratio=meta_min["mask_ratio"],
        gb_max=meta_min["gb_max"],
        hard_ok=meta_min["gb_hard_ok"],
        seed=seed_text,
    )


def _summarize_from_images(meta_min: dict, output_path: Path) -> str:
    return (
        "OK from-images output={output} shape={shape} mask_ratio={mask_ratio:.3f} "
        "theta_min={theta_min} theta_max={theta_max}"
    ).format(
        output=output_path,
        shape=_format_shape(tuple(meta_min["shape"])),
        mask_ratio=meta_min["mask_ratio"],
        theta_min=meta_min["theta_min"],
        theta_max=meta_min["theta_max"],
    )


def _summarize_export(input_path: Path, outputs: dict) -> str:
    formats = ",".join(outputs.keys())
    output_list = ",".join(str(path) for path in outputs.values())
    return f"OK export input={input_path} formats={formats} outputs={output_list}"


def _resolve_lang(ctx: click.Context, local_lang: Optional[str]) -> str:
    if local_lang:
        return local_lang
    return ctx.obj.get("lang", "zh")


def _normalize_lang(lang: Optional[str]) -> str:
    lang_value = lang or "zh"
    if lang_value not in MESSAGES:
        return "zh"
    return lang_value


def _output_root() -> Path:
    root = get_output_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _default_npz_output(prefix: str) -> Path:
    return _output_root() / f"{prefix}.npz"


def _default_preview_prefix(npz_path: Path) -> Path:
    preview_dir = _output_root() / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    return preview_dir / npz_path.stem


def _default_outdir(name: str) -> Path:
    path = _output_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _confirm_export_cli(ctx: click.Context, yes: bool, message: str) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        return True
    return click.confirm(message, default=False)


@click.group()
# Output behavior:
# - default: single-line summary per command
# - --verbose: add a couple of extra info lines
# - --debug: add diagnostics and write meta_debug_json
@click.option("--lang", type=click.Choice(["zh", "en"]), default="zh", show_default=True, help="UI language.")
@click.option("--verbose", is_flag=True, default=False, help="Show extra summary information.")
@click.option("--debug", is_flag=True, default=False, help="Show debug diagnostics and write meta_debug_json.")
@click.pass_context
def cli(ctx: click.Context, lang: str, verbose: bool, debug: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["lang"] = lang
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug


@cli.command(
    "random",
    help=(
        "Generate a random orientation field (grain topology + orientations).\n\n"
        "Absolute orientation (texture): 控制每个晶粒的绝对取向 θ_i，"
        "取值范围在 [theta_min, theta_max] 内。\n"
        "Grain-boundary misorientation: 控制相邻晶粒错配角 Δθ_ij 的统计分布，"
        "作用于邻接图。"
    ),
)
@click.option("--lang", type=click.Choice(["zh", "en"]), default=None, help="Override global language.")
@click.option("--dim", type=click.Choice(["2", "3"]), default="2", show_default=True)
@click.option("--shape", type=str, default="256,256", show_default=True)
@click.option("--grain-size-mean", type=float, default=32.0, show_default=True)
@click.option("--grain-size-distribution", type=click.Choice(["uniform", "normal", "lognormal"]), default="normal")
@click.option("--grain-size-std", type=float, default=6.0, show_default=True)
@click.option(
    "--orientation-distribution",
    type=click.Choice(["random", "normal"]),
    default="random",
    help="Absolute orientation sampling: random (uniform) or normal. 绝对取向采样方式。",
)
@click.option(
    "--mean-theta",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "Used only when --orientation-distribution=normal (absolute orientation mean). "
        "仅在 orientation_distribution=normal 时用于绝对取向采样。"
    ),
)
@click.option(
    "--orientation-std",
    type=float,
    default=10.0,
    show_default=True,
    help=(
        "Used only when --orientation-distribution=normal (absolute orientation std). "
        "仅在 orientation_distribution=normal 时用于绝对取向采样。"
    ),
)
@click.option("--orientation-format", type=click.Choice(["euler", "matrix", "quaternion"]), default="matrix")
@click.option(
    "--texture-mixture",
    type=str,
    default=None,
    help=(
        "Misorientation mixture spec (used when --gb-mis-sampler=mixture). "
        "Format: 'mean,std,weight;...'. 注意：这是错配角分布参数，不是绝对取向纹理混合。"
    ),
)
@click.option("--theta-min", type=float, default=None)
@click.option("--theta-max", type=float, default=None)
@click.option("--theta-boundary-policy", type=click.Choice(["truncate", "clip"]), default="truncate", show_default=True)
@click.option("--gb-mis-hard-max", type=float, default=None, show_default=True)
@click.option("--gb-mis-p95", type=float, default=None, show_default=True)
@click.option(
    "--gb-mis-sampler",
    type=click.Choice(["normal", "mixture"]),
    default="normal",
    show_default=True,
    help=(
        "Misorientation sampler: normal=single-peak distribution; "
        "mixture=multi-peak distribution defined by --texture-mixture. "
        "错配角采样：normal=单峰；mixture=由 --texture-mixture 定义的多峰。"
    ),
)
@click.option(
    "--gb-mis-std",
    type=float,
    default=None,
    show_default=True,
    help=(
        "Used only when --gb-mis-sampler=normal (misorientation std). "
        "仅在 gb_mis_sampler=normal 时生效。"
    ),
)
@click.option("--gb-alpha", type=float, default=0.7, show_default=True)
@click.option("--defect-enabled", is_flag=True, default=False)
@click.option("--defect-fraction", type=float, default=0.0, show_default=True)
@click.option("--defect-size-mean", type=float, default=8.0, show_default=True)
@click.option("--defect-size-distribution", type=click.Choice(["uniform", "normal", "lognormal"]), default="normal")
@click.option("--defect-size-std", type=float, default=2.0, show_default=True)
@click.option("--defect-threshold", type=float, default=1.0, show_default=True)
@click.option("--defect-shape", type=str, default=None, help="Defect shape: disk/ellipse/blob (2D) or sphere/ellipsoid/blob (3D).")
@click.option("--periodic", type=click.Choice(["none", "x", "y", "z", "xy", "xyz"]), default="none")
@click.option("--seed", type=int, default=None, show_default=True)
@click.option("--seed-entropy", type=int, default=None, show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--preview", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--preview-prefix", type=click.Path(path_type=Path), default=None)
@click.option("--slice-axis", type=click.Choice(["x", "y", "z"]), default="z", show_default=True)
@click.option("--slice-index", type=str, default="mid", show_default=True)
@click.option("--slice-mode", type=click.Choice(["voxel", "face"]), default="voxel", show_default=True)
@click.option("--face-policy", type=click.Choice(["avg", "left", "right"]), default="avg", show_default=True)
@click.pass_context
def random_command(
    ctx: click.Context,
    lang: Optional[str],
    dim: str,
    shape: str,
    grain_size_mean: float,
    grain_size_distribution: str,
    grain_size_std: float,
    orientation_distribution: str,
    mean_theta: float,
    orientation_std: float,
    orientation_format: str,
    texture_mixture: Optional[str],
    theta_min: Optional[float],
    theta_max: Optional[float],
    theta_boundary_policy: str,
    gb_mis_hard_max: Optional[float],
    gb_mis_p95: Optional[float],
    gb_mis_sampler: str,
    gb_mis_std: Optional[float],
    gb_alpha: float,
    defect_enabled: bool,
    defect_fraction: float,
    defect_size_mean: float,
    defect_size_distribution: str,
    defect_size_std: float,
    defect_threshold: float,
    defect_shape: Optional[str],
    periodic: str,
    seed: Optional[int],
    seed_entropy: Optional[int],
    output: Optional[Path],
    preview: Optional[Path],
    preview_prefix: Optional[Path],
    slice_axis: str,
    slice_index: str,
    slice_mode: str,
    face_policy: str,
) -> None:
    if output is None:
        output = _default_npz_output("random_field")
    dim_value = int(dim)
    shape_value = _parse_shape(shape, dim_value)
    periodic_value = periodic
    lang = _normalize_lang(_resolve_lang(ctx, lang))
    if theta_min is None or theta_max is None:
        raise click.BadParameter("theta-min/theta-max are required")
    if gb_mis_sampler == "mixture" and not texture_mixture:
        raise click.BadParameter("texture-mixture is required when gb-mis-sampler is mixture")
    if defect_shape is None:
        defect_shape = "disk" if dim_value == 2 else "sphere"
    mixture_components = None
    mixture_weights = None
    if texture_mixture:
        components = []
        weights = []
        for part in texture_mixture.split(";"):
            if not part.strip():
                continue
            mean_str, std_str, weight_str = [p.strip() for p in part.split(",")]
            mean_val = float(mean_str)
            std_val = float(std_str)
            weight_val = float(weight_str)
            components.append((mean_val, std_val))
            weights.append(weight_val)
        if not components:
            raise click.BadParameter("texture-mixture is empty")
        if any(np.isnan(weight) for weight in weights):
            raise click.BadParameter("texture-mixture weights must be valid numbers")
        if any(weight <= 0 for weight in weights):
            raise click.BadParameter("texture-mixture weights must be positive")
        total = sum(weights)
        if total <= 0:
            raise click.BadParameter("texture-mixture weights must sum to a positive value")
        mixture_weights = [w / total for w in weights]
        mixture_components = components
        if len(mixture_components) != len(mixture_weights):
            raise click.BadParameter("texture-mixture components and weights length mismatch")

    config = RandomFieldConfig(
        dim=dim_value,
        shape=shape_value,
        grain_size_mean=grain_size_mean,
        grain_size_distribution=grain_size_distribution,
        grain_size_std=grain_size_std,
        orientation_distribution=orientation_distribution,
        mean_theta_deg=mean_theta,
        orientation_std_deg=orientation_std,
        orientation_format=orientation_format,
        theta_min=theta_min,
        theta_max=theta_max,
        theta_boundary_policy=theta_boundary_policy,
        texture_mixture_components=mixture_components,
        texture_mixture_weights=mixture_weights,
        gb_mis_hard_max=gb_mis_hard_max,
        gb_mis_p95=gb_mis_p95,
        gb_mis_sampler=gb_mis_sampler,
        gb_mis_std=gb_mis_std,
        gb_alpha=gb_alpha,
        defect_enabled=defect_enabled,
        defect_fraction=defect_fraction,
        defect_size_mean=defect_size_mean,
        defect_size_distribution=defect_size_distribution,
        defect_size_std=defect_size_std,
        defect_threshold=defect_threshold,
        defect_shape=defect_shape,
        periodic=periodic_value,
        seed=seed,
        seed_entropy=seed_entropy,
    )
    field = generate_random_field(config)
    debug_enabled = ctx.obj.get("debug", False)
    verbose_enabled = ctx.obj.get("verbose", False)
    gb_stats = field.pop("gb_stats", {}) if isinstance(field.get("gb_stats"), dict) else {}
    seed_sequence = field.pop("seed_sequence", None)
    seed_meta = field.pop("_seed_meta", None)
    field.pop("theta_mean_used", None)
    field.pop("mixture_component", None)
    field.pop("mixture_weights", None)

    mask_ratio = _mask_ratio(field["mask"])
    grain_count = int(np.max(field["grain_id"])) + 1
    theta_min_cb = _range_bound(theta_min, is_min=True)
    theta_max_cb = _range_bound(theta_max, is_min=False)
    gb_max = float(gb_stats.get("gb_mis_max", 0.0))
    gb_hard_ok = gb_max <= float(gb_mis_hard_max or 0.0)

    if seed_meta is None:
        raise click.ClickException("seed metadata missing from generator output")
    seed_entropy_value = seed_meta.get("seed_entropy")
    seed_sequence_meta = seed_meta.get("seed_sequence")
    seed_user_value = seed_meta.get("seed_user")
    seed_effective = seed_user_value if seed_user_value is not None else seed_entropy_value
    meta_min = {
        "schema_version": "1.0",
        "source": "random",
        "dim": dim_value,
        "shape": list(shape_value),
        "angle_unit": "deg",
        "theta_reference": "+x",
        "theta_positive": "CCW",
        "theta_to_rotation": "Rz(theta)",
        "orientation_format": orientation_format,
        "quat_order": "(w,x,y,z)" if orientation_format == "quaternion" else None,
        "periodic": periodic_value,
        "mask_ratio": mask_ratio,
        "theta_min": theta_min_cb,
        "theta_max": theta_max_cb,
        "seed_user": seed_user_value,
        "seed": seed_user_value,
        "seed_entropy": seed_entropy_value,
        "seed_sequence": seed_sequence_meta,
        "seed_effective": seed_effective,
        "theta_min_input": float(theta_min),
        "theta_max_input": float(theta_max),
        "gb_mis_hard_max": gb_mis_hard_max,
        "gb_max": gb_max,
        "gb_hard_ok": gb_hard_ok,
        "grain_count": grain_count,
    }
    if seed_entropy_value is None or seed_effective is None:
        raise click.ClickException("seed_entropy and seed_effective must be present in meta")
    if mixture_components:
        meta_min["texture_mixture_components"] = mixture_components
        meta_min["texture_mixture_normalized_weights"] = mixture_weights

    field["meta"] = meta_min
    if debug_enabled:
        meta_debug = {
            "seed_sequence": seed_sequence,
            "gb_stats": gb_stats,
            "mask_diagnostics": _mask_diagnostics(field["mask"]),
        }
        field["meta_debug"] = meta_debug

    output_path = write_npz(field, output)

    click.echo(_summarize_random(meta_min, output_path))
    if verbose_enabled:
        click.echo(f"theta_range={theta_min_cb}..{theta_max_cb} gb_p95={gb_stats.get('gb_mis_p95', 0):.3f}")
        click.echo(f"orientation_format={orientation_format} periodic={periodic_value}")
    if debug_enabled:
        if seed_sequence:
            click.echo(f"debug seed_sequence={seed_sequence}")
        click.echo(f"debug mask={_mask_diagnostics(field['mask'])}")
        if "R" in field:
            ortho = _sample_orthogonality(field["R"])
            click.echo(f"debug R orthogonality max|RRT-I|: {ortho:.2e}")
        if "q" in field:
            norm_delta = _sample_quat_norm(field["q"])
            click.echo(f"debug q norm max|1-|q||: {norm_delta:.2e}")
        click.echo(
            "debug gb edges_total={total} wrap_x={wrap_x} wrap_y={wrap_y} wrap_z={wrap_z}".format(
                total=gb_stats.get("gb_edges_total", 0),
                wrap_x=gb_stats.get("gb_edges_wrap_x", 0),
                wrap_y=gb_stats.get("gb_edges_wrap_y", 0),
                wrap_z=gb_stats.get("gb_edges_wrap_z", 0),
            )
        )
    preview_target = preview_prefix if preview_prefix is not None else preview
    if preview_target is None:
        preview_target = _default_preview_prefix(output_path)
    if preview_target:
        prefix = preview_target.with_suffix("")
        vmin, vmax = resolve_global_vmin_vmax(field)
        theta = _theta_from_field(field)
        if theta is None:
            raise click.ClickException("theta_deg, R, or euler_deg required for preview")
        mask = field["mask"]
        if theta.ndim == 2:
            theta_slice = theta
            mask_slice = mask
        else:
            if slice_index == "mid":
                axis_map = {"x": 0, "y": 1, "z": 2}
                axis_index = axis_map.get(slice_axis, 2)
                index_value = theta.shape[axis_index] // 2
            else:
                index_value = int(slice_index)
            theta_slice = _get_slice(theta, slice_axis, index_value, slice_mode, face_policy, is_mask=False)
            mask_slice = _get_slice(mask, slice_axis, index_value, slice_mode, face_policy, is_mask=True)
        _render_preview_png(theta_slice, mask_slice, prefix, vmin, vmax)
        click.echo(MESSAGES[lang]["preview_done"].format(preview=prefix))


@cli.command("from-images", help="Invert a 2D orientation field from a color map image and color bar.")
@click.option("--lang", type=click.Choice(["zh", "en"]), default=None, help="Override global language.")
@click.option("--field-image", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--cbar-image", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--cbar-min", type=float, required=True)
@click.option("--cbar-max", type=float, required=True)
@click.option("--cbar-direction", type=click.Choice(["top=high", "top=low", "top_high", "top_low"]), default="top=high")
@click.option("--defect-threshold", type=int, default=8, show_default=True)
@click.option("--color-tolerance", type=float, default=50.0, show_default=True)
@click.option("--defect-policy", type=click.Choice(["nan", "zero", "fill"]), default="nan")
@click.option("--orientation-format", type=click.Choice(["euler", "matrix", "quaternion"]), default="matrix")
@click.option("--mask-min-area", type=int, default=9, show_default=True)
@click.option("--mask-morph", type=click.Choice(["none", "open", "close"]), default="open", show_default=True)
@click.option("--mask-kernel", type=int, default=3, show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--preview", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--preview-prefix", type=click.Path(path_type=Path), default=None)
@click.pass_context
def from_images_command(
    ctx: click.Context,
    lang: Optional[str],
    field_image: Path,
    cbar_image: Path,
    cbar_min: float,
    cbar_max: float,
    cbar_direction: str,
    defect_threshold: int,
    color_tolerance: float,
    defect_policy: str,
    orientation_format: str,
    mask_min_area: int,
    mask_morph: str,
    mask_kernel: int,
    output: Optional[Path],
    preview: Optional[Path],
    preview_prefix: Optional[Path],
) -> None:
    direction_value = "top_high" if cbar_direction in {"top=high", "top_high"} else "top_low"
    lang = _normalize_lang(_resolve_lang(ctx, lang))
    if output is None:
        output = _default_npz_output("image_field")

    config = FromImagesConfig(
        field_image=field_image,
        cbar_image=cbar_image,
        cbar_min=cbar_min,
        cbar_max=cbar_max,
        cbar_direction=direction_value,
        defect_threshold=defect_threshold,
        color_tolerance=color_tolerance,
        defect_policy=defect_policy,
        orientation_format=orientation_format,
        mask_min_area=mask_min_area,
        mask_morph=mask_morph,
        mask_kernel=mask_kernel,
    )
    field = invert_field_from_images(config)
    debug_enabled = ctx.obj.get("debug", False)
    verbose_enabled = ctx.obj.get("verbose", False)
    mask_raw = field.get("mask_raw", field["mask"])
    mask_ratio_raw = _mask_ratio(mask_raw)
    mask_ratio_final = _mask_ratio(field["mask"])
    removed_pixels = int(field.get("mask_removed_pixels", 0))
    theta_min_val = float(np.nanmin(field["theta_deg"]))
    theta_max_val = float(np.nanmax(field["theta_deg"]))

    meta_min = {
        "schema_version": "1.0",
        "source": "from-images",
        "dim": 2,
        "shape": list(field["theta_deg"].shape),
        "angle_unit": "deg",
        "theta_reference": "+x",
        "theta_positive": "CCW",
        "theta_to_rotation": "Rz(theta)",
        "orientation_format": orientation_format,
        "quat_order": "(w,x,y,z)" if orientation_format == "quaternion" else None,
        "periodic": "none",
        "mask_ratio": mask_ratio_final,
        "theta_min": theta_min_val,
        "theta_max": theta_max_val,
        "cbar_min": cbar_min,
        "cbar_max": cbar_max,
        "cbar_direction": direction_value,
        "defect_policy": defect_policy,
        "mask_params": {
            "defect_threshold": defect_threshold,
            "color_tolerance": color_tolerance,
            "mask_min_area": mask_min_area,
            "mask_morph": mask_morph,
            "mask_kernel": mask_kernel,
        },
    }

    field["meta"] = meta_min
    if debug_enabled:
        meta_debug = {
            "mask_ratio_raw": mask_ratio_raw,
            "mask_removed_pixels": removed_pixels,
            "mask_postprocess_applied": bool(field.get("mask_postprocess_applied", False)),
            "mask_diagnostics": _mask_diagnostics(field["mask"]),
        }
        field["meta_debug"] = meta_debug

    if not debug_enabled:
        field.pop("mask_raw", None)
        field.pop("mask_postprocess_applied", None)
        field.pop("mask_removed_pixels", None)

    output_path = write_npz(field, output)
    click.echo(_summarize_from_images(meta_min, output_path))
    if verbose_enabled:
        click.echo(f"mask_ratio_raw={mask_ratio_raw:.3f} removed_pixels={removed_pixels}")
        click.echo(f"orientation_format={orientation_format}")
    if debug_enabled:
        click.echo(f"debug mask={_mask_diagnostics(field['mask'])}")
        if "R" in field:
            ortho = _sample_orthogonality(field["R"])
            click.echo(f"debug R orthogonality max|RRT-I|: {ortho:.2e}")
        if "q" in field:
            norm_delta = _sample_quat_norm(field["q"])
            click.echo(f"debug q norm max|1-|q||: {norm_delta:.2e}")
    preview_target = preview_prefix if preview_prefix is not None else preview
    if preview_target is None:
        preview_target = _default_preview_prefix(output_path)
    if preview_target:
        prefix = preview_target.with_suffix("")
        vmin, vmax = resolve_global_vmin_vmax(field)
        theta = _theta_from_field(field)
        if theta is None:
            raise click.ClickException("theta_deg, R, or euler_deg required for preview")
        _render_preview_png(theta, field["mask"], prefix, vmin, vmax)
        click.echo(MESSAGES[lang]["preview_done"].format(preview=prefix))


@cli.command("preview", help="Render a quick preview from an NPZ output.")
@click.option("--lang", type=click.Choice(["zh", "en"]), default=None, help="Override global language.")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--preview-prefix", type=click.Path(path_type=Path), default=None)
@click.option("--slice-axis", type=click.Choice(["x", "y", "z"]), default="z")
@click.option("--slice-index", type=str, default="mid", show_default=True)
@click.option("--slice-mode", type=click.Choice(["voxel", "face"]), default="voxel", show_default=True)
@click.option("--face-policy", type=click.Choice(["avg", "left", "right"]), default="avg", show_default=True)
@click.option("--output-mode", type=click.Choice(["both", "field", "mask"]), default="both", show_default=True)
@click.pass_context
def preview_command(
    ctx: click.Context,
    lang: Optional[str],
    input_path: Path,
    output: Optional[Path],
    preview_prefix: Optional[Path],
    slice_axis: str,
    slice_index: str,
    slice_mode: str,
    face_policy: str,
    output_mode: str,
) -> None:
    preview_target = preview_prefix if preview_prefix is not None else output
    if preview_target is None:
        preview_target = _default_outdir("preview") / input_path.stem
    prefix = preview_target.with_suffix("")
    field = read_npz(input_path)
    vmin, vmax = resolve_global_vmin_vmax(field)
    mask = field.get("mask")
    if mask is None:
        raise click.ClickException("mask is required for preview")
    theta = field.get("theta_deg")
    matrix = field.get("R")
    if theta is None:
        if matrix is None:
            raise click.ClickException("theta_deg or R required for preview")
        if matrix.ndim <= 4:
            theta = _theta_deg_from_R(matrix)
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
            theta_slice = _get_slice(theta, slice_axis, index_value, slice_mode, face_policy, is_mask=False)
        mask_slice = _get_slice(mask, slice_axis, index_value, slice_mode, face_policy, is_mask=True)
    _render_preview_png(theta_slice, mask_slice, prefix, vmin, vmax, output_mode=output_mode)
    preview_path = prefix
    lang = _normalize_lang(_resolve_lang(ctx, lang))
    click.echo(MESSAGES[lang]["preview_done"].format(preview=preview_path))


@cli.command("render-slices", help="Batch render slices with colorbars.")
@click.option("--lang", type=click.Choice(["zh", "en"]), default=None, help="Override global language.")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--axis", type=click.Choice(["x", "y", "z"]), default="z", show_default=True)
@click.option("--slice-mode", type=click.Choice(["voxel", "face"]), default="voxel", show_default=True)
@click.option("--face-policy", type=click.Choice(["avg", "left", "right"]), default="avg", show_default=True)
@click.option("--range", "range_text", type=str, required=True, help="start:end (end not included)")
@click.option("--outdir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--prefix", type=str, default=None)
@click.option("--dpi", type=int, default=150, show_default=True)
@click.option("--output-mode", type=click.Choice(["both", "field", "mask"]), default="both", show_default=True)
@click.pass_context
def render_slices_command(
    ctx: click.Context,
    lang: Optional[str],
    input_path: Path,
    axis: str,
    slice_mode: str,
    face_policy: str,
    range_text: str,
    outdir: Optional[Path],
    prefix: Optional[str],
    dpi: int,
    output_mode: str,
) -> None:
    if outdir is None:
        outdir = _default_outdir("slices")
    field = read_npz(input_path)
    theta = _theta_from_field(field)
    if theta is None:
        raise click.ClickException("theta_deg, R, or euler_deg required for render-slices")
    outdir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = f"slice_{axis}_"
    try:
        start_text, end_text = range_text.split(":")
        start = int(start_text)
        end = int(end_text)
    except ValueError:
        raise click.BadParameter("range must be start:end")

    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_index = axis_map[axis]
    shape = field["shape"] if isinstance(field["shape"], np.ndarray) else np.array(field["shape"])
    axis_len = int(shape[axis_index])
    max_index = axis_len if slice_mode == "face" else axis_len - 1
    if start < 0 or end < 0 or end > max_index + 1:
        raise click.BadParameter(f"range out of bounds for axis length {axis_len} and mode {slice_mode}")

    indices = list(range(start, end))
    render_slices_batch(
        field,
        axis,
        indices,
        outdir,
        prefix,
        slice_mode=slice_mode,
        face_policy=face_policy,
        dpi=dpi,
        output_mode=output_mode,
    )
    lang = _normalize_lang(_resolve_lang(ctx, lang))
    click.echo(MESSAGES[lang]["preview_done"].format(preview=outdir))


@cli.command("export", help="Export NPZ to JSON/CSV/HDF5.")
@click.option("--lang", type=click.Choice(["zh", "en"]), default=None, help="Override global language.")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--formats", type=str, required=True, help="comma-separated formats: json,csv,h5")
@click.option("--outdir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--json-data", type=click.Choice(["none", "theta", "all"]), default="none", show_default=True)
@click.option("--json-downsample", type=int, default=1, show_default=True)
@click.option("--csv-mode", type=click.Choice(["auto", "2d", "slice"]), default="auto", show_default=True)
@click.option("--csv-slice-axis", type=click.Choice(["x", "y", "z"]), default=None)
@click.option("--csv-slice-index", type=str, default=None)
@click.option("--yes", is_flag=True, default=False, help="Skip interactive confirmation.")
@click.pass_context
def export_command(
    ctx: click.Context,
    lang: Optional[str],
    input_path: Path,
    formats: str,
    outdir: Optional[Path],
    json_data: str,
    json_downsample: int,
    csv_mode: str,
    csv_slice_axis: Optional[str],
    csv_slice_index: Optional[str],
    yes: bool,
) -> None:
    if outdir is None:
        outdir = _default_outdir("exports")
    format_list = [item.strip() for item in formats.split(",") if item.strip()]
    slice_index: Optional[int | str]
    if csv_slice_index is None:
        slice_index = None
    elif csv_slice_index == "mid":
        slice_index = "mid"
    else:
        slice_index = int(csv_slice_index)
    confirm_msg = f"Export {input_path.name} -> {outdir} formats={','.join(format_list)}. Continue?"
    if not _confirm_export_cli(ctx, yes, confirm_msg):
        return
    outputs = export_npz(
        input_path,
        outdir,
        format_list,
        json_data=json_data,
        json_downsample=json_downsample,
        csv_mode=csv_mode,
        csv_slice_axis=csv_slice_axis,
        csv_slice_index=slice_index,
    )
    click.echo(_summarize_export(input_path, outputs))
    if ctx.obj.get("verbose", False):
        click.echo(f"outputs={outputs}")


@cli.command("render-orthoslices", help="Render orthogonal x/y/z slices with shared colorbar.")
@click.option("--lang", type=click.Choice(["zh", "en"]), default=None, help="Override global language.")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--outdir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--index", type=str, default="mid", show_default=True)
@click.option("--dpi", type=int, default=150, show_default=True)
@click.pass_context
def render_orthoslices_command(
    ctx: click.Context,
    lang: Optional[str],
    input_path: Path,
    outdir: Optional[Path],
    index: str,
    dpi: int,
) -> None:
    if outdir is None:
        outdir = _default_outdir("orthoslices")
    field = read_npz(input_path)
    theta = _theta_from_field(field)
    if theta is None:
        raise click.ClickException("theta_deg, R, or euler_deg required for render-orthoslices")
    outdir.mkdir(parents=True, exist_ok=True)
    render_orthoslices(field, index, outdir, dpi=dpi)
    lang = _normalize_lang(_resolve_lang(ctx, lang))
    click.echo(MESSAGES[lang]["preview_done"].format(preview=outdir))


if __name__ == "__main__":
    cli()

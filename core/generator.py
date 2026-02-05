"""
Module: core/generator.py
Responsibility:
  - Generate random orientation fields with grain topology and defects.
  - Enforce misorientation constraints and assemble output field dictionaries.

Public API:
  - RandomFieldConfig
  - generate_random_field(config)
  - build_meta(config, source, seed_meta=None)
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from core.representation import (
    rotation_matrix_to_quaternion,
    rotation_matrix_z,
)


__all__ = ["RandomFieldConfig", "generate_random_field", "build_meta"]


@dataclass(frozen=True)
class RandomFieldConfig:
    dim: int
    shape: Tuple[int, ...]
    grain_size_mean: float
    grain_size_distribution: str
    grain_size_std: float
    orientation_distribution: str
    mean_theta_deg: float
    orientation_std_deg: float
    orientation_format: str
    theta_min: Optional[float]
    theta_max: Optional[float]
    theta_boundary_policy: str
    texture_mixture_components: Optional[list[tuple[float, float]]]
    texture_mixture_weights: Optional[list[float]]
    gb_mis_hard_max: Optional[float]
    gb_mis_p95: Optional[float]
    gb_mis_sampler: str
    gb_mis_std: Optional[float]
    gb_alpha: float
    defect_enabled: bool
    defect_fraction: float
    defect_size_mean: float
    defect_size_distribution: str
    defect_size_std: float
    defect_threshold: float
    defect_shape: str
    periodic: str
    texture_mixture: Optional[str] = None
    seed: Optional[int] = None
    seed_entropy: Optional[int] = None


def _validate_config(config: RandomFieldConfig) -> None:
    if config.dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    if len(config.shape) != config.dim:
        raise ValueError("shape must match dim")
    if config.grain_size_mean <= 0:
        raise ValueError("grain_size_mean must be positive")
    if config.grain_size_distribution not in {"uniform", "normal", "lognormal"}:
        raise ValueError("grain_size_distribution must be uniform/normal/lognormal")
    if config.orientation_distribution not in {"random", "normal"}:
        raise ValueError("orientation_distribution must be random/normal")
    if config.orientation_format not in {"euler", "matrix", "quaternion"}:
        raise ValueError("orientation_format must be euler/matrix/quaternion")
    if config.theta_min is None or config.theta_max is None:
        raise ValueError("theta_min/theta_max are required for random generation")
    if config.theta_min >= config.theta_max:
        raise ValueError("theta_min must be less than theta_max")
    if config.theta_boundary_policy not in {"truncate", "clip"}:
        raise ValueError("theta_boundary_policy must be truncate or clip")
    if config.defect_fraction < 0 or config.defect_fraction > 1:
        raise ValueError("defect_fraction must be in [0,1]")
    if config.defect_size_mean <= 0:
        raise ValueError("defect_size_mean must be positive")
    if config.defect_size_distribution not in {"uniform", "normal", "lognormal"}:
        raise ValueError("defect_size_distribution must be uniform/normal/lognormal")
    if config.dim == 2 and config.defect_shape not in {"disk", "ellipse", "blob"}:
        raise ValueError("defect_shape must be disk/ellipse/blob for 2D")
    if config.dim == 3 and config.defect_shape not in {"sphere", "ellipsoid", "blob"}:
        raise ValueError("defect_shape must be sphere/ellipsoid/blob for 3D")
    if config.gb_mis_hard_max is None:
        raise ValueError("gb_mis_hard_max is required")
    if config.gb_alpha < 0 or config.gb_alpha > 1:
        raise ValueError("gb_alpha must be in [0,1]")
    if config.gb_mis_sampler not in {"normal", "mixture"}:
        raise ValueError("gb_mis_sampler must be normal or mixture")
    if config.gb_mis_std is not None and config.gb_mis_std <= 0:
        raise ValueError("gb_mis_std must be positive")
    if config.texture_mixture_components:
        if not config.texture_mixture_weights:
            raise ValueError("texture_mixture_weights are required with texture_mixture_components")
        if len(config.texture_mixture_components) != len(config.texture_mixture_weights):
            raise ValueError("texture_mixture_components and texture_mixture_weights must have same length")
        for mean, std in config.texture_mixture_components:
            if std <= 0:
                raise ValueError("texture mixture std must be positive")
        for weight in config.texture_mixture_weights:
            if weight <= 0:
                raise ValueError("texture mixture weight must be positive")
    if config.texture_mixture_weights and not config.texture_mixture_components:
        raise ValueError("texture_mixture_components are required with texture_mixture_weights")
    if config.gb_mis_sampler == "mixture" and not config.texture_mixture_components:
        raise ValueError(
            "texture_mixture_components are required when gb_mis_sampler is mixture (CLI: --texture-mixture)"
        )


def _sample_radii(
    rng: np.random.Generator,
    count: int,
    mean: float,
    std: float,
    distribution: str,
) -> np.ndarray:
    if distribution == "uniform":
        radii = np.full(count, mean)
    elif distribution == "normal":
        radii = rng.normal(loc=mean, scale=std, size=count)
    else:
        sigma = max(std / max(mean, 1e-6), 1e-6)
        mu = np.log(max(mean, 1e-6))
        radii = rng.lognormal(mean=mu, sigma=sigma, size=count)
    return np.clip(radii, mean * 0.2, None)


def _resolve_theta_mean(
    mean_theta: float, theta_min: Optional[float], theta_max: Optional[float]
) -> Tuple[float, bool]:
    if theta_min is None or theta_max is None:
        return mean_theta, False
    if mean_theta < theta_min:
        return theta_min, True
    if mean_theta > theta_max:
        return theta_max, True
    return mean_theta, False


def _angular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def _build_adjacency(
    grain_map: np.ndarray, periodic: str
) -> tuple[list[tuple[int, int, Optional[str]]], dict]:
    edges: set[tuple[int, int, Optional[str]]] = set()
    wrap_counts = {"x": 0, "y": 0, "z": 0}
    dims = grain_map.ndim
    # neighbor adjacency within the grid (non-periodic)
    axis_labels = ["x", "y", "z"]
    for axis in range(dims):
        slicer_a = [slice(None)] * dims
        slicer_b = [slice(None)] * dims
        slicer_a[axis] = slice(1, None)
        slicer_b[axis] = slice(None, -1)
        a = grain_map[tuple(slicer_a)]
        b = grain_map[tuple(slicer_b)]
        mask = a != b
        if np.any(mask):
            pairs_a = a[mask].ravel()
            pairs_b = b[mask].ravel()
            for i, j in zip(pairs_a.tolist(), pairs_b.tolist()):
                if i == j:
                    continue
                edge = (i, j, None) if i < j else (j, i, None)
                edges.add(edge)
        axis_label = axis_labels[axis]
        if axis_label in periodic:
            slicer_wrap_a = [slice(None)] * dims
            slicer_wrap_b = [slice(None)] * dims
            slicer_wrap_a[axis] = 0
            slicer_wrap_b[axis] = -1
            wrap_a = grain_map[tuple(slicer_wrap_a)]
            wrap_b = grain_map[tuple(slicer_wrap_b)]
            wrap_mask = wrap_a != wrap_b
            if np.any(wrap_mask):
                pairs_a = wrap_a[wrap_mask].ravel()
                pairs_b = wrap_b[wrap_mask].ravel()
                for i, j in zip(pairs_a.tolist(), pairs_b.tolist()):
                    if i == j:
                        continue
                    edge = (i, j, axis_label) if i < j else (j, i, axis_label)
                    if edge not in edges:
                        wrap_counts[axis_label] += 1
                    edges.add(edge)
    wrap_counts["total"] = len(edges)
    return list(edges), wrap_counts


def _edges_to_adjacency(edges: list[tuple[int, int, Optional[str]]], grain_count: int) -> list[list[int]]:
    adjacency = [set() for _ in range(grain_count)]
    for i, j, _ in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    return [sorted(list(neighbors)) for neighbors in adjacency]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _constraint_intervals(theta: float, hard_max: float, theta_min: float, theta_max: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for k in (-1, 0, 1):
        low = theta - hard_max + 360.0 * k
        high = theta + hard_max + 360.0 * k
        start = max(low, theta_min)
        end = min(high, theta_max)
        if start <= end:
            intervals.append((start, end))
    return _merge_intervals(intervals)


def _intersect_intervals(
    base: list[tuple[float, float]], other: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    if not base or not other:
        return []
    result: list[tuple[float, float]] = []
    for a_start, a_end in base:
        for b_start, b_end in other:
            start = max(a_start, b_start)
            end = min(a_end, b_end)
            if start <= end:
                result.append((start, end))
    return _merge_intervals(result)


def _sample_texture_mixture(
    rng: np.random.Generator,
    components: list[tuple[float, float]],
    weights: list[float],
) -> tuple[float, int]:
    means = np.array([m for m, _ in components], dtype=np.float32)
    stds = np.array([s for _, s in components], dtype=np.float32)
    weight_array = np.array(weights, dtype=np.float32)
    weight_array = weight_array / np.sum(weight_array)
    component = int(rng.choice(len(components), p=weight_array))
    theta = float(rng.normal(means[component], stds[component]))
    return theta, component


def _sample_from_intervals(
    rng: np.random.Generator,
    intervals: list[tuple[float, float]],
    config: RandomFieldConfig,
    max_attempts: int = 200,
) -> tuple[float, Optional[int]]:
    if not intervals:
        raise ValueError("No feasible intervals for sampling")
    component = None
    for _ in range(max_attempts):
        if config.texture_mixture_components and config.gb_mis_sampler == "mixture":
            candidate, component = _sample_texture_mixture(
                rng,
                config.texture_mixture_components,
                config.texture_mixture_weights or [],
            )
        elif config.orientation_distribution == "normal":
            std = config.gb_mis_std if config.gb_mis_std is not None else config.orientation_std_deg
            candidate = float(rng.normal(config.mean_theta_deg, std))
        else:
            candidate = float(rng.uniform(config.theta_min, config.theta_max))
        for start, end in intervals:
            if start <= candidate <= end:
                return candidate, component
    interval_lengths = np.array([end - start for start, end in intervals], dtype=np.float32)
    interval_weights = interval_lengths / np.sum(interval_lengths)
    index = int(rng.choice(len(intervals), p=interval_weights))
    start, end = intervals[index]
    return float(rng.uniform(start, end)), component


def _compute_gb_stats(
    theta_grain: np.ndarray,
    edges: list[tuple[int, int, Optional[str]]],
    hard_max: float,
    target_p95: Optional[float],
    wrap_counts: dict,
) -> dict:
    if not edges:
        return {
            "gb_mis_max": 0.0,
            "gb_mis_mean": 0.0,
            "gb_mis_p95": 0.0,
            "gb_mis_hard_max": hard_max,
            "gb_mis_target_p95": target_p95,
            "gb_converged": True,
            "gb_iterations": 1,
            "gb_edges": 0,
            "gb_edges_total": wrap_counts.get("total", 0),
            "gb_edges_wrap_x": wrap_counts.get("x", 0),
            "gb_edges_wrap_y": wrap_counts.get("y", 0),
            "gb_edges_wrap_z": wrap_counts.get("z", 0),
        }
    deltas = np.array([_angular_difference(theta_grain[i], theta_grain[j]) for i, j, _ in edges], dtype=np.float32)
    return {
        "gb_mis_max": float(np.max(deltas)),
        "gb_mis_mean": float(np.mean(deltas)),
        "gb_mis_p95": float(np.percentile(deltas, 95)),
        "gb_mis_hard_max": hard_max,
        "gb_mis_target_p95": target_p95,
        "gb_converged": True,
        "gb_iterations": 1,
        "gb_edges": len(edges),
        "gb_edges_total": wrap_counts.get("total", len(edges)),
        "gb_edges_wrap_x": wrap_counts.get("x", 0),
        "gb_edges_wrap_y": wrap_counts.get("y", 0),
        "gb_edges_wrap_z": wrap_counts.get("z", 0),
    }

def _periodic_delta(delta: np.ndarray, size: int) -> np.ndarray:
    return np.minimum(delta, size - delta)


# Grain-level orientation assignment
# Hard GB constraint enforcement
def _assign_grain_orientations_with_gb(
    config: RandomFieldConfig,
    rng: np.random.Generator,
    grain_count: int,
    adjacency: list[list[int]],
    edges: list[tuple[int, int, Optional[str]]],
    wrap_counts: dict,
) -> tuple[np.ndarray, dict, Optional[np.ndarray], Optional[np.ndarray]]:
    theta_min = float(config.theta_min)
    theta_max = float(config.theta_max)
    hard_max = float(config.gb_mis_hard_max)
    target_p95 = config.gb_mis_p95
    max_attempts_per_grain = 80
    max_regen = 6

    mixture_weights = None
    if config.texture_mixture_weights:
        weights = np.array(config.texture_mixture_weights, dtype=np.float32)
        mixture_weights = weights / np.sum(weights)

    def build_order(root: int) -> list[int]:
        order = []
        seen = np.zeros(grain_count, dtype=bool)
        queue = [root]
        seen[root] = True
        while queue:
            current = queue.pop(0)
            order.append(current)
            for nbr in adjacency[current]:
                if not seen[nbr]:
                    seen[nbr] = True
                    queue.append(nbr)
        for idx in range(grain_count):
            if not seen[idx]:
                order.append(idx)
        return order

    theta_grain = np.full(grain_count, np.nan, dtype=np.float32)
    for regen in range(max_regen):
        root = int(rng.integers(0, grain_count))
        order = build_order(root)
        theta_grain[:] = np.nan
        component = np.full(grain_count, -1, dtype=np.int32)
        attempts = np.zeros(grain_count, dtype=np.int32)
        i = 0
        while i < len(order):
            grain = order[i]
            if attempts[grain] >= max_attempts_per_grain:
                theta_grain[grain] = np.nan
                component[grain] = -1
                attempts[grain] = 0
                i -= 1
                if i < 0:
                    break
                continue
            intervals = [(theta_min, theta_max)]
            assigned_neighbors = [nbr for nbr in adjacency[grain] if not np.isnan(theta_grain[nbr])]
            for nbr in assigned_neighbors:
                nbr_intervals = _constraint_intervals(theta_grain[nbr], hard_max, theta_min, theta_max)
                intervals = _intersect_intervals(intervals, nbr_intervals)
                if not intervals:
                    break
            if not intervals:
                attempts[grain] = max_attempts_per_grain
                continue
            sample, comp = _sample_from_intervals(rng, intervals, config)
            theta_grain[grain] = sample
            component[grain] = comp if comp is not None else -1
            attempts[grain] += 1
            i += 1
        if np.any(np.isnan(theta_grain)):
            continue
        theta_grain = np.clip(theta_grain, theta_min, theta_max)
        gb_stats = _compute_gb_stats(theta_grain, edges, hard_max, target_p95, wrap_counts)
        gb_stats["gb_iterations"] = regen + 1
        if target_p95 is not None and gb_stats["gb_mis_p95"] > target_p95:
            gb_stats["gb_converged"] = False
            continue
        gb_stats["gb_converged"] = True
        if np.any(component >= 0):
            return theta_grain, gb_stats, component, mixture_weights
        return theta_grain, gb_stats, None, mixture_weights

    raise RuntimeError("Failed to satisfy GB p95 target while enforcing hard constraints")


def _generate_grain_map(config: RandomFieldConfig, rng: np.random.Generator) -> np.ndarray:
    total = np.prod(config.shape)
    if config.dim == 2:
        grain_count = max(1, int(total / (config.grain_size_mean**2)))
    else:
        grain_count = max(1, int(total / (config.grain_size_mean**3)))

    seeds = rng.integers(0, np.array(config.shape), size=(grain_count, config.dim))
    radii = _sample_radii(rng, grain_count, config.grain_size_mean, config.grain_size_std, config.grain_size_distribution)

    grid = np.indices(config.shape)
    distances = np.zeros((*config.shape, grain_count), dtype=np.float32)
    for axis in range(config.dim):
        delta = np.abs(grid[axis][..., None] - seeds[:, axis])
        if (axis == 0 and "x" in config.periodic) or (axis == 1 and "y" in config.periodic) or (
            axis == 2 and "z" in config.periodic
        ):
            delta = _periodic_delta(delta, config.shape[axis])
        distances += (delta / radii) ** 2
    grain_map = np.argmin(distances, axis=-1).astype(np.int32)
    return grain_map


def _build_orientation_data(
    config: RandomFieldConfig,
    theta_grain: np.ndarray,
) -> Dict[str, np.ndarray]:
    if config.orientation_format == "matrix":
        return {"R": rotation_matrix_z(theta_grain)}
    if config.orientation_format == "quaternion":
        return {"q": rotation_matrix_to_quaternion(rotation_matrix_z(theta_grain))}
    euler = np.zeros((theta_grain.shape[0], 3), dtype=np.float32)
    euler[:, 0] = theta_grain
    return {"euler_deg": euler}


def _apply_defects(config: RandomFieldConfig, rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    if not config.defect_enabled or config.defect_fraction <= 0:
        return mask
    total = mask.size
    target = int(total * config.defect_fraction)
    current = 0
    grid = np.indices(config.shape).astype(np.float32)
    while current < target:
        center = rng.uniform(0, np.array(config.shape), size=(config.dim,))
        radius = _sample_radii(
            rng, 1, config.defect_size_mean, config.defect_size_std, config.defect_size_distribution
        )[0]
        if config.defect_shape in {"disk", "sphere"}:
            scales = np.full(config.dim, radius)
        elif config.defect_shape in {"ellipse", "ellipsoid"}:
            axis_ratio = rng.uniform(1.0, 3.0, size=config.dim)
            scales = radius * axis_ratio
        else:
            scales = np.full(config.dim, radius * 0.6)

        dist = np.zeros(config.shape, dtype=np.float32)
        for axis in range(config.dim):
            delta = np.abs(grid[axis] - center[axis])
            if (axis == 0 and "x" in config.periodic) or (axis == 1 and "y" in config.periodic) or (
                axis == 2 and "z" in config.periodic
            ):
                delta = _periodic_delta(delta, config.shape[axis])
            dist += (delta / scales[axis]) ** 2
        defect = dist <= config.defect_threshold
        if config.defect_shape == "blob":
            blob_count = rng.integers(3, 7)
            for _ in range(blob_count):
                offset = rng.normal(0.0, radius * 0.5, size=config.dim)
                blob_center = (center + offset) % np.array(config.shape)
                blob_radius = radius * rng.uniform(0.3, 0.8)
                blob_dist = np.zeros(config.shape, dtype=np.float32)
                for axis in range(config.dim):
                    delta = np.abs(grid[axis] - blob_center[axis])
                    if (axis == 0 and "x" in config.periodic) or (axis == 1 and "y" in config.periodic) or (
                        axis == 2 and "z" in config.periodic
                    ):
                        delta = _periodic_delta(delta, config.shape[axis])
                    blob_dist += (delta / blob_radius) ** 2
                defect = np.logical_or(defect, blob_dist <= config.defect_threshold)
        newly = np.logical_and(mask == 1, defect)
        mask[newly] = 0
        current += int(np.sum(newly))
    return mask


def generate_random_field(config: RandomFieldConfig) -> Dict[str, np.ndarray]:
    _validate_config(config)
    if config.seed is not None:
        seed_sequence = np.random.SeedSequence(config.seed)
        root_entropy = int(seed_sequence.entropy)
    elif config.seed_entropy is not None:
        seed_sequence = np.random.SeedSequence(config.seed_entropy)
        root_entropy = int(seed_sequence.entropy)
    else:
        root_entropy = secrets.randbits(128)
        seed_sequence = np.random.SeedSequence(root_entropy)
    spawned = seed_sequence.spawn(3)
    rng_topo, rng_orient, rng_defect = [np.random.default_rng(seq) for seq in spawned]
    seed_streams = {
        "topology": int(spawned[0].generate_state(1, dtype=np.uint64)[0]),
        "orientation": int(spawned[1].generate_state(1, dtype=np.uint64)[0]),
        "defect": int(spawned[2].generate_state(1, dtype=np.uint64)[0]),
    }
    seed_meta = {
        "seed_user": config.seed,
        "seed_entropy": root_entropy,
        "seed_sequence": seed_streams,
    }
    grain_map = _generate_grain_map(config, rng_topo)
    grain_count = grain_map.max() + 1
    edges, wrap_counts = _build_adjacency(grain_map, config.periodic)
    adjacency = _edges_to_adjacency(edges, grain_count)
    theta_grain, gb_stats, mixture_component, mixture_weights = _assign_grain_orientations_with_gb(
        config, rng_orient, grain_count, adjacency, edges, wrap_counts
    )
    if np.any(np.isnan(theta_grain)):
        raise RuntimeError("Failed to assign grain orientations with hard GB constraints")
    if gb_stats["gb_mis_max"] > float(config.gb_mis_hard_max) + 1e-6:
        raise RuntimeError("Grain-boundary misorientation exceeds hard max")
    mean_used, mean_clamped = _resolve_theta_mean(config.mean_theta_deg, config.theta_min, config.theta_max)
    orientation_data = _build_orientation_data(config, theta_grain)

    mask = np.ones(config.shape, dtype=np.uint8)
    mask = _apply_defects(config, rng_defect, mask)

    field: Dict[str, np.ndarray] = {
        "dim": np.array(config.dim, dtype=np.int32),
        "shape": np.array(config.shape, dtype=np.int32),
        "mask": mask,
        "grain_id": grain_map,
    }

    for key, value in orientation_data.items():
        if isinstance(value, np.ndarray) and value.shape[0] == grain_count:
            field[key] = value[grain_map]
        else:
            field[key] = value

    if mean_clamped:
        field["theta_mean_used"] = np.array(mean_used, dtype=np.float32)
    field["gb_stats"] = gb_stats
    if mixture_component is not None:
        field["mixture_component"] = mixture_component
        if mixture_weights is not None:
            field["mixture_weights"] = mixture_weights
    field["seed_sequence"] = {
        "seed_user": config.seed,
        "seed_entropy": root_entropy,
        "seed_topology": seed_streams["topology"],
        "seed_orientation": seed_streams["orientation"],
        "seed_defect": seed_streams["defect"],
        "seed_streams": seed_streams,
    }
    field["_seed_meta"] = seed_meta
    return field


def build_meta(
    config: RandomFieldConfig,
    source: str,
    seed_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    meta = {
        "schema_version": "1.0",
        "source": source,
        "dim": config.dim,
        "shape": config.shape,
        "angle_unit": "deg",
        "theta_reference": "+x",
        "theta_positive": "CCW",
        "theta_to_rotation": "Rz(theta)",
        "periodic": config.periodic,
        "orientation_format": config.orientation_format,
        "quat_order": "(w,x,y,z)" if config.orientation_format == "quaternion" else None,
        "theta_min": config.theta_min,
        "theta_max": config.theta_max,
        "theta_min_input": config.theta_min,
        "theta_max_input": config.theta_max,
        "gb_mis_hard_max": config.gb_mis_hard_max,
        "seed_user": config.seed,
    }
    if config.texture_mixture_components:
        meta["texture_mixture_components"] = config.texture_mixture_components
        meta["texture_mixture_normalized_weights"] = config.texture_mixture_weights
    if seed_meta is not None:
        meta.update(seed_meta)
    return meta

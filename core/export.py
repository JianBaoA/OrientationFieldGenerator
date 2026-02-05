"""
Module: core/export.py
Responsibility:
  - Export NPZ orientation fields to JSON/CSV/HDF5 formats.
  - Provide helpers for CLI/GUI export commands.

Public API:
  - export_npz(input_path, outdir, formats, ...)
  - export_field(npz_path, fmt, out_path, options=None)
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


NPZ_META_KEY = "meta_json"
NPZ_META_DEBUG_KEY = "meta_debug_json"
__all__ = [
    "export_npz",
    "export_field",
]


def _load_json_from_npz(data: np.lib.npyio.NpzFile, key: str) -> Any:
    raw = data[key]
    value = raw.item() if hasattr(raw, "item") else raw
    if not isinstance(value, str):
        value = str(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON content for NPZ key: {key}") from exc


def read_npz_field(npz_path: Path) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        meta = _load_json_from_npz(data, NPZ_META_KEY)
        field = {key: data[key] for key in data.files if key not in {NPZ_META_KEY, NPZ_META_DEBUG_KEY}}
        field["meta"] = meta
        if NPZ_META_DEBUG_KEY in data.files:
            field["meta_debug"] = _load_json_from_npz(data, NPZ_META_DEBUG_KEY)
        return field


def _field_shape(field: Dict[str, Any]) -> List[int]:
    shape = field.get("shape")
    if shape is None:
        return list(field["mask"].shape)
    if isinstance(shape, np.ndarray):
        return [int(value) for value in shape.tolist()]
    return [int(value) for value in shape]


def _field_dim(field: Dict[str, Any]) -> int:
    dim = field.get("dim")
    if dim is None:
        return len(_field_shape(field))
    if isinstance(dim, np.ndarray):
        return int(dim.item())
    return int(dim)


def _theta_from_R(matrix: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]))


def _dtype_summary(field: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for key, value in field.items():
        if key == "meta":
            continue
        if isinstance(value, np.ndarray):
            summary[key] = {"dtype": str(value.dtype), "shape": list(value.shape)}
    return summary


def _fields_present(field: Dict[str, Any]) -> List[str]:
    fields = []
    for key in ("R", "q", "theta_deg", "euler_deg", "grain_id", "mask"):
        if key in field:
            fields.append(key)
    return fields


def export_to_json(
    field: Dict[str, Any],
    out_json: Path,
    data_policy: str = "none",
    downsample: int = 1,
    size_warn: int = 1_000_000,
) -> Path:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "meta": field["meta"],
        "dim": _field_dim(field),
        "shape": _field_shape(field),
        "fields_present": _fields_present(field),
        "dtype_summary": _dtype_summary(field),
    }

    if data_policy != "none":
        data: Dict[str, Any] = {}
        keys: Iterable[str]
        if data_policy == "theta":
            keys = ("theta_deg",)
        elif data_policy == "all":
            keys = [key for key in field.keys() if key != "meta"]
        else:
            raise ValueError("json data_policy must be none|theta|all")

        for key in keys:
            if key not in field:
                continue
            array = field[key]
            if isinstance(array, np.ndarray):
                array_to_dump = array[::downsample] if array.ndim == 1 else array[::downsample, ...]
                if array_to_dump.size > size_warn:
                    print(f"warning: dumping large array {key} with size {array_to_dump.size}")
                data[key] = array_to_dump.tolist()
        payload["data"] = data

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json


def _slice_index_value(size: int, slice_index: Optional[int | str]) -> int:
    if slice_index is None or slice_index == "mid":
        return size // 2
    return int(slice_index)


def export_to_csv(
    field: Dict[str, Any],
    out_csv: Path,
    mode: str = "auto",
    slice_axis: Optional[str] = None,
    slice_index: Optional[int | str] = None,
) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dim = _field_dim(field)
    shape = _field_shape(field)

    if mode == "auto":
        mode = "2d" if dim == 2 else "slice"

    if mode == "2d":
        if dim != 2:
            raise ValueError("2d CSV export requires dim==2")
        axis_index = None
        slice_idx = None
    elif mode == "slice":
        if dim != 3:
            raise ValueError("slice CSV export requires dim==3")
        if slice_axis is None:
            raise ValueError("slice_axis is required for 3D CSV export")
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_index = axis_map.get(slice_axis)
        if axis_index is None:
            raise ValueError("slice_axis must be x|y|z")
        slice_idx = _slice_index_value(shape[axis_index], slice_index)
    else:
        raise ValueError("mode must be auto|2d|slice")

    theta = field.get("theta_deg")
    theta_note = None
    if theta is None and "R" in field:
        theta = _theta_from_R(field["R"])
        theta_note = "theta_deg derived from R via atan2(R[...,1,0], R[...,0,0])"
    elif theta is None and "euler_deg" in field:
        theta = field["euler_deg"][..., 0]
        theta_note = "theta_deg derived from euler_deg[...,0]"
    elif theta is None and "q" in field:
        theta_note = "theta_deg not exported: quaternion present without theta or R"

    mask = field.get("mask")
    grain_id = field.get("grain_id")

    if mode == "2d":
        rows = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                row = {
                    "i": i,
                    "j": j,
                    "mask": int(mask[i, j]) if mask is not None else 1,
                    "grain_id": int(grain_id[i, j]) if grain_id is not None else -1,
                }
                if theta is not None:
                    row["theta_deg"] = float(theta[i, j])
                rows.append(row)
    else:
        rows = []
        if axis_index == 0:
            mask_slice = mask[slice_idx, :, :] if mask is not None else None
            grain_slice = grain_id[slice_idx, :, :] if grain_id is not None else None
            theta_slice = theta[slice_idx, :, :] if theta is not None else None
        elif axis_index == 1:
            mask_slice = mask[:, slice_idx, :] if mask is not None else None
            grain_slice = grain_id[:, slice_idx, :] if grain_id is not None else None
            theta_slice = theta[:, slice_idx, :] if theta is not None else None
        else:
            mask_slice = mask[:, :, slice_idx] if mask is not None else None
            grain_slice = grain_id[:, :, slice_idx] if grain_id is not None else None
            theta_slice = theta[:, :, slice_idx] if theta is not None else None

        shape_u, shape_v = mask_slice.shape if mask_slice is not None else grain_slice.shape
        for u in range(shape_u):
            for v in range(shape_v):
                row = {
                    "u": u,
                    "v": v,
                    "mask": int(mask_slice[u, v]) if mask_slice is not None else 1,
                    "grain_id": int(grain_slice[u, v]) if grain_slice is not None else -1,
                }
                if theta_slice is not None:
                    row["theta_deg"] = float(theta_slice[u, v])
                rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["i", "j", "mask", "grain_id"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    meta_payload = {
        "meta": field["meta"],
        "dim": dim,
        "shape": shape,
        "slice": None,
        "computed_fields": theta_note,
    }
    if mode == "slice":
        meta_payload["slice"] = {
            "axis": slice_axis,
            "index": int(slice_idx),
        }
    out_meta = out_csv.with_suffix(out_csv.suffix + ".meta.json")
    out_meta.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_csv


def export_to_hdf5(field: Dict[str, Any], out_h5: Path) -> Path:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for HDF5 export (pip install h5py)") from exc

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(field["meta"], ensure_ascii=False)

    with h5py.File(out_h5, "w") as h5:
        group_field = h5.create_group("field")
        for key in ("R", "q", "theta_deg", "euler_deg", "mask", "grain_id"):
            if key not in field:
                continue
            data = field[key]
            if isinstance(data, np.ndarray):
                dtype = np.float32 if data.dtype.kind == "f" else data.dtype
                dset = group_field.create_dataset(
                    key,
                    data=data.astype(dtype, copy=False),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    chunks=True,
                )
                if key == "theta_deg":
                    dset.attrs["units"] = "deg"
                if key == "mask":
                    dset.attrs["mask_semantics"] = "1=valid,0=defect"

        group_meta = h5.create_group("meta")
        dt = h5py.string_dtype(encoding="utf-8")
        group_meta.create_dataset("json", data=meta_json, dtype=dt)
        if "meta_debug" in field:
            group_meta.create_dataset("debug_json", data=json.dumps(field["meta_debug"], ensure_ascii=False), dtype=dt)
        dim_value = field.get("dim")
        if isinstance(dim_value, np.generic):
            dim_value = dim_value.item()
        if isinstance(dim_value, (int, float, bool)):
            group_meta.attrs["dim"] = int(dim_value)
        shape_value = _field_shape(field)
        group_meta.attrs["ndim"] = int(len(shape_value))

    return out_h5


def export_npz(
    npz_path: Path,
    out_dir: Path,
    formats: List[str],
    json_data: str = "none",
    json_downsample: int = 1,
    csv_mode: str = "auto",
    csv_slice_axis: Optional[str] = None,
    csv_slice_index: Optional[int | str] = None,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    field = read_npz_field(npz_path)
    outputs: Dict[str, Path] = {}
    for fmt in formats:
        fmt_lower = fmt.lower()
        if fmt_lower == "json":
            out_path = out_dir / (npz_path.stem + ".json")
            outputs["json"] = export_to_json(field, out_path, data_policy=json_data, downsample=json_downsample)
        elif fmt_lower == "csv":
            out_path = out_dir / (npz_path.stem + ".csv")
            outputs["csv"] = export_to_csv(
                field,
                out_path,
                mode=csv_mode,
                slice_axis=csv_slice_axis,
                slice_index=csv_slice_index,
            )
        elif fmt_lower in {"h5", "hdf5"}:
            out_path = out_dir / (npz_path.stem + ".h5")
            outputs["h5"] = export_to_hdf5(field, out_path)
        else:
            raise ValueError(f"unknown export format: {fmt}")
    return outputs


def export_field(
    npz_path: Path,
    fmt: str,
    out_path: Path,
    options: Optional[Dict[str, object]] = None,
) -> Path:
    fmt_lower = fmt.lower()
    options = options or {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt_lower == "npz":
        shutil.copy2(npz_path, out_path)
        return out_path
    field = read_npz_field(npz_path)
    if fmt_lower == "json":
        return export_to_json(
            field,
            out_path,
            data_policy=str(options.get("json_data", "none")),
            downsample=int(options.get("json_downsample", 1)),
        )
    if fmt_lower == "csv":
        return export_to_csv(
            field,
            out_path,
            mode=str(options.get("csv_mode", "auto")),
            slice_axis=options.get("csv_slice_axis"),
            slice_index=options.get("csv_slice_index"),
        )
    if fmt_lower in {"h5", "hdf5"}:
        return export_to_hdf5(field, out_path)
    raise ValueError(f"unknown export format: {fmt}")

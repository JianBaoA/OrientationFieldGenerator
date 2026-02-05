"""
Module: core/io.py
Responsibility:
  - Read/write orientation field NPZ bundles and associated metadata.
  - Provide a single IO entry point for CLI/GUI.

Public API:
  - read_npz(path)
  - write_npz(field, path)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np


NPZ_META_KEY = "meta_json"
NPZ_META_DEBUG_KEY = "meta_debug_json"
__all__ = ["write_npz", "read_npz", "is_frozen", "get_output_root"]


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False) and getattr(sys, "executable", ""))


def get_output_root(app_name: str = "OrientationFieldGenerator") -> Path:
    if not is_frozen():
        return Path("output")
    if os.name == "nt":
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return local_app_data / app_name / "output"
    return Path.home() / ".local" / "share" / app_name / "output"


def write_npz(field: Dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(field["meta"], ensure_ascii=False)
    payload = {key: value for key, value in field.items() if key not in {"meta", "meta_debug"}}
    # NumPy 2.0+: use np.str_ (np.unicode_ removed)
    payload[NPZ_META_KEY] = np.array(meta_json, dtype=np.str_)
    if "meta_debug" in field:
        meta_debug_json = json.dumps(field["meta_debug"], ensure_ascii=False)
        payload[NPZ_META_DEBUG_KEY] = np.array(meta_debug_json, dtype=np.str_)
    np.savez_compressed(output_path, **payload)
    return output_path


def read_npz(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    raw = data[NPZ_META_KEY]
    meta_json = raw.item() if hasattr(raw, "item") else raw
    if not isinstance(meta_json, str):
        meta_json = str(meta_json)
    meta = json.loads(meta_json)
    field = {key: data[key] for key in data.files if key not in {NPZ_META_KEY, NPZ_META_DEBUG_KEY}}
    field["meta"] = meta
    if NPZ_META_DEBUG_KEY in data.files:
        meta_debug_raw = data[NPZ_META_DEBUG_KEY]
        meta_debug_json = meta_debug_raw.item() if hasattr(meta_debug_raw, "item") else meta_debug_raw
        if not isinstance(meta_debug_json, str):
            meta_debug_json = str(meta_debug_json)
        field["meta_debug"] = json.loads(meta_debug_json)
    return field

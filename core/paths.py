from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["get_app_data_dir"]


def get_app_data_dir(app_name: str) -> Path:
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / app_name
    return Path.home() / f".{app_name}"

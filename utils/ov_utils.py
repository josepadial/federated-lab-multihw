"""
ov_utils.py

Utilities to query available OpenVINO devices.
"""

from __future__ import annotations

from typing import List


def get_available_devices() -> List[str]:
    try:
        from openvino import Core
        core = Core()
        return list(core.available_devices)
    except Exception:
        return []

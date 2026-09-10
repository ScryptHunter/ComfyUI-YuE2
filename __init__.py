"""Dependency-safe YuE2 generation and SheetSage2 transcription for ComfyUI.

The YuE2 wheel is loaded from the selected model directory without pip dependency
resolution. SheetSage2 can use a fully local MERT2 parent model.
"""
from __future__ import annotations

import os
import sys

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def _try_register(module_name: str, mapping: dict) -> None:
    """Register one node module without breaking unrelated node groups."""
    import importlib

    try:
        module = importlib.import_module(f".{module_name}", __package__)
    except Exception as exc:  # noqa: BLE001
        mapping[module_name] = f"{type(exc).__name__}: {exc}"
        print(f"[ComfyUI-YuE2] Skipping {module_name}: {exc}")
        return
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))


_failed: dict = {}

_try_register("nodes.yue2", _failed)
_try_register("nodes.sheetsage2", _failed)

if not NODE_CLASS_MAPPINGS:
    print("[ComfyUI-YuE2] No nodes were registered; check dependencies and the errors above")
else:
    print(f"[ComfyUI-YuE2] Registered {len(NODE_CLASS_MAPPINGS)} nodes")
    for name, err in _failed.items():
        print(f"[ComfyUI-YuE2]   Failed to load {name}: {err}")

WEB_DIRECTORY = None  # No frontend extension required.

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

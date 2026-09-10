# -*- coding: utf-8 -*-
"""Load the official YuE2 wheel without letting pip alter ComfyUI.

The model repository ships ``yue2_infer-*.whl`` as a pure-Python wheel.  A
normal ``pip install`` also processes the wheel's exact dependency pins (torch,
transformers, numpy, ...), which can replace the versions bundled with
ComfyUI.  For a custom node that is the wrong ownership boundary.

This module treats the wheel as a code archive: it extracts it into a private
cache under this node and prepends that directory to ``sys.path``.  Dependency
metadata is deliberately not installed or resolved.  ComfyUI therefore keeps
full control of its Python environment.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile


_WHEEL_RE = re.compile(
    r"^yue2[_-]infer-(?P<version>[^-]+)-py3-none-any\.whl$", re.IGNORECASE
)
_CACHE_ROOT = Path(__file__).resolve().parent.parent / ".yue2_runtime"


def _wheel_candidates(model_path: str | os.PathLike[str]) -> list[Path]:
    """Return colocated YuE2 wheels, nearest directory first."""
    path = Path(model_path).expanduser()
    if not path.is_dir():
        return []

    roots = [path, path.parent]
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            entries = root.iterdir()
        except OSError:
            continue
        local: list[Path] = []
        for entry in entries:
            if entry.is_file() and _WHEEL_RE.match(entry.name) and entry not in seen:
                seen.add(entry)
                local.append(entry)
        # A wheel in the model directory always wins over one in its parent.
        # Within one directory the highest wheel filename/version wins.
        found.extend(sorted(local, key=lambda p: p.name, reverse=True))
    return found


def _safe_extract(wheel: Path, destination: Path) -> None:
    """Extract *wheel* while rejecting absolute and parent-traversal members."""
    destination_abs = os.path.abspath(destination)
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            member = info.filename.replace("\\", "/")
            target = os.path.abspath(os.path.join(destination_abs, member))
            if target != destination_abs and not target.startswith(destination_abs + os.sep):
                raise RuntimeError(f"unsafe path in YuE2 wheel: {info.filename!r}")
        archive.extractall(destination)


def _runtime_dir(wheel: Path) -> Path:
    """Materialize a wheel in the node-private cache and return its import root."""
    match = _WHEEL_RE.match(wheel.name)
    if match is None:
        raise ValueError(f"not a supported YuE2 wheel: {wheel}")
    # Include file size and mtime so replacing a wheel cannot reuse stale code.
    stat = wheel.stat()
    cache_name = f"{wheel.stem}-{stat.st_size}-{stat.st_mtime_ns}"
    destination = _CACHE_ROOT / cache_name
    marker = destination / ".complete"
    if marker.is_file():
        return destination

    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=cache_name + "-", dir=_CACHE_ROOT))
    try:
        _safe_extract(wheel, temp)
        if not (temp / "yue2" / "__init__.py").is_file():
            raise RuntimeError(f"wheel does not contain the yue2 package: {wheel}")
        (temp / ".complete").write_text(wheel.name, encoding="utf-8")
        try:
            temp.replace(destination)
        except FileExistsError:
            # Another ComfyUI worker completed the same extraction first.
            shutil.rmtree(temp, ignore_errors=True)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return destination


def import_yue2(model_path: str | os.PathLike[str]):
    """Import ``yue2`` without running pip or changing installed packages.

    A wheel beside the selected model (or one directory above it) takes
    precedence.  If no wheel is present, an already importable installation is
    accepted for backwards compatibility.
    """
    loaded = sys.modules.get("yue2")
    if loaded is not None:
        return loaded

    wheels = _wheel_candidates(model_path)
    if wheels:
        wheel = wheels[0]
        runtime = _runtime_dir(wheel)
        runtime_text = str(runtime)
        if runtime_text not in sys.path:
            sys.path.insert(0, runtime_text)
        print(f"[ComfyUI-YuE2] Using isolated runtime {wheel.name} "
              "(pip was not invoked; ComfyUI dependencies were not changed)")

    # YuE2 0.1.x targets transformers 4.x.  Apply the node's compatibility
    # shims before importing it when ComfyUI provides transformers 5.x.
    from .transformers5 import apply_transformers5_compat

    apply_transformers5_compat(verbose=True)
    try:
        return importlib.import_module("yue2")
    except ModuleNotFoundError as exc:
        if exc.name == "yue2":
            raise ModuleNotFoundError(
                "YuE2 inference library not found. Put "
                "yue2_infer-0.1.5-py3-none-any.whl in the selected YuE2 model "
                "directory. This node loads it in isolation; do not pip-install it."
            ) from exc
        raise ModuleNotFoundError(
            f"The YuE2 runtime is missing dependency {exc.name!r}. Install only "
            "that missing package; do not let pip resolve the wheel's pinned dependencies."
        ) from exc


__all__ = ["import_yue2"]

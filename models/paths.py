# -*- coding: utf-8 -*-
"""模型路径管理。

遵循 ComfyUI 的模型目录约定，在 ``ComfyUI/models/`` 下使用三个目录，
每个目录既可放 Hugging Face 快照（含 ``config.json`` 的子目录），
也可把模型文件直接放在目录顶层（``SheetSage2`` 就是这样）：

.. code-block::

    ComfyUI/models/
    ├── YuE2/                        歌曲生成
    │   ├── YuE2-3B/                 生成模型   (m-a-p/YuE2-3B 或 mrfakename/YuE2-3B)
    │   ├── YuE2-Vae/                音频解码器 (m-a-p/YuE2-Vae)
    │   └── YuE2-Vae-legacy/         可选       (m-a-p/YuE2-Vae-legacy)
    └── SheetSage2/                  歌曲转录
        ├── config.json              模型文件直接放在这一层 (m-a-p/SheetSage2)
        ├── model.safetensors
        └── ...
    └── MERT2/
        └── MERT-v2-FullSong/        SheetSage2 父模型（可完全本地加载）

``SheetSage2`` 用 ``huggingface-cli`` 下载时注意它默认会多套一层同名目录::

    # 想要"文件直接放顶层"就要显式指定内层，再手动上移
    huggingface-cli download m-a-p/SheetSage2 --local-dir /tmp/ss2
    # 然后 把 /tmp/ss2/* 移入 ComfyUI/models/SheetSage2/

两个目录都可用 ``extra_model_paths.yaml`` 重定向::

    yue2_music:
        base_path: F:/models/music
        yue2: YuE2/
        sheetsage2: SheetSage2/
"""
from __future__ import annotations

import os

import folder_paths

# "模型文件就在该类别目录本身"时，用该目录的**名字**作为下拉框显示值
# （如 ``SheetSage2``）。这样用户看到的就是模型名，而不是 "." 或占位符。


def _self_label(key: str) -> str:
    """目录本身即模型时，用目录名当显示名（如 ``SheetSage2``）。"""
    return _KEYS.get(key, key)


def is_self(key: str, name: str) -> bool:
    """判断下拉框里的值是否代表"该类别目录本身"。"""
    return name in (".", "", "（本目录）") or name == _self_label(key)

# 目录键 -> models/ 下的默认子目录名
_KEYS = {
    "yue2": "YuE2",
    "sheetsage2": "SheetSage2",
    "mert2": "MERT2",
}

_dir_cache: dict[str, list[str]] = {}


def _register() -> None:
    """把四个模型目录注册进 ComfyUI 的路径系统（幂等）。"""
    for key, name in _KEYS.items():
        default = os.path.join(folder_paths.models_dir, name)
        os.makedirs(default, exist_ok=True)
        existing = folder_paths.folder_names_and_paths.get(key)
        if existing is None:
            folder_paths.add_model_folder_path(key, default)
        elif default not in existing[0]:
            folder_paths.add_model_folder_path(key, default)


def model_dirs(key: str) -> list[str]:
    """返回某个模型类别的所有搜索目录（含 extra_model_paths 追加的）。"""
    _register()
    entry = folder_paths.folder_names_and_paths.get(key)
    dirs = list(entry[0]) if entry else []
    if not dirs:
        dirs = [os.path.join(folder_paths.models_dir, _KEYS[key])]
    return dirs


def list_snapshots(key: str, require: str = "config.json") -> list[str]:
    """列出某类别下所有可用的模型快照名（升序）。

    识别两种布局：
    - ``<目录>/<快照名>/config.json``   → 返回 ``快照名``（如 ``YuE2-3B``）
    - ``<目录>/config.json``            → 返回该目录的名字（如 ``SheetSage2``）

    第二种布局下模型文件直接放在类别目录顶层，下拉框用目录名做显示值，
    ``resolve`` 会把它解析回该目录。
    """
    names: list[str] = []
    for base in model_dirs(key):
        if not os.path.isdir(base):
            continue
        if os.path.isfile(os.path.join(base, require)):
            names.append(_self_label(key))
        try:
            entries = sorted(os.listdir(base))
        except OSError:
            continue
        for name in entries:
            path = os.path.join(base, name)
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, require)):
                names.append(name)
    # 去重保序
    seen, out = set(), []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def resolve(key: str, name: str) -> str:
    """把快照名解析为本地绝对路径。

    找不到本地目录时按 Hugging Face repo id 原样返回（由调用方决定是否联网下载）。
    """
    if is_self(key, name):
        for base in model_dirs(key):
            if os.path.isdir(base) and os.path.isfile(os.path.join(base, "config.json")):
                return base
        for base in model_dirs(key):
            if os.path.isdir(base):
                return base
    if os.path.isabs(name) and os.path.isdir(name):
        return name
    for base in model_dirs(key):
        candidate = os.path.join(base, name)
        if os.path.isdir(candidate):
            return candidate
    return name  # 视为 repo id


def resolve_optional(key: str, name: str) -> str | None:
    """解析可选的配套模型；``"none"``/空表示不使用。"""
    if not name or name.lower() == "none":
        return None
    path = resolve(key, name)
    return path if os.path.isdir(path) else None


def default_snapshot(key: str, fallback: str) -> str:
    """返回第一个可用快照；没有本地模型时返回给定的 HF repo id。"""
    found = list_snapshots(key)
    return found[0] if found else fallback


def vae_options() -> list[str]:
    """YuE2 解码器候选项（含官方 hub id 作为兜底）。"""
    local = [n for n in list_snapshots("yue2") if "vae" in n.lower()]
    for name in ("YuE2-Vae", "YuE2-Vae-legacy"):
        if name not in local:
            local.append(name)
    return local


def vae_dir(model_dir: str, name: str) -> str:
    """解析 YuE2 解码器路径，优先使用与生成模型同目录的副本。"""
    if name and name != "none":
        for candidate in (os.path.join(model_dir, name),):
            if os.path.isdir(candidate):
                return candidate
        for base in model_dirs("yue2"):
            candidate = os.path.join(base, name)
            if os.path.isdir(candidate):
                return candidate
    return f"m-a-p/{name or 'YuE2-Vae'}"


def clear_cache() -> None:
    _dir_cache.clear()

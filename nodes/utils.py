# -*- coding: utf-8 -*-
"""节点共用工具：进度回调、音频张量处理、输出目录。"""
from __future__ import annotations

import os
import re
import time

import torch

import folder_paths

try:
    from comfy.utils import ProgressBar
except Exception:  # 脱离 ComfyUI 环境时的兜底
    ProgressBar = None


def progress_bar(total: int):
    """创建 ComfyUI 进度条；不可用时返回 None。"""
    if ProgressBar is None:
        return None
    try:
        return ProgressBar(int(total))
    except Exception:
        return None


def advance(bar, n: int = 1) -> None:
    if bar is not None:
        try:
            bar.update(n)
        except Exception:
            pass


def timestamp_dir(root: str, prefix: str = "") -> str:
    """在 ComfyUI 输出目录下创建带时间戳的子目录。"""
    path = os.path.join(folder_paths.get_output_directory(), root,
                        time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(path, exist_ok=True)
    return path


def safe_stem(text: str, fallback: str = "output", limit: int = 48) -> str:
    """把任意文本转成安全的文件名片段。"""
    cleaned = re.sub(r"[^0-9A-Za-z一-鿿_-]+", "_", (text or "").strip())
    cleaned = cleaned.strip("_")[:limit]
    return cleaned or fallback


def mono_waveform(audio: dict) -> tuple[torch.Tensor, int]:
    """从 ComfyUI AUDIO 字典取出单声道波形 ``[T]`` 与采样率。"""
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("Invalid audio input; connect a ComfyUI AUDIO output")
    waveform = audio["waveform"]
    rate = int(audio.get("sample_rate", 44100))
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0) if waveform.shape[0] <= 32 else waveform[0]
    return waveform.detach().float().cpu().reshape(-1), rate


def audio_dict(waveform: torch.Tensor, sample_rate: int) -> dict:
    """构造 ComfyUI AUDIO 字典。waveform 支持 ``[T]`` / ``[C, T]`` / ``[B, C, T]``。"""
    wav = torch.as_tensor(waveform).float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.ndim == 2:
        wav = wav.unsqueeze(0)
    return {"waveform": wav.contiguous(), "sample_rate": int(sample_rate)}

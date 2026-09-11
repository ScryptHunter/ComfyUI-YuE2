# -*- coding: utf-8 -*-
"""SheetSage2 歌曲转录（音频 → ABC 乐谱 + 曲式结构）。

transformsers 5.x 的适配见 ``compat.sheetsage2``；这里只管加载、缓存与调用。
"""
from __future__ import annotations

import gc
import io
import os

import torch

from ..compat.sheetsage2 import load_sheetsage2
from .paths import model_dirs, resolve

_SAMPLE_RATE = 24000
_cache: dict = {}


def _local_mert2(model_path: str, mert2_name: str = "auto") -> str | None:
    """解析本地 MERT2 父模型；auto 兼容独立目录和嵌套目录。"""
    if mert2_name and mert2_name != "auto":
        candidate = resolve("mert2", mert2_name)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
        raise FileNotFoundError(f"Local MERT2 model directory does not exist: {candidate}")

    candidates = []
    if os.path.isdir(model_path):
        candidates.append(os.path.join(model_path, "MERT-v2-FullSong"))
    for base in model_dirs("mert2"):
        candidates.extend((os.path.join(base, "MERT-v2-FullSong"), base))
    for candidate in candidates:
        if (os.path.isfile(os.path.join(candidate, "config.json"))
                and os.path.isfile(os.path.join(candidate, "model.safetensors"))):
            return os.path.abspath(candidate)
    return None


def load(model_name: str = "SheetSage2", device: str = "cuda",
         dtype: str = "bfloat16", mert2_name: str = "auto"):
    """加载或复用 SheetSage2 模型。"""
    path = resolve("sheetsage2", model_name)
    base_path = _local_mert2(path, mert2_name)
    dt = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    key = (path, base_path, device, str(dt))
    if _cache.get("key") == key and _cache.get("model") is not None:
        return _cache["model"]

    if _cache.get("model") is not None:
        close()

    if base_path:
        print(f"[ComfyUI-YuE2] SheetSage2 is using local MERT2: {base_path}")
    else:
        print("[ComfyUI-YuE2] Local MERT2 not found; using the Hugging Face cache/download")
    model = load_sheetsage2(path, device=device, dtype=dt,
                            base_model_path=base_path)
    _cache["model"] = model
    _cache["key"] = key
    return model


def close() -> None:
    _cache.pop("model", None)
    _cache.pop("key", None)
    torch.cuda.empty_cache()
    gc.collect()


def transcribe(model, waveform: torch.Tensor, sample_rate: int, *,
               melody_only: bool = False, preset: str = "default",
               max_seconds: float = 0.0, output_dir: str | None = None,
               progress=None, abc_error_mode: str = "strict",
               overlap_seconds: float = -1.0,
               lookahead_seconds: float = -1.0):
    """转录音频。

    Args:
        waveform: ``[C, T]`` 或 ``[1, C, T]`` 张量
        melody_only: 输出不含和弦的旋律谱（翻唱用）
        max_seconds: 只转录前 N 秒；0 表示完整
        output_dir: 保存 score.abc/midi/lab 的目录

    Returns:
        ``(abc_text, structure_text, result_dict)``。``structure_text`` 每行形如
        ``起始秒<tab>结束秒<tab>段落名``，供歌词格式化节点对齐字幕分段。
    """
    wav = waveform.detach().float().cpu()
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 2:
        wav = wav.mean(dim=0) if wav.shape[0] <= 32 else wav[0]
    wav = wav.reshape(-1)

    if sample_rate and int(sample_rate) != _SAMPLE_RATE:
        import torchaudio.functional as AF
        wav = AF.resample(wav, int(sample_rate), _SAMPLE_RATE)
    wav = wav.float().contiguous()

    dtype = "bf16" if next(model.parameters()).dtype == torch.bfloat16 else "fp32"
    kwargs = dict(dtype=dtype, preset=preset, melody_only=bool(melody_only))
    if max_seconds and max_seconds > 0:
        kwargs["max_seconds"] = float(max_seconds)
    if progress is not None:
        kwargs["progress"] = progress
    if overlap_seconds >= 0:
        kwargs["overlap_seconds"] = float(overlap_seconds)
    if lookahead_seconds >= 0:
        kwargs["lookahead_seconds"] = float(lookahead_seconds)

    try:
        result = model.transcribe(wav, sampling_rate=_SAMPLE_RATE,
                                  output_dir=output_dir, **kwargs)
    except RuntimeError as exc:
        result = getattr(exc, "result", None)
        if not isinstance(result, dict) or abc_error_mode == "strict":
            raise
        if abc_error_mode == "fallback_full":
            full_kwargs = dict(kwargs)
            full_kwargs["melody_only"] = False
            result = model.transcribe(wav, sampling_rate=_SAMPLE_RATE,
                                      output_dir=output_dir, **full_kwargs)
            result = dict(result)
            result["abc_recovery"] = "fallback_full"
            result["abc_error"] = str(exc)
        else:
            result = dict(result)
            result.setdefault("abc_error", str(exc))

    abc_text = result.get("abc") or ""
    if not abc_text and abc_error_mode in {"snap_invalid_notes", "skip_invalid_notes"}:
        abc_text = _fallback_abc_from_midi(result, skip_invalid=abc_error_mode == "skip_invalid_notes")
        result["abc"] = abc_text
        result["abc_recovery"] = abc_error_mode
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with io.open(os.path.join(output_dir, "score_recovered.abc"), "w", encoding="utf-8") as f:
                f.write(abc_text)
    structure_text = structure_from_events(result.get("events") or [])
    midi = result.get("midi") or (result.get("midis") or {}).get("melody") or b""
    return abc_text, structure_text, result, midi


_ABC_PITCH = {0: "C", 1: "^C", 2: "D", 3: "^D", 4: "E", 5: "F",
              6: "^F", 7: "G", 8: "^G", 9: "A", 10: "^A", 11: "B"}


def _midi_note_name(pitch: int) -> str:
    """Return an explicit-accidental ABC pitch, independent of key signature."""
    name = _ABC_PITCH[int(pitch) % 12]
    octave = int(pitch) // 12 - 1
    if octave >= 5:
        return name.lower() + "'" * (octave - 5)
    if octave < 4:
        return name + "," * (4 - octave)
    return name


def _fallback_abc_from_midi(result: dict, *, skip_invalid: bool = False) -> str:
    """Build a conservative 1/32-grid ABC score when SheetSage2 ABC export fails.

    SheetSage2 occasionally produces a valid MIDI result but rejects a note that
    is off its symbolic sub-beat grid.  This recovery path keeps the MIDI as the
    authority and creates a simple, editable two-voice score.  It intentionally
    omits chord symbols rather than inventing harmony.
    """
    import pretty_midi

    data = result.get("midi") or (result.get("midis") or {}).get("melody")
    if not data:
        raise RuntimeError("SheetSage2 did not return ABC or recoverable MIDI data")
    midi = pretty_midi.PrettyMIDI(io.BytesIO(data))
    tempi_t, tempi = midi.get_tempo_changes()
    bpm = float(tempi[0]) if len(tempi) else 120.0
    bpm = min(300.0, max(30.0, bpm))
    step_seconds = 60.0 / bpm / 8.0  # L:1/32
    bar_steps = 32                  # M:4/4

    pitched = [inst for inst in midi.instruments if not inst.is_drum and inst.notes]
    vocal = [i for i in pitched if "vocal" in (i.name or "").lower()]
    instrumental = [i for i in pitched if i not in vocal]
    if not vocal and pitched:
        vocal, instrumental = [pitched[0]], pitched[1:]

    def events(instruments):
        rows = []
        cursor = 0
        for note in sorted((n for i in instruments for n in i.notes), key=lambda n: (n.start, n.pitch)):
            start = max(0, int(round(note.start / step_seconds)))
            end = max(start + 1, int(round(note.end / step_seconds)))
            if start < cursor:
                if skip_invalid:
                    continue
                start = cursor
            if end <= start:
                if skip_invalid:
                    continue
                end = start + 1
            rows.append((start, end, int(note.pitch)))
            cursor = end
        return rows

    def duration(n):
        return "" if n == 1 else str(n)

    def voice_text(rows):
        if not rows:
            return "z32 |"
        out, cursor = [], 0
        for start, end, pitch in rows:
            if start > cursor:
                out.append("z" + duration(start - cursor))
            remaining = end - start
            pos = start
            while remaining:
                part = min(remaining, bar_steps - (pos % bar_steps))
                token = _midi_note_name(pitch) + duration(part)
                remaining -= part
                pos += part
                out.append(token + ("-" if remaining else ""))
                if pos % bar_steps == 0:
                    out.append("|")
            cursor = end
        if cursor % bar_steps:
            out.append("z" + duration(bar_steps - cursor % bar_steps))
            out.append("|")
        return " ".join(out)

    title = str(result.get("title") or "SheetSage2 recovered score").replace("\n", " ")
    lines = ["X:1", f"T:{title}", "M:4/4", "L:1/32", f"Q:1/4={bpm:.2f}", "K:C"]
    lines += ["V:Vocal clef=treble name=\"Vocal\"", "[V:Vocal] " + voice_text(events(vocal))]
    if instrumental:
        lines += ["V:Ins clef=treble name=\"Instrumental\"",
                  "[V:Ins] " + voice_text(events(instrumental))]
    return "\n".join(lines) + "\n"


def structure_from_events(events) -> str:
    """把转录事件里的 structure 标注转成 ``起<tab>止<tab>名`` 文本。

    事件只标出每段的起点，这里补齐终点：下一段的起点即本段终点，
    最后一段延伸到最后一个事件时间。
    """
    points: list[tuple[float, str]] = []
    end_hint = 0.0
    for event in events:
        time = float(event.get("time", 0.0))
        end_hint = max(end_hint, time)
        name = (event.get("values") or {}).get("structure")
        if name:
            points.append((time, str(name)))
    if not points:
        return ""

    # 合并相邻的同名段（SheetSage2 会把一段拆成多个事件）
    merged: list[list] = []
    for start, name in sorted(points):
        if merged and merged[-1][1] == name:
            merged[-1][2] = start
        else:
            merged.append([start, start, name])

    lines = []
    for i, (start, _cur_end, name) in enumerate(merged):
        end = merged[i + 1][0] if i + 1 < len(merged) else max(end_hint + 1.0, start + 1.0)
        lines.append(f"{start:.2f}\t{end:.2f}\t{name}")
    return "\n".join(lines)

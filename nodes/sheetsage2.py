# -*- coding: utf-8 -*-
"""SheetSage2 转录节点 + 歌词格式化节点。

SheetSage2: 歌曲 -> ABC 乐谱 + 曲式结构（供翻唱与歌词分段）
歌词格式化: 带时间戳的字幕 + 曲式结构 -> 带段落标签的 YuE2 歌词
"""
from __future__ import annotations

import io
import os
import re

from ..models import sheetsage2 as ss2_model
from ..models.paths import list_snapshots
from .utils import mono_waveform, timestamp_dir

# SheetSage2 的结构名 -> YuE2 段落标签（未识别的原样保留）
_SECTION_ALIASES = {
    "silence": "intro",
    "instrumental": "interlude",
    "solo": "interlude",
}

# 纯音乐段落（歌词结构化时不分配歌词行）
_NON_SINGING = {"intro", "silence", "interlude", "instrumental", "solo"}
_SECTION_TAG_LINE = re.compile(r"^\s*\[[^\]\r\n]+\]\s*$", re.IGNORECASE)


def parse_structure_spans(text: str, duration_hint: float):
    """解析结构文本为 ``[(起点, 终点, 名)]``，合并相邻同名段。

    支持 ``起点<tab>终点<tab>名``、``起-止: 名``、``起点: 名`` 三种行格式。
    """
    spans: list[tuple[float, float, str]] = []
    points: list[tuple[float, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = re.split(r"\t+", line)
        if len(parts) == 3:
            try:
                spans.append((float(parts[0]), float(parts[1]), parts[2].strip()))
                continue
            except ValueError:
                pass
        m = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.+)$", line)
        if m:
            spans.append((float(m.group(1)), float(m.group(2)), m.group(3).strip()))
            continue
        m = re.match(r"^(\d+(?:\.\d+)?)\s*:\s*(.+)$", line)
        if m:
            points.append((float(m.group(1)), m.group(2).strip()))

    if not spans and points:
        points.sort()
        for i, (start, name) in enumerate(points):
            end = points[i + 1][0] if i + 1 < len(points) else duration_hint
            spans.append((start, end, name))

    # 合并相邻同名段
    merged: list[tuple[float, float, str]] = []
    for start, end, name in sorted(spans):
        if merged and merged[-1][2] == name and abs(merged[-1][1] - start) < 1e-3:
            merged[-1] = (merged[-1][0], end, name)
        else:
            merged.append((start, end, name))
    return merged


class SheetSage2Loader:
    """加载 SheetSage2 歌曲转录模型。"""

    DESCRIPTION = (
        "Loads the SheetSage2 transcription adapter and its MERT-v2 FullSong "
        "encoder for legacy workflows. Downloads are disabled by default. New "
        "workflows should use the native sheetsage2_bf16.safetensors encoder, "
        "which already contains MERT2."
    )

    @classmethod
    def INPUT_TYPES(cls):
        models = list_snapshots("sheetsage2") or ["m-a-p/SheetSage2"]
        mert2 = list_snapshots("mert2")
        mert2_options = mert2 + ["auto"] if mert2 else ["auto"]
        return {
            "required": {
                "model": (models,),
                "device": (["cuda", "cpu"],),
                "dtype": (["bfloat16", "float32"],),
            },
            "optional": {
                "mert2_model": (mert2_options, {"tooltip":
                    "MERT-v2-FullSong parent model. Place it under models/MERT2/. "
                    "Auto checks local folders and the existing Hugging Face cache without downloading."}),
                "download_policy": (["local_only", "allow_huggingface_download"], {"tooltip":
                    "local_only never starts a MERT2 download. Enable downloading explicitly only for the legacy backend."}),
            },
        }

    RETURN_TYPES = ("SS2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "YuE2/Legacy HF Runtime/SheetSage2"

    def load(self, model, device, dtype, mert2_model="auto",
             download_policy="local_only"):
        return (ss2_model.load(model, device=device, dtype=dtype,
                               mert2_name=mert2_model,
                               allow_download=download_policy == "allow_huggingface_download"),)


class SheetSage2Transcribe:
    """歌曲转录: 音频 -> ABC 乐谱 + 曲式结构。"""

    DESCRIPTION = (
        "Transcribes source audio into an editable ABC score, MIDI, and timed "
        "song-section structure for cover and rearrangement workflows."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SS2_MODEL",),
                "audio": ("AUDIO",),
                "melody_only": ("BOOLEAN", {"default": False,
                    "tooltip": "Output a melody-only score without chord symbols. Recommended for covers with cot=melody."}),
                "preset": (["default", "paper"],),
                "max_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0,
                                          "step": 1.0, "tooltip": "Transcribe only the first N seconds. 0 processes the full recording."}),
                "save_outputs": ("BOOLEAN", {"default": True,
                    "tooltip": "Save score.abc, MIDI, and LAB files under output/SheetSage2/."}),
            },
            "optional": {
                "abc_error_mode": (["strict", "snap_invalid_notes", "skip_invalid_notes", "fallback_full", "return_midi_only"], {
                    "tooltip": "What to do when SheetSage2 cannot place a note on the ABC grid. Strict reports the original error; recovery modes rebuild a simple score from MIDI."}),
                "overlap_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Window overlap override. -1 uses the SheetSage2 preset."}),
                "lookahead_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Window look-ahead override. -1 uses the SheetSage2 preset."}),
                "auto_unload": ("BOOLEAN", {"default": False,
                    "tooltip": "Release SheetSage2 and MERT2 after transcription to recover VRAM."}),
            },
        }

    # Keep the first three outputs in their original order so saved workflows
    # continue to receive info on output 2. MIDI is appended as output 3.
    RETURN_TYPES = ("STRING", "STRING", "STRING", "YUE2_MIDI",)
    RETURN_NAMES = ("abc", "structure", "info", "midi",)
    OUTPUT_NODE = True
    FUNCTION = "transcribe"
    CATEGORY = "YuE2/Legacy HF Runtime/SheetSage2"

    def transcribe(self, model, audio, melody_only, preset, max_seconds, save_outputs,
                   abc_error_mode="strict", overlap_seconds=-1.0,
                   lookahead_seconds=-1.0, auto_unload=False):
        wav, rate = mono_waveform(audio)
        out_dir = timestamp_dir("SheetSage2") if save_outputs else None

        abc_text, structure_text, result, midi = ss2_model.transcribe(
            model, wav, rate, melody_only=melody_only, preset=preset,
            max_seconds=max_seconds, output_dir=out_dir,
            abc_error_mode=abc_error_mode, overlap_seconds=overlap_seconds,
            lookahead_seconds=lookahead_seconds)

        chords = sum(1 for e in (result.get("events") or [])
                     if (e.get("values") or {}).get("chord"))
        info = (f"duration={result.get('duration_seconds', 0):.1f}s "
                f"vocal_notes={result.get('vocal_notes', 0)} "
                f"ins_notes={result.get('instrumental_notes', 0)} "
                f"chords={chords} sections={len(structure_text.splitlines())} "
                f"elapsed={result.get('elapsed_seconds', 0):.1f}s")
        if result.get("abc_error"):
            info += f" | abc_error: {result['abc_error']}"
        if result.get("abc_recovery"):
            info += (f" | recovered={result['abc_recovery']} (simple MIDI-derived ABC; "
                     "review note timing and harmony)")
        if not abc_text and abc_error_mode == "return_midi_only":
            info += " | ABC unavailable; MIDI returned"
        if out_dir:
            info += f" | saved: {out_dir}"
        if auto_unload:
            ss2_model.close()
            info += " | model unloaded"
        print(f"[SheetSage2] {info}")
        return (abc_text, structure_text, info, midi)


class SheetSage2Unload:
    """Explicitly releases the cached SheetSage2/MERT2 model and CUDA memory."""

    DESCRIPTION = "Unloads SheetSage2 and MERT2 from memory after transcription."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("SS2_MODEL",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    OUTPUT_NODE = True
    CATEGORY = "YuE2/Legacy HF Runtime/SheetSage2"

    def unload(self, model):
        del model
        ss2_model.close()
        return ("SheetSage2 and MERT2 unloaded",)


class SaveMidiFile:
    """Saves SheetSage2 MIDI bytes without depending on its automatic output mode."""

    DESCRIPTION = "Saves a SheetSage2 MIDI output to ComfyUI's output folder."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"midi": ("YUE2_MIDI",), "filename_prefix": ("STRING", {"default": "YuE2/melody"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "YuE2/ABC & Files"

    def save(self, midi, filename_prefix):
        import folder_paths
        folder, filename, counter, _subfolder, _prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{filename}_{counter:05}_.mid")
        with open(path, "wb") as f:
            f.write(bytes(midi or b""))
        return (path,)


class LyricsFormatter:
    """把带时间戳的歌词字幕按曲式结构分段，整理成可交给 YuE2 的歌词。

    输入（二者都可用纯文本手填）:
    - ``subtitles``: 每行 ``起-止: 文本``，或标准 SRT
    - ``structure``: 每行 ``起点<tab>终点<tab>段落名``（SheetSage2 节点输出）

    处理: 解析时间 -> 按时间中点归入结构区间 -> 段首插 ``[verse]``/``[chorus]``
    等标签 -> 无词区间输出空标签（间奏） -> 清理标点与相邻重复行。
    """

    _PUNCT = str.maketrans("", "", "。！？，、；：,.!?;:\"'`()[]{}<>~…·\u2014“”‘’")

    DESCRIPTION = (
        "Converts timed LRC, SRT, or aligned lyric lines into YuE2 sectioned "
        "lyrics, using optional SheetSage2 structure boundaries."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 多行文本框：可直接粘贴 LRC/SRT/纯歌词，也可从上游连线
                # （连线后 widget 自动隐藏）。LRC 详见 parse_subtitles。
                "subtitles": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Paste LRC, SRT, or 'start-end: text' lines, or connect an ASR/subtitle node."}),
                "structure": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Optional SheetSage2 structure output: start<TAB>end<TAB>section. Empty input places all lyrics in [verse]."}),
                "clean_punct": ("BOOLEAN", {"default": True,
                    "tooltip": "Remove punctuation from lyric lines."}),
                "dedupe": ("BOOLEAN", {"default": True,
                    "tooltip": "Remove immediately repeated identical lines without removing repeated choruses."}),
                "min_line_chars": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Discard lines shorter than this length. Use 0 to disable filtering."}),
                "map_sections": ("BOOLEAN", {"default": True,
                    "tooltip": "Map SheetSage2 section names to YuE2 tags, for example silence to intro."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("lyrics", "report",)
    FUNCTION = "format"
    CATEGORY = "YuE2/SheetSage2"

    # ── 解析 ──────────────────────────────────────────────────────────

    def parse_subtitles(self, text: str):
        """支持 ``起-止: 文本``、标准 SRT、LRC 三种格式，返回 ``[(start, end, text)]``。

        对时间戳不可靠的行做兜底：强制对齐器在纯音乐/气声片段上可能给出
        ``0.00-0.00`` 这类零长度区间，若直接使用会让这些行全部落到第一段。
        这里把无效时间标记为 ``None``，交由 :meth:`format` 按行序补位。
        """
        rows: list[tuple[float | None, float | None, str]] = []
        pending_start = None
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.+)$", line)
            if m:
                rows.append((float(m.group(1)), float(m.group(2)), m.group(3).strip()))
                continue
            # LRC: [mm:ss.xx]歌词（同一行可有多个连续时间标签）。
            # 只有行起点没有终点 → end 记 None，末尾统一延伸到下一行起点。
            lrc = re.match(r"^((?:\[\d+:\d+(?:\.\d+)?\])+)(.*)$", line)
            if lrc:
                text_part = lrc.group(2).strip()
                if text_part:
                    for tag in re.findall(r"\[(\d+):(\d+(?:\.\d+)?)\]", lrc.group(1)):
                        t = int(tag[0]) * 60 + float(tag[1])
                        rows.append((t, None, text_part))
                continue
            m = re.match(
                r"^(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)$", line)
            if m:
                g = [int(x) for x in m.groups()]
                start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
                end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
                rows.append((start, end, ""))
                pending_start = start
                continue
            # SRT 的正文行：补上一条时间轴
            if pending_start is not None and rows and rows[-1][2] == "":
                s, e, _ = rows[-1]
                rows[-1] = (s, e, line)
                pending_start = None

        out = []
        for start, end, text in rows:
            if not text:
                continue
            # LRC 头部元数据行（歌名/词曲作者等），不是歌词
            if re.match(r"^(词|曲|编曲|作词|作曲|编|混音|母带|和声|吉他|钢琴|"
                        r"制作|监制|歌词|专辑|歌手|title|artist|album|by)[:：]",
                        text, re.IGNORECASE):
                continue
            # "歌名 - 歌手" 形式的标题行（LRC 首行惯例: 时间 0 且含 " - "）
            if start == 0 and " - " in text:
                continue
            # 零长度、倒挂或 (0,0) 哨兵：对齐器在歌声上失败的产物，时间视为未知。
            # end 为 None 的 LRC 行是"只有起点"的正常形态，下一循环补终点。
            if start is None or (end is not None and (end <= start or
                                                      (start == 0.0 and end == 0.0))):
                out.append((None, None, text))
            else:
                out.append((start, end, text))
        # LRC 行只有起点：延伸到下一行起点（末行 +8s），保证区间有效
        fixed = []
        for i, (s, e, t) in enumerate(out):
            if s is not None and e is None:
                nxt = next((out[j][0] for j in range(i + 1, len(out))
                            if out[j][0] is not None), None)
                e = nxt if (nxt is not None and nxt > s) else s + 8.0
            fixed.append((s, e, t))
        return fixed

    def _fill_missing_times(self, rows):
        """为时间无效的行按行序插值补位，使其仍能归入合理的段落。

        对齐器在歌声上常出现成片失败（尤其副歌与纯音乐段）。做法是找出连续
        的无效区段，在左右两个可靠锚点之间均匀铺开-&#x20;-&#x20;而不是按固定步长从锚点
        回推，后者会把整段未对齐歌词挤到锚点附近。
        """
        n = len(rows)
        valid_idx = [i for i, (s, _, _) in enumerate(rows) if s is not None]

        if not valid_idx:
            # 完全没有任何可靠时间（对齐器在纯音乐/长段歌声上可能全失败）。
            # 此时歌词本身仍然正确，绝不能丢弃：按行序均匀铺在 [0, 1) 上，
            # 保证全部归入第一段并可被人工编辑。
            total = max(n, 1)
            return [(i / total, (i + 1) / total, t) for i, (_, _, t) in enumerate(rows)]

        # 兜底步长：可靠行的平均时长，用于无法在锚点间铺开的极端情形
        fallback = sum(rows[i][1] - rows[i][0] for i in valid_idx) / len(valid_idx)
        fallback = max(float(fallback), 0.05)

        filled = list(rows)
        # 在每段连续无效区段的两侧找锚点，然后均匀铺开
        bounds = [-1] + valid_idx + [n]
        for left, right in zip(bounds, bounds[1:]):
            lo, hi = left + 1, right          # 无效区段 [lo, hi)
            if lo >= hi:
                continue
            count = hi - lo
            anchor_l = filled[left][1] if left >= 0 else 0.0
            anchor_r = filled[right][0] if right < n else None
            if anchor_r is None:              # 尾部：从左侧锚点按兜底步长铺开
                step = fallback
                base = anchor_l
            else:
                span = max(anchor_r - anchor_l, 0.0)
                step = span / count if span > 0 else fallback
                base = anchor_l

            for k in range(count):
                s = base + step * k
                filled[lo + k] = (s, s + step, filled[lo + k][2])
        return filled

    def parse_structure(self, text: str, duration_hint: float):
        """兼容保留：见模块级 :func:`parse_structure_spans`。"""
        return parse_structure_spans(text, duration_hint)

    # ── 执行 ──────────────────────────────────────────────────────────

    def format(self, subtitles, structure, clean_punct, dedupe, min_line_chars,
               map_sections):
        sub_rows = self.parse_subtitles(subtitles)
        if not sub_rows:
            raise ValueError(
                "subtitles is empty or unsupported; use LRC, SRT, or 'start-end: text' lines")

        # 先按文本过滤/清理，再补时间（避免为将被丢弃的行插值）
        lines = []
        for start, end, text in sub_rows:
            cleaned = text.translate(self._PUNCT).strip() if clean_punct else text.strip()
            if not cleaned or (min_line_chars > 0 and len(cleaned) < min_line_chars):
                continue
            lines.append((start, end, cleaned))
        if dedupe:
            deduped = []
            for row in lines:
                if deduped and deduped[-1][2] == row[2]:
                    continue
                deduped.append(row)
            lines = deduped
        if not lines:
            raise ValueError("No lyric lines remain after cleanup; check min_line_chars")

        # 统计对齐失败、时间靠插值推断的行数，用于在 report 中如实告知
        inferred = sum(1 for s, _, _ in lines if s is None)
        lines = self._fill_missing_times(lines)
        timed = [(s, e, t) for s, e, t in lines if s is not None]
        duration = max((e for _, e, _ in timed), default=float(len(lines)))

        spans = self.parse_structure(structure or "", duration + 1.0)
        if not spans:
            # 没有结构信息：全部归入一个段落
            sections = [("verse", [t for _, _, t in lines])]
        else:
            sections = []
            for idx, (start, end, name) in enumerate(spans):
                label = _SECTION_ALIASES.get(name.lower(), name) if map_sections else name
                last = idx == len(spans) - 1
                bucket = []
                for s, e, t in lines:
                    if s is None:
                        continue
                    mid = (s + e) / 2
                    if start <= mid < end or (last and mid >= start):
                        bucket.append(t)
                if sections and sections[-1][0] == label:
                    sections[-1][1].extend(bucket)
                else:
                    sections.append((label, bucket))
            # 歌词掉进纯音乐段（间奏/前奏）说明边界有偏移：并入下一个歌唱段，
            # 间奏保持为空标签（YuE2 把空段当器乐）。
            singing = [i for i, (name, _) in enumerate(sections)
                       if (name or "").lower() not in _NON_SINGING]
            if singing:
                def nearest_singing(i):
                    return min(singing, key=lambda j: (abs(j - i), j))
                for i, (name, bucket) in enumerate(sections):
                    if (name or "").lower() in _NON_SINGING and bucket:
                        sections[nearest_singing(i)][1].extend(bucket)
                        bucket.clear()

        out_lines: list[str] = []
        for name, bucket in sections:
            out_lines.append(f"[{(name or 'verse').strip().lower()}]")
            out_lines.extend(bucket)
            out_lines.append("")
        lyrics = "\n".join(out_lines).rstrip() + "\n"

        report = (f"lines={len(sub_rows)} kept={len(lines)} "
                  f"sections={len(sections)} chars={sum(len(t) for _, _, t in lines)}")
        if inferred:
            # 对齐器在歌声上大面积失败时，段落归属是推断出来的，必须告知用户
            report += (f" | warning: {inferred}/{len(lines)} lines have unreliable "
                       "timestamps; section placement was inferred and should be reviewed")
        print(f"[Lyrics Formatter] {report}")
        return (lyrics, report)


class LoadLyricsFile:
    """从文件读取歌词（LRC / SRT / 纯文本），交给歌词格式化或歌词结构化。

    文件放 ComfyUI 的 input 目录（也可填绝对路径）。
    """

    DESCRIPTION = "Loads an LRC, SRT, or plain-text lyrics file from the ComfyUI input folder."

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        try:
            files = sorted(f for f in os.listdir(input_dir)
                           if f.lower().endswith((".lrc", ".srt", ".txt")))
        except OSError:
            files = []
        return {
            "required": {
                "lyrics_file": (sorted(files) or ["Place an .lrc, .srt, or .txt file in the ComfyUI input folder"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load"
    CATEGORY = "YuE2/SheetSage2"

    def load(self, lyrics_file):
        import folder_paths
        path = lyrics_file if os.path.isabs(lyrics_file) else \
            os.path.join(folder_paths.get_input_directory(), lyrics_file)
        if not os.path.isfile(path):
            raise ValueError(f"Lyrics file does not exist: {path}")
        text = io.open(path, encoding="utf-8", errors="replace").read()
        print(f"[Lyrics File] {lyrics_file}: {len(text)} characters")
        return (text,)


class LyricsStructurer:
    """把纯文本歌词自动加上 [verse]/[chorus] 段落标签，直接交给 YuE2。

    两种用法（structure 留空时按行数自动规划）：
    - 只填 ``lyrics_text``：每 ``verse_lines`` 行算一段，全部标 [verse]
    - 填 ``section_plan``：逗号分隔的段落序列，如 ``verse,chorus,verse,chorus``
      （可带行数 ``verse:8,chorus:8``），按顺序每段取对应行数

    也接受手工标注：文本里已带 ``[xxx]`` 行的段落原样保留。
    """

    DESCRIPTION = (
        "Adds [verse], [chorus], and other YuE2 section tags to plain lyrics, "
        "optionally distributing lines across SheetSage2 song sections."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics_text": ("STRING", {"multiline": True,
                    "tooltip": "Plain lyrics with one phrase per line. Empty lines are ignored."}),
                "structure": ("STRING", {"default": "",
                    "tooltip": "Optional SheetSage2 structure output. Lyric lines are distributed across vocal sections by duration."}),
                "verse_lines": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Number of lines per section when structure is empty."}),
                "section_plan": ("STRING", {"default": "",
                    "tooltip": "Section plan such as verse,chorus,verse or verse:8,chorus:8. Empty means verse sections only."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("lyrics", "report",)
    FUNCTION = "format"
    CATEGORY = "YuE2/SheetSage2"

    def format(self, lyrics_text, structure, verse_lines, section_plan):
        # 已带 [tag] 的文本原样保留（用户手工标好就尊重）
        if any(_SECTION_TAG_LINE.fullmatch(line) for line in (lyrics_text or "").splitlines()):
            cleaned = "\n".join(
                ln for ln in (l.strip() for l in lyrics_text.splitlines())
                if ln and not _SECTION_TAG_LINE.fullmatch(ln))
            report = f"Existing section tags preserved ({len(cleaned.splitlines())} lyric lines)"
            print(f"[Lyrics Structurer] {report}")
            return (lyrics_text.rstrip() + "\n", report)

        lines = [ln.strip() for ln in (lyrics_text or "").splitlines() if ln.strip()]
        if not lines:
            raise ValueError("lyrics_text must not be empty")

        plan = self._parse_plan(section_plan, verse_lines, len(lines))

        # 有结构信息时按歌唱段时长比例分配行数
        spans = parse_structure_spans(structure or "", 0.0) if structure else []
        if spans:
            singing = [(name, end - start) for start, end, name in spans
                       if _SECTION_ALIASES.get(name.lower(), name.lower()) not in _NON_SINGING
                       and (end - start) > 1.0]
            if singing:
                total = sum(d for _, d in singing)
                plan = [(name, max(1, round(len(lines) * dur / total)))
                        for name, dur in singing]
                plan[-1] = (plan[-1][0], len(lines) - sum(n for _, n in plan[:-1]))

        out: list[str] = []
        idx, sections = 0, 0
        for name, count in plan:
            if idx >= len(lines):
                break
            take = lines[idx:idx + count]
            idx += len(take)
            out.append(f"[{name}]")
            out.extend(take)
            out.append("")
            sections += 1
        if idx < len(lines):          # 行数超出规划：剩余全部进尾段
            out.append(f"[{plan[-1][0]}]" if not out or out[-1] else f"[{plan[-1][0]}]")
            out.extend(lines[idx:])

        lyrics = "\n".join(out).rstrip() + "\n"
        report = (f"lines={len(lines)} sections={sections} "
                  f"plan={','.join(f'{n}:{c}' for n, c in plan)}")
        print(f"[Lyrics Structurer] {report}")
        return (lyrics, report)

    @staticmethod
    def _parse_plan(section_plan, verse_lines, n_lines):
        """解析段落规划为 ``[(段名, 行数)]``。"""
        if not section_plan.strip():
            n = max(1, -(-n_lines // max(verse_lines, 1)))
            return [("verse", verse_lines)] * n
        plan = []
        for part in section_plan.split(","):
            part = part.strip().lower()
            if not part:
                continue
            if ":" in part:
                name, cnt = part.split(":", 1)
                plan.append((name.strip() or "verse", max(1, int(cnt))))
            else:
                plan.append((part, verse_lines))
        return plan or [("verse", verse_lines)]


NODE_CLASS_MAPPINGS = {
    "SheetSage2Loader": SheetSage2Loader,
    "SheetSage2Transcribe": SheetSage2Transcribe,
    "SheetSage2Unload": SheetSage2Unload,
    "YuE2SaveMidiFile": SaveMidiFile,
    "YuE2LyricsFormatter": LyricsFormatter,
    "YuE2LyricsStructurer": LyricsStructurer,
    "YuE2LoadLyricsFile": LoadLyricsFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SheetSage2Loader": "SheetSage2 Model Loader (Legacy HF)",
    "SheetSage2Transcribe": "SheetSage2 Audio Transcriber (Legacy HF)",
    "SheetSage2Unload": "SheetSage2 Unload Model (Legacy HF)",
    "YuE2SaveMidiFile": "YuE2 MIDI File Saver",
    "YuE2LyricsFormatter": "YuE2 Lyrics Formatter (Timed to Sections)",
    "YuE2LyricsStructurer": "YuE2 Lyrics Structurer (Text to Sections)",
    "YuE2LoadLyricsFile": "YuE2 Lyrics File Loader (LRC/SRT)",
}

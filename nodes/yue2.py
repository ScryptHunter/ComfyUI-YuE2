# -*- coding: utf-8 -*-
"""YuE2 生成节点：加载、生成歌曲、只出乐谱、卸载。"""
from __future__ import annotations

import os

import torch

from ..models import yue2 as yue2_model
from ..models.paths import list_snapshots, vae_options
from .utils import advance, audio_dict, progress_bar, safe_stem, timestamp_dir

COT_MODES = ["full", "melody", "off"]
VAE_DECODE_MODES = ["tiled", "full"]
ATTN_BACKENDS = ["auto", "external-flash", "cudnn", "sdpa"]
QUANT_MODES = ["none", "fp8"]


class YuE2Loader:
    """加载 YuE2 生成模型与解码器（VAE）。"""

    DESCRIPTION = (
        "Loads the YuE2 generator and audio VAE. The official inference wheel "
        "is isolated so it cannot replace ComfyUI's core dependencies."
    )

    @classmethod
    def INPUT_TYPES(cls):
        models = [n for n in list_snapshots("yue2") if "vae" not in n.lower()]
        if not models:
            models = ["YuE2-3B"]
        return {
            "required": {
                "model": (models, {"tooltip": "Model folder under models/YuE2/."}),
                "vae": (vae_options(), {"tooltip": "Audio decoder. Use YuE2-Vae for listening quality or legacy for benchmark reproduction."}),
                "device": (["cuda", "cpu"],),
                "memory_budget_gib": ("INT", {"default": 24, "min": 6, "max": 96, "step": 1,
                    "tooltip": "VRAM budget. Keep 24 GiB for a 24 GB GPU."}),
                "offload_ar": ("BOOLEAN", {"default": False,
                    "tooltip": "Temporarily offload the AR module during NAR synthesis to save VRAM (slower)."}),
                "offline": ("BOOLEAN", {"default": True,
                    "tooltip": "Use local files only. Disable to allow missing models to download."}),
                "attention_backend": (ATTN_BACKENDS, {"default": "auto",
                    "tooltip": "AR attention backend. Auto falls back to cuDNN when built-in Flash Attention is unavailable; external-flash requires flash-attn."}),
                "quantization": (QUANT_MODES, {"default": "none",
                    "tooltip": "Experimental FP8 quantization for AR linear layers. Saves about 1.3 GB but can be much slower because CUDA Graph is disabled."}),
            },
        }

    RETURN_TYPES = ("YUE2_PIPE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "YuE2"

    def load(self, model, vae, device, memory_budget_gib, offload_ar, offline,
             attention_backend="auto", quantization="none"):
        pipe = yue2_model.load(model, vae, device=device,
                               memory_budget_gib=memory_budget_gib,
                               offload_ar=offload_ar, offline=offline,
                               attention_backend=attention_backend,
                               quantization=quantization)
        return (pipe,)


class YuE2Unload:
    """卸载 YuE2 管线释放显存。"""

    DESCRIPTION = "Unloads the cached YuE2 pipeline and releases its VRAM."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipeline": ("YUE2_PIPE",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "unload"
    CATEGORY = "YuE2"

    def unload(self, pipeline):
        yue2_model.close()
        return ()


class YuE2Sampler:
    """YuE2 歌曲生成：风格 + 歌词（+可选 ABC 乐谱）-> 48 kHz 完整歌曲。"""

    DESCRIPTION = (
        "Generates a complete 48 kHz stereo song from a style prompt and "
        "sectioned lyrics, with optional editable ABC score conditioning."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("YUE2_PIPE",),
                "style": ("STRING", {"default":
                    "Mandarin pop, warm female vocal, clean electric guitar, soft drums",
                    "multiline": True,
                    "tooltip": "Describe genre, instruments, vocal character, language, tempo, and mood."}),
                "lyrics": ("STRING", {"default":
                    "[verse]\nCity lights are fading softly\nMorning waits beyond the blue\n\n"
                    "[chorus]\nCarry every spark of wonder\nLet the open road come through",
                    "multiline": True,
                    "tooltip": "Lyrics with [verse], [chorus], and other section tags. Use the lyrics utility nodes to prepare them."}),
                "cot": (COT_MODES, {"tooltip":
                    "full=melody and harmony plan; melody=melody-only plan (recommended for covers); off=direct generation without a score."}),
                "seed": ("INT", {"default": 831001, "min": 0, "max": 2**63 - 1}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01,
                    "tooltip": "Text guidance strength. 1.0 is the official default (off mode uses 1.01)."}),
                "ode_steps": ("INT", {"default": 32, "min": 4, "max": 64, "step": 1,
                    "tooltip": "Acoustic flow-matching steps. The official default is 32."}),
                "abc_max_tokens": ("INT", {"default": 4096, "min": 64, "max": 8192, "step": 32}),
                "semantic_max_tokens": ("INT", {"default": 9000, "min": 200, "max": 16384, "step": 32}),
                "abc_temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 5.0, "step": 0.01}),
                "semantic_temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "save_flac": ("BOOLEAN", {"default": True,
                    "tooltip": "Also save a 48 kHz FLAC file under output/YuE2/."}),
                "save_abc": ("BOOLEAN", {"default": True,
                    "tooltip": "Save the generated ABC score under output/YuE2/."}),
                "vae_decode": (VAE_DECODE_MODES, {"default": "tiled",
                    "tooltip": "full decodes the entire song at once (faster, more VRAM); tiled is memory-efficient. Full falls back automatically on OOM."}),
                "vae_tile_frames": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 128,
                    "tooltip": "Tile size in latent frames. 0=automatic; use 256 for low-VRAM GPUs."}),
            },
            "optional": {
                "abc": ("STRING", {"forceInput": True,
                    "tooltip": "Optional ABC score from SheetSage2 or manual editing. Requires cot=melody or full."}),
                "abort_after_plan": ("BOOLEAN", {"default": False,
                    "tooltip": "Generate only the ABC plan and skip audio synthesis."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING",)
    RETURN_NAMES = ("audio", "abc_score", "info",)
    FUNCTION = "generate"
    CATEGORY = "YuE2"

    @classmethod
    def VALIDATE_INPUTS(cls, vae_decode):
        """Accept ``False`` saved by workflows created before vae_decode existed.

        Older ComfyUI workflow JSON filled newly added widgets with boolean
        ``false``.  Treat that legacy value as the default ``tiled`` mode rather
        than rejecting the whole prompt before the node can migrate it.
        """
        if vae_decode is False or vae_decode in VAE_DECODE_MODES:
            return True
        return f"vae_decode must be 'tiled' or 'full'; received {vae_decode!r}"

    def generate(self, pipeline, style, lyrics, cot, seed, cfg_scale, ode_steps,
                 abc_max_tokens, semantic_max_tokens, abc_temperature,
                 semantic_temperature, save_flac, save_abc,
                 vae_decode="tiled", vae_tile_frames=0,
                 abc=None, abort_after_plan=False):
        if vae_decode is False:  # Migrate workflows saved before this option existed.
            vae_decode = "tiled"
        style, lyrics = (style or "").strip(), (lyrics or "").strip()
        if not style:
            raise ValueError("style must not be empty")
        if not lyrics:
            raise ValueError("lyrics must not be empty")
        abc_text = (abc or "").strip() or None
        if abc_text and cot == "off":
            raise ValueError("An external ABC score requires cot='melody' or cot='full'")

        abc_sampling = {"temperature": abc_temperature, "max_tokens": int(abc_max_tokens)}
        semantic_sampling = {"temperature": semantic_temperature,
                             "max_tokens": int(semantic_max_tokens)}

        bar = progress_bar(int(abc_max_tokens) + int(semantic_max_tokens))

        def on_token(_phase, _token):
            if bar is not None:
                try:
                    bar.update(1)
                except Exception:
                    pass

        vae_bar_holder: list = []

        def on_vae_progress(completed, total):
            """Create the tiled VAE progress bar after the tile count is known."""
            if not vae_bar_holder:
                vae_bar_holder.append(progress_bar(int(total) if total else 1))
            advance(vae_bar_holder[0],
                    max(0, int(completed) - getattr(vae_bar_holder[0], "_yue2_done", 0)))
            if vae_bar_holder[0] is not None:
                vae_bar_holder[0]._yue2_done = int(completed)

        # Plan-only mode returns an editable score without synthesizing audio.
        if abort_after_plan:
            plan = yue2_model.plan(pipeline, style=style, lyrics=lyrics, cot=cot,
                                   seed=seed, abc=abc_text, cfg_scale=cfg_scale,
                                   abc_sampling=abc_sampling, on_progress=on_token)
            score = plan.abc or ""
            out_dir = timestamp_dir("YuE2") if save_abc else None
            if out_dir and score:
                with open(os.path.join(out_dir, f"{safe_stem(style)}_s{seed}.abc"),
                          "w", encoding="utf-8") as fh:
                    fh.write(score)
            info = f"score only | cot={cot} seed={seed} | abc_tokens={len(plan.abc_ids)}"
            print(f"[YuE2] {info}")
            silence = audio_dict(torch.zeros(1, 160), 48000)
            return (silence, score, info)

        song = yue2_model.generate(
            pipeline, style=style, lyrics=lyrics, cot=cot, seed=seed,
            abc=abc_text, cfg_scale=cfg_scale,
            abc_sampling=abc_sampling, semantic_sampling=semantic_sampling,
            ode_steps=ode_steps, on_progress=on_token,
            vae_decode=vae_decode,
            vae_tile_frames=int(vae_tile_frames) if vae_tile_frames else None,
            on_vae_progress=on_vae_progress)

        out_dir = timestamp_dir("YuE2") if (save_flac or save_abc) else None
        stem = f"{safe_stem(style)}_s{seed}"
        saved = []
        if out_dir:
            if save_flac:
                saved.append(song.save(os.path.join(out_dir, stem + ".flac")))
            if save_abc and song.abc:
                abc_path = os.path.join(out_dir, stem + ".abc")
                with open(abc_path, "w", encoding="utf-8") as fh:
                    fh.write(song.abc)
                saved.append(abc_path)

        seconds = len(song.audio) / song.sample_rate
        info = (f"{seconds:.1f}s @{song.sample_rate}Hz | cot={cot} seed={seed} | "
                f"abc_tokens={song.timing['abc'].get('output_tokens', 0)} "
                f"sem_tokens={song.timing['semantic'].get('output_tokens', 0)} | "
                f"truncated={song.truncated} | e2e={song.timing['e2e_seconds']:.1f}s")
        if saved:
            info += f" | saved: {out_dir}"
        print(f"[YuE2] {info}")

        return (audio_dict(torch.from_numpy(song.audio).T, song.sample_rate),
                song.abc or "", info)


NODE_CLASS_MAPPINGS = {
    "YuE2Loader": YuE2Loader,
    "YuE2Sampler": YuE2Sampler,
    "YuE2Unload": YuE2Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2Loader": "YuE2 Model Loader",
    "YuE2Sampler": "YuE2 Song Generator",
    "YuE2Unload": "YuE2 Unload Model",
}

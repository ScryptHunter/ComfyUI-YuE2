"""Advanced, staged YuE2 controls built on the official pipeline API."""
from __future__ import annotations

import os
import time

import torch

from ..models import yue2 as yue2_model
from .utils import audio_dict, timestamp_dir


OFFICIAL_ABC = dict(temperature=.7, top_p=.9, top_k=30,
                    repetition_penalty=1.005, penalty_window=100,
                    min_tokens=32, max_tokens=4096)
OFFICIAL_SEMANTIC = dict(temperature=1., top_p=.95, top_k=100,
                         repetition_penalty=1.2, penalty_window=50,
                         min_tokens=200, max_tokens=9000)


class YuE2SamplingSettings:
    DESCRIPTION = "Builds complete, separate sampling configurations for score planning and semantic song generation."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preset": (["official", "stable", "creative", "custom"],),
            "abc_temperature": ("FLOAT", {"default": .7, "min": 0., "max": 5., "step": .01}),
            "abc_top_p": ("FLOAT", {"default": .9, "min": .01, "max": 1., "step": .01}),
            "abc_top_k": ("INT", {"default": 30, "min": 1, "max": 1000}),
            "abc_repetition_penalty": ("FLOAT", {"default": 1.005, "min": .1, "max": 3., "step": .005}),
            "abc_penalty_window": ("INT", {"default": 100, "min": 1, "max": 100}),
            "abc_min_tokens": ("INT", {"default": 32, "min": 0, "max": 8192}),
            "abc_max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
            "semantic_temperature": ("FLOAT", {"default": 1., "min": 0., "max": 5., "step": .01}),
            "semantic_top_p": ("FLOAT", {"default": .95, "min": .01, "max": 1., "step": .01}),
            "semantic_top_k": ("INT", {"default": 100, "min": 1, "max": 2000}),
            "semantic_repetition_penalty": ("FLOAT", {"default": 1.2, "min": .1, "max": 3., "step": .01}),
            "semantic_penalty_window": ("INT", {"default": 50, "min": 1, "max": 100}),
            "semantic_min_tokens": ("INT", {"default": 200, "min": 0, "max": 16384}),
            "semantic_max_tokens": ("INT", {"default": 9000, "min": 1, "max": 16384}),
        }}

    RETURN_TYPES = ("YUE2_SAMPLING", "STRING")
    RETURN_NAMES = ("settings", "info")
    FUNCTION = "build"
    CATEGORY = "YuE2/Advanced"

    def build(self, preset, **values):
        if preset == "official":
            abc, sem = dict(OFFICIAL_ABC), dict(OFFICIAL_SEMANTIC)
        elif preset == "stable":
            abc = {**OFFICIAL_ABC, "temperature": .55, "top_p": .85}
            sem = {**OFFICIAL_SEMANTIC, "temperature": .85, "top_p": .9}
        elif preset == "creative":
            abc = {**OFFICIAL_ABC, "temperature": .9, "top_p": .95, "top_k": 60}
            sem = {**OFFICIAL_SEMANTIC, "temperature": 1.15, "top_p": .98, "top_k": 160}
        else:
            abc = {k[4:]: values[k] for k in values if k.startswith("abc_")}
            sem = {k[9:]: values[k] for k in values if k.startswith("semantic_")}
        if abc["min_tokens"] > abc["max_tokens"] or sem["min_tokens"] > sem["max_tokens"]:
            raise ValueError("min_tokens cannot exceed max_tokens")
        result = {"abc": abc, "semantic": sem, "preset": preset}
        return (result, f"preset={preset} | abc={abc} | semantic={sem}")


class YuE2Plan:
    DESCRIPTION = "Generates only the exact symbolic YuE2 plan for inspection, editing, or later rendering."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipeline": ("YUE2_PIPE",),
            "style": ("STRING", {"multiline": True}),
            "lyrics": ("STRING", {"multiline": True}),
            "cot": (["full", "melody"],),
            "seed": ("INT", {"default": 831001, "min": 0, "max": 2**63 - 1}),
            "cfg_scale": ("FLOAT", {"default": -1., "min": -1., "max": 20., "step": .01}),
        }, "optional": {
            "abc": ("STRING", {"forceInput": True}),
            "sampling_settings": ("YUE2_SAMPLING",),
        }}

    RETURN_TYPES = ("YUE2_PLAN", "STRING", "STRING")
    RETURN_NAMES = ("plan", "abc", "info")
    FUNCTION = "plan"
    CATEGORY = "YuE2/Advanced"

    def plan(self, pipeline, style, lyrics, cot, seed, cfg_scale, abc=None, sampling_settings=None):
        if not style.strip() or not lyrics.strip():
            raise ValueError("style and lyrics must not be empty")
        cfg = None if cfg_scale < 0 else cfg_scale
        started = time.perf_counter()
        result = yue2_model.plan(
            pipeline, style=style.strip(), lyrics=lyrics.strip(), cot=cot,
            seed=seed, abc=(abc or "").strip() or None, cfg_scale=cfg,
            abc_sampling=(sampling_settings or {}).get("abc"))
        info = (f"seed={seed} cot={cot} abc_tokens={len(result.abc_ids)} "
                f"truncated={result.truncated} elapsed={time.perf_counter()-started:.1f}s")
        return (result, result.abc or "", info)


class YuE2RenderPlan:
    DESCRIPTION = "Renders an unchanged YuE2 plan through semantic generation, acoustic synthesis, and VAE decoding."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipeline": ("YUE2_PIPE",), "plan": ("YUE2_PLAN",),
            "ode_steps": ("INT", {"default": 32, "min": 1, "max": 64}),
            "vae_decode": (["tiled", "full"],),
            "vae_tile_frames": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 128}),
            "save_flac": ("BOOLEAN", {"default": True}),
            "save_artifacts": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "sampling_settings": ("YUE2_SAMPLING",),
            "abc_override": ("STRING", {"forceInput": True,
                "tooltip": "Optional edited ABC. It replaces the score in the connected plan while preserving its style, lyrics, seed, and COT mode."}),
        }}

    RETURN_TYPES = ("AUDIO", "YUE2_LATENTS", "STRING", "STRING")
    RETURN_NAMES = ("audio", "latents", "abc", "info")
    FUNCTION = "render"
    CATEGORY = "YuE2/Advanced"

    def render(self, pipeline, plan, ode_steps, vae_decode, vae_tile_frames,
               save_flac, save_artifacts, sampling_settings=None, abc_override=None):
        started = time.perf_counter()
        if (abc_override or "").strip() and (abc_override or "").strip() != (plan.abc or "").strip():
            req=plan.request
            plan=yue2_model.plan(pipeline, style=req.style, lyrics=req.lyrics, cot=req.cot,
                                 seed=req.seed, abc=abc_override.strip(), cfg_scale=req.cfg_scale,
                                 abc_sampling=(sampling_settings or {}).get("abc"))
        semantic, latents, audio = yue2_model.render_plan(
            pipeline, plan, semantic_sampling=(sampling_settings or {}).get("semantic"),
            ode_steps=ode_steps, vae_decode=vae_decode,
            vae_tile_frames=vae_tile_frames or None)
        seconds = len(audio) / 48000
        out_dir = None
        if save_flac:
            import soundfile as sf
            out_dir = timestamp_dir("YuE2")
            sf.write(os.path.join(out_dir, f"plan_s{plan.request.seed}.flac"), audio, 48000, subtype="PCM_24")
        # Full official artifacts need a SongResult; staged outputs retain their
        # exact objects and can still be saved in a compact reproducibility set.
        if save_artifacts:
            import json, numpy as np
            artifact = timestamp_dir("YuE2_Artifacts")
            plan.save(artifact)
            np.save(os.path.join(artifact, "semantic.npy"), np.asarray(semantic.tokens, dtype=np.int32))
            np.save(os.path.join(artifact, "latent.npy"), np.asarray(latents, dtype=np.float32))
            with open(os.path.join(artifact, "staged.json"), "w", encoding="utf-8") as fh:
                json.dump({"semantic_timing": semantic.timing, "semantic_truncated": semantic.truncated,
                           "ode_steps": ode_steps, "vae_decode": vae_decode}, fh, indent=2)
        info = (f"{seconds:.1f}s @48000Hz seed={plan.request.seed} "
                f"semantic_tokens={len(semantic.tokens)} truncated={semantic.truncated} "
                f"elapsed={time.perf_counter()-started:.1f}s")
        if out_dir: info += f" | saved: {out_dir}"
        return (audio_dict(torch.from_numpy(audio).T, 48000), latents, plan.abc or "", info)


class YuE2DecodeLatents:
    DESCRIPTION = "Decodes previously generated YuE2 acoustic latents without rerunning planning or synthesis."
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipeline": ("YUE2_PIPE",), "latents": ("YUE2_LATENTS",),
                             "mode": (["tiled", "full"],)}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "YuE2/Advanced"
    def decode(self, pipeline, latents, mode):
        audio = pipeline.decode(latents, full=mode == "full")
        return (audio_dict(torch.from_numpy(audio).T, 48000),)


class YuE2PlanBatch:
    DESCRIPTION = "Generates several score-only candidates from consecutive seeds; it is not the benchmark best-of-N evaluator."
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipeline": ("YUE2_PIPE",), "style": ("STRING", {"multiline": True}),
                             "lyrics": ("STRING", {"multiline": True}), "cot": (["full", "melody"],),
                             "seed_start": ("INT", {"default": 1000, "min": 0, "max": 2**63-1}),
                             "count": ("INT", {"default": 4, "min": 1, "max": 16})},
                "optional": {"sampling_settings": ("YUE2_SAMPLING",)}}
    RETURN_TYPES = ("YUE2_PLAN_BATCH", "STRING")
    RETURN_NAMES = ("plans", "summary")
    FUNCTION = "run"
    CATEGORY = "YuE2/Advanced"
    def run(self, pipeline, style, lyrics, cot, seed_start, count, sampling_settings=None):
        plans=[]; blocks=[]
        for i in range(count):
            seed=seed_start+i
            p=yue2_model.plan(pipeline, style=style, lyrics=lyrics, cot=cot, seed=seed,
                              abc_sampling=(sampling_settings or {}).get("abc"))
            plans.append(p); blocks.append(f"===== candidate {i} | seed {seed} =====\n{p.abc or ''}")
        return (plans, "\n\n".join(blocks))


class YuE2PlanSelector:
    DESCRIPTION = "Selects one plan from a YuE2 Plan Batch by zero-based index."
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plans": ("YUE2_PLAN_BATCH",), "index": ("INT", {"default": 0, "min": 0, "max": 15})}}
    RETURN_TYPES = ("YUE2_PLAN", "STRING", "INT")
    RETURN_NAMES = ("plan", "abc", "seed")
    FUNCTION = "select"
    CATEGORY = "YuE2/Advanced"
    def select(self, plans, index):
        if not plans: raise ValueError("Plan batch is empty")
        if not 0 <= index < len(plans): raise ValueError(f"index must be 0..{len(plans)-1}")
        p=plans[index]; return (p, p.abc or "", p.request.seed)


NODE_CLASS_MAPPINGS = {"YuE2SamplingSettings": YuE2SamplingSettings, "YuE2Plan": YuE2Plan,
                       "YuE2RenderPlan": YuE2RenderPlan, "YuE2DecodeLatents": YuE2DecodeLatents,
                       "YuE2PlanBatch": YuE2PlanBatch, "YuE2PlanSelector": YuE2PlanSelector}
NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2SamplingSettings": "YuE2 Advanced Sampling Settings",
    "YuE2Plan": "YuE2 Generate Plan", "YuE2RenderPlan": "YuE2 Render Plan",
    "YuE2DecodeLatents": "YuE2 Decode Latents", "YuE2PlanBatch": "YuE2 Plan Batch",
    "YuE2PlanSelector": "YuE2 Plan Selector"}

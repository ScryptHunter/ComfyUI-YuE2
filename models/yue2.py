# -*- coding: utf-8 -*-
"""YuE2 生成管线（加载、缓存、生成）。

管线构造有两个会**影响同进程其他模型**的副作用，必须处理：
- ``torch.cuda.set_per_process_memory_fraction`` 把进程显存上限压到
  ``memory_budget_gib`` 附近 → 构造完立即复位，否则 ComfyUI 加载视频等大模型会莫名 OOM
- 全局 TF32 / matmul 精度开关 → 用 :func:`runtime_flags` 上下文包裹生成过程并恢复
"""
from __future__ import annotations

import contextlib
import gc

import torch

from .paths import resolve, vae_dir

# 进程级管线缓存（键为构造参数）
_cache: dict = {}


def cancelled() -> bool:
    """Return ComfyUI's interrupt state without requiring ComfyUI during tests."""
    try:
        import comfy.model_management as mm
        check = getattr(mm, "processing_interrupted", None)
        return bool(check()) if check else False
    except Exception:
        return False


@contextlib.contextmanager
def runtime_flags():
    """生成期间应用 YuE2 要求的全局精度开关，结束后恢复原值。"""
    saved = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.get_float32_matmul_precision(),
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        (torch.backends.cudnn.benchmark,
         torch.backends.cudnn.deterministic,
         torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32,
         torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction) = saved[:5]
        torch.set_float32_matmul_precision(saved[5])


def _reset_memory_fraction() -> None:
    """复位被 YuE2 管线压低的进程显存配额。"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_per_process_memory_fraction(1.0, i)
            except Exception:
                pass


def _apply_windows_patch() -> None:
    """Windows/torch 构建缺少内置 flash 内核时，把 CUDA Graph 解码切到 cuDNN。"""
    from .yue2_patch import apply_yue2_windows_patch
    apply_yue2_windows_patch()


def force_attention_backend(backend: str) -> bool:
    """强制 GraphAR 使用指定 attention 后端（external-flash 需先注入）。

    返回是否成功。``auto`` 什么都不做（走 yue2_patch 的自动降级逻辑）。
    """
    if backend in ("", "auto"):
        return True
    try:
        import yue2.cuda_graph as cg
    except ImportError:
        return False
    if getattr(cg, "_comfy_yue2_forced_backend", None) == backend:
        return True
    from . import yue2_patch
    if backend == "external-flash":
        if not yue2_patch.apply_external_flash_backend():
            print("[ComfyUI-YuE2] external-flash is unavailable (requires CUDA and "
                  "the pip flash-attn package); falling back to auto")
            return False
    orig_init = cg.GraphAR.__init__

    def patched_init(self, model, prefixes, max_tokens, *, capture=True,
                     attention_backend="auto", fuse_projections=False):
        if attention_backend == "auto":
            attention_backend = backend
        return orig_init(self, model, prefixes, max_tokens, capture=capture,
                         attention_backend=attention_backend,
                         fuse_projections=fuse_projections)

    cg.GraphAR.__init__ = patched_init
    cg._comfy_yue2_forced_backend = backend
    print(f"[ComfyUI-YuE2] Forced attention backend: {backend}")
    return True


def load(model_name: str, vae_name: str = "YuE2-Vae", device: str = "cuda",
         memory_budget_gib: int = 24, offload_ar: bool = False,
         offline: bool = False, attention_backend: str = "auto",
         quantization: str = "none"):
    """创建或复用 YuE2 管线（按参数缓存）。"""
    model_path = resolve("yue2", model_name)
    vae_path = vae_dir(model_path, vae_name)

    # 不用 pip 安装官方 wheel：从模型目录解包到节点私有缓存后导入，避免
    # wheel 内的 torch/transformers/numpy 精确版本约束破坏 ComfyUI 环境。
    from ..compat.yue2_runtime import import_yue2
    yue2 = import_yue2(model_path)

    _apply_windows_patch()
    force_attention_backend(attention_backend)

    key = (model_path, vae_path, device, int(memory_budget_gib), bool(offload_ar),
           bool(offline), attention_backend, quantization)
    cached = _cache.get("pipeline")
    if cached is not None and _cache.get("key") == key:
        return cached

    if cached is not None:
        close()
        torch.cuda.empty_cache()
        gc.collect()

    pipe = yue2.YuE2Pipeline.from_pretrained(
        model_path, vae=vae_path, device=device,
        memory_budget_gib=int(memory_budget_gib), offload_ar=bool(offload_ar),
        local_files_only=bool(offline), progress=False,
        quantization=quantization,
    )
    _reset_memory_fraction()
    _cache["pipeline"] = pipe
    _cache["key"] = key
    return pipe


def close() -> None:
    """释放管线并清空缓存。"""
    pipe = _cache.pop("pipeline", None)
    _cache.pop("key", None)
    if pipe is not None:
        try:
            pipe.close()
        except Exception:
            pass
    torch.cuda.empty_cache()
    gc.collect()


def _with_ode_steps(pipe, ode_steps: int | None):
    """``GenerationConfig`` 是 frozen dataclass，需构造新实例而非就地赋值。"""
    if not ode_steps:
        return
    import dataclasses
    config = pipe.generation_config
    if int(config.ode_steps) == int(ode_steps):
        return
    pipe.generation_config = dataclasses.replace(config, ode_steps=int(ode_steps))


def generate(pipe, *, style: str, lyrics: str, cot: str = "full", seed: int = 831001,
             abc: str | None = None, cfg_scale: float | None = None,
             abc_sampling: dict | None = None, semantic_sampling: dict | None = None,
             ode_steps: int | None = None, on_progress=None,
             vae_decode: str = "tiled", vae_tile_frames: int | None = None,
             on_vae_progress=None):
    """生成歌曲，返回 ``SongResult``。``on_progress`` 在每生成一个 token 时回调。

    vae_decode: ``tiled``=官方分块解码（显存友好）; ``full``=整曲一次解码
    （快，整曲波形驻留 GPU，需大显存; 不足时自动回退 tiled）。
    vae_tile_frames: tiled 模式的块大小（latent 帧数）。None=管线默认
    （预算 ≤12GB 为 512，否则 1024）。更小的值峰值显存更低、块数更多。
    on_vae_progress: ``(completed, total)`` 回调，VAE 分块解码每完成一块调用一次。
    """
    with runtime_flags():
        _with_ode_steps(pipe, ode_steps)
        orig_core = pipe.vae_core_frames
        if vae_tile_frames:
            pipe.vae_core_frames = int(vae_tile_frames)

        if vae_decode == "full":
            # pipe.__call__ 的 decode 步骤写死 tiled；这里临时改写该方法走 full，
            # 显存不足时自动回退官方 tiled 路径。
            import torch as _torch
            orig_decode = pipe.decode

            def full_or_fallback(latents, **kw):
                if on_vae_progress is not None:
                    on_vae_progress(0, 1)
                try:
                    result = orig_decode(latents, full=True)
                except _torch.OutOfMemoryError:
                    print("[ComfyUI-YuE2] Full-song VAE decode ran out of VRAM; "
                          "falling back to tiled decoding")
                    _torch.cuda.empty_cache()
                    result = orig_decode(latents)
                if on_vae_progress is not None:
                    on_vae_progress(1, 1)
                return result

            pipe.decode = full_or_fallback
            try:
                return pipe(style=style, lyrics=lyrics, cot=cot, seed=int(seed),
                            abc=abc, cfg_scale=cfg_scale,
                            abc_sampling=abc_sampling, semantic_sampling=semantic_sampling,
                            cancelled=cancelled, on_token=on_progress)
            finally:
                pipe.decode = orig_decode
                pipe.vae_core_frames = orig_core

        # tiled 路径: 官方 decode() 只有在 pipe.progress=True 时才把每块进度
        # 传给 Progress 阶段（否则 report=None）。这里临时打开 progress 并包装
        # _status，把 "Decoding audio" 阶段的 update 转发到 ComfyUI 进度条。
        if on_vae_progress is not None:
            import contextlib as _cl
            orig_decode = pipe.decode
            orig_status = pipe._status
            orig_progress = pipe.progress

            def decode_with_progress(latents, **kw):
                try:
                    import torch as _torch
                    z = _torch.as_tensor(latents)
                    # latents 可能是 [T,64]（decode 内部才转置），统一取 64 通道维
                    frames = z.shape[0] if z.ndim == 2 and z.shape[1] == 64 else z.shape[-1]
                    core = int(pipe.vae_core_frames)
                    total = max(1, (int(frames) + core - 1) // core)
                except Exception:
                    total = None
                on_vae_progress(0, total or 1)
                pipe.progress = True          # 激活官方 report 闭包

                @_cl.contextmanager
                def hooked_status(label, *, total=None, unit=None):
                    with orig_status(label, total=total, unit=unit) as stage:
                        if label == "Decoding audio":
                            orig_update = stage.update

                            def spy_update(completed, total=None):
                                try:
                                    on_vae_progress(completed, total if total is not None
                                                    else stage.total)
                                except Exception:
                                    pass
                                return orig_update(completed, total=total)

                            stage.update = spy_update
                        yield stage

                pipe._status = hooked_status
                try:
                    return orig_decode(latents, **kw)
                finally:
                    pipe._status = orig_status
                    pipe.progress = orig_progress

            pipe.decode = decode_with_progress
            try:
                return pipe(style=style, lyrics=lyrics, cot=cot, seed=int(seed),
                            abc=abc, cfg_scale=cfg_scale,
                            abc_sampling=abc_sampling, semantic_sampling=semantic_sampling,
                            cancelled=cancelled, on_token=on_progress)
            finally:
                pipe.decode = orig_decode
                pipe.vae_core_frames = orig_core

        try:
            return pipe(style=style, lyrics=lyrics, cot=cot, seed=int(seed),
                        abc=abc, cfg_scale=cfg_scale,
                        abc_sampling=abc_sampling, semantic_sampling=semantic_sampling,
                        cancelled=cancelled, on_token=on_progress)
        finally:
            pipe.vae_core_frames = orig_core


def plan(pipe, *, style: str, lyrics: str, cot: str = "full", seed: int = 831001,
         abc: str | None = None, cfg_scale: float | None = None,
         abc_sampling: dict | None = None, on_progress=None):
    """只生成符号规划（ABC 乐谱），不合成音频。"""
    with runtime_flags():
        request = pipe._request(style=style, lyrics=lyrics, cot=cot, seed=int(seed),
                                abc=abc, cfg_scale=cfg_scale)
        return pipe.plan(request=request, abc_sampling=abc_sampling,
                         cancelled=cancelled, on_token=on_progress)


def render_plan(pipe, plan_result, *, semantic_sampling=None, ode_steps=None,
                vae_decode="tiled", vae_tile_frames=None):
    """Run the official semantic -> NAR -> VAE stages for an exact plan object."""
    with runtime_flags():
        _with_ode_steps(pipe, ode_steps)
        semantic = pipe.generate_semantic(
            plan_result, sampling=semantic_sampling, cancelled=cancelled)
        latents = pipe.synthesize(semantic, cancelled=cancelled)
        if cancelled():
            raise InterruptedError("Cancelled before VAE decode")
        original_core = pipe.vae_core_frames
        if vae_tile_frames:
            pipe.vae_core_frames = int(vae_tile_frames)
        try:
            try:
                audio = pipe.decode(latents, full=vae_decode == "full")
            except torch.OutOfMemoryError:
                if vae_decode != "full":
                    raise
                torch.cuda.empty_cache()
                audio = pipe.decode(latents, full=False)
        finally:
            pipe.vae_core_frames = original_core
        return semantic, latents, audio

# -*- coding: utf-8 -*-
"""YuE2 on Windows / ComfyUI 兼容补丁。

问题:Windows 版 torch 2.10+cu130 构建未编译内置 flash-attention 内核,
官方 yue2_infer 的 CUDA Graph 解码(GraphAR)在 attention_backend="auto" 时
会直接调用 torch.ops.aten._flash_attention_forward 并报
"USE_FLASH_ATTENTION was not enabled for build"。

方案:运行时探测该算子是否真的可用;不可用则把 GraphAR 切到 cuDNN SDPA 后端
(实测支持 GQA、bool mask、且可被 CUDA Graph 捕获)。Linux 正常构建不受影响。

另外 yue2 管线构造时会设置进程级 torch.cuda.set_per_process_memory_fraction,
会限制同进程 ComfyUI 其他模型的显存分配,构造完成后必须复位。
"""
from __future__ import annotations

import contextlib

import torch

_probe_cache: dict = {}


def builtin_flash_available(device="cuda") -> bool:
    """探测 torch 内置 varlen flash-attention 算子是否真实可用(带缓存)。"""
    key = str(device)
    if key in _probe_cache:
        return _probe_cache[key]
    ok = False
    if torch.cuda.is_available() and hasattr(torch.ops.aten, "_flash_attention_forward"):
        try:
            dev = torch.device(device)
            q = torch.zeros(1, 1, 4, 128, device=dev, dtype=torch.bfloat16)
            k = torch.zeros(1, 1, 2, 128, device=dev, dtype=torch.bfloat16)
            v = torch.zeros(1, 1, 2, 128, device=dev, dtype=torch.bfloat16)
            cu_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
            cu_k = torch.tensor([0, 128], dtype=torch.int32, device=dev)
            used = torch.tensor([128], dtype=torch.int32, device=dev)
            out = torch.ops.aten._flash_attention_forward(
                q, k, v, cu_q, cu_k, 1, 128, 0.0, False, False, seqused_k=used)
            ok = bool(torch.isfinite(out[0].float()).all().item())
        except Exception:
            ok = False
    _probe_cache[key] = ok
    return ok


def _fuse_supported(model) -> bool:
    """fuse_projections 要求无 bias 的标准 Linear (YuE2 AR 满足), fp8 时不融合。"""
    if getattr(model, "_yue2_fp8_originals", None):
        return False
    try:
        for layer in model.model.layers:
            for module in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                           layer.self_attn.v_proj, layer.self_attn.o_proj,
                           layer.mlp.gate_proj, layer.mlp.up_proj):
                if not isinstance(module, torch.nn.Linear) or module.bias is not None:
                    return False
        return True
    except Exception:
        return False


def apply_yue2_windows_patch() -> bool:
    """给 yue2.cuda_graph.GraphAR 打自动后端降级补丁(幂等)。"""
    try:
        import yue2.cuda_graph as cg
    except ImportError:
        return False
    if getattr(cg, "_comfy_yue2_flash_patch", False):
        return True
    if builtin_flash_available("cuda"):
        cg._comfy_yue2_flash_patch = True
        return True
    orig_init = cg.GraphAR.__init__

    def patched_init(self, model, prefixes, max_tokens, *, capture=True,
                     attention_backend="auto", fuse_projections=False):
        if attention_backend == "auto":
            dev = None
            try:
                dev = next(model.parameters()).device
            except StopIteration:
                pass
            if dev is not None and dev.type == "cuda" and not builtin_flash_available(dev):
                attention_backend = "cudnn"
        # fuse_projections 合并 qkv/gate_up 权重矩阵, 少一半 GEMV kernel 启动;
        # 官方默认关闭 (要求无 bias 的 checkpoint, YuE2 满足), 实测 +3%。
        return orig_init(self, model, prefixes, max_tokens, capture=capture,
                         attention_backend=attention_backend,
                         fuse_projections=fuse_projections or _fuse_supported(model))

    cg.GraphAR.__init__ = patched_init
    cg._comfy_yue2_flash_patch = True
    print("[ComfyUI-YuE2] Built-in Flash Attention is unavailable; "
          "CUDA Graph decoding will use the cuDNN SDPA backend")
    return True


# ---------------------------------------------------------------------------
# external-flash 后端: 用环境里 pip 安装的 flash-attn 包替代 torch 内置算子。
#
# GraphAR 的 KV 缓存是 (branches, capacity, kv_heads, head_dim) 的
# sequence-major 固定容量张量 —— 与 flash_attn_with_kvcache 的
# (batch, max_seqlen, nheads_k, head_dim) 完全同构, 零拷贝对接:
#   q/k/v: 当前 token, 形状 (branches, 1, heads, head_dim)
#   cache_seqlens = positions + 1 (每支已完成 token 数)
# CUDA Graph 捕获要求: 张量地址固定(满足, 预分配), 无 CPU 同步
# (causal=True 是编译期常量, 不产生同步)。
# ---------------------------------------------------------------------------

_ext_flash_probe: dict = {}


def external_flash_available(device="cuda") -> bool:
    """探测 pip flash-attn 的 kvcache API 是否真实可用(带缓存)。"""
    key = str(device)
    if key in _ext_flash_probe:
        return _ext_flash_probe[key]
    ok = False
    try:
        from flash_attn import flash_attn_with_kvcache  # noqa: F401
        import torch
        dev = torch.device(device)
        q = torch.zeros(1, 1, 4, 128, device=dev, dtype=torch.bfloat16)
        cache = torch.zeros(1, 8, 2, 128, device=dev, dtype=torch.bfloat16)
        k = torch.zeros(1, 1, 2, 128, device=dev, dtype=torch.bfloat16)
        seqlens = torch.tensor([4], dtype=torch.int32, device=dev)
        out = flash_attn_with_kvcache(q, cache, cache, k, k,
                                      cache_seqlens=seqlens, causal=True)
        ok = bool(torch.isfinite(out.float()).all().item())
    except Exception:
        ok = False
    _ext_flash_probe[key] = ok
    return ok


def apply_external_flash_backend() -> bool:
    """给 GraphAR 注入 ``external-flash`` attention 后端(幂等)。

    成功后 ``GraphAR(..., attention_backend="external-flash")`` 会改走
    pip flash-attn 的 ``flash_attn_with_kvcache``。返回是否注入成功。
    """
    try:
        import yue2.cuda_graph as cg
    except ImportError:
        return False
    if getattr(cg, "_comfy_yue2_ext_flash", False):
        return True
    if not external_flash_available("cuda"):
        return False

    import flash_attn as _fa_pkg
    from flash_attn import flash_attn_with_kvcache
    import torch
    from torch.nn import functional as F

    _VALID = {"auto", "flash", "cudnn", "sdpa", "external-flash"}

    orig_init = cg.GraphAR.__init__

    def patched_init(self, model, prefixes, max_tokens, *, capture=True,
                     attention_backend="auto", fuse_projections=False):
        self._ext_flash_requested = False
        if attention_backend == "external-flash":
            dev = None
            try:
                dev = next(model.parameters()).device
            except StopIteration:
                pass
            if dev is None or dev.type != "cuda" or not external_flash_available(dev):
                raise ValueError(
                    "external-flash requires CUDA and a working pip flash-attn "
                    f"package (version {_fa_pkg.__version__})")
            # 原版校验只认 auto/flash/cudnn/sdpa; 以 cudnn 走完校验与初始化
            # (visible mask 等张量照常创建), 随后替换后端标志, _decode 分流。
            self._ext_flash_requested = True
            attention_backend = "cudnn"
        orig_init(self, model, prefixes, max_tokens, capture=capture,
                  attention_backend=attention_backend,
                  fuse_projections=fuse_projections)
        if self._ext_flash_requested:
            self.attention_backend = "external-flash"

    orig_decode = cg.GraphAR._decode

    def patched_decode(self):
        if self.attention_backend != "external-flash":
            return orig_decode(self)
        backbone = self.model.model
        cos, sin = backbone.rotary_emb(self.positions[:, None])
        x = backbone.embed_tokens(self.tokens)
        config = self.model.config
        slots = self.positions[:, None, None, None].expand(
            self.branches, 1, config.num_key_value_heads, config.head_dim)
        seqlens = (self.positions + 1).to(torch.int32)
        for layer, keys, values in zip(backbone.layers, self.keys, self.values):
            normalized = layer.input_layernorm(x)
            q, k, v = layer.self_attn.project_qkv(normalized, cos, sin)
            keys.scatter_(1, slots, k)
            values.scatter_(1, slots, v)
            # cache 已含当前 token (上面 scatter_ 写入), k/v 必须传 None ——
            # 再传会把同一 token 追加在 seqlens 位置重复计算 (且原地写 cache,
            # 破坏 CUDA Graph 捕获的固定地址约定)。
            h = flash_attn_with_kvcache(
                q, keys, values, None, None,
                cache_seqlens=seqlens, causal=False,
                softmax_scale=config.head_dim ** -0.5)
            x = x + layer.self_attn.o_proj(h.reshape(self.branches, 1, -1))
            normalized = layer.post_attention_layernorm(x)
            x = x + layer.mlp(normalized)
        output = self.model.lm_head(backbone.norm(x))[:, 0]
        self.positions.add_(1)
        return output

    cg.GraphAR.__init__ = patched_init
    cg.GraphAR._decode = patched_decode
    cg._VALID_BACKENDS = _VALID
    cg._comfy_yue2_ext_flash = True
    print(f"[ComfyUI-YuE2] external-flash backend enabled (pip flash-attn "
          f"{_fa_pkg.__version__}); GraphAR(attention_backend='external-flash') is available")
    return True


def reset_cuda_memory_fraction():
    """yue2 管线构造会把进程显存上限压到 memory_budget 附近,必须复位为不限制,
    否则同进程的 ComfyUI 视频等大模型会莫名 OOM。"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(1.0, i)


@contextlib.contextmanager
def yue2_runtime_flags():
    """生成期间应用 yue2 官方要求的全局精度/确定性开关,结束后恢复 ComfyUI 原值。"""
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

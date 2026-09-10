# -*- coding: utf-8 -*-
"""SheetSage2 在 transformers 5.x 下的兼容处理。

通用 API 差异（``_tied_weights_keys``、``tie_weights`` 签名等）由
``compat.transformers5`` 统一处理；这里只保留 SheetSage2 特有的两项：

1. ``BartDecoder.__init__`` 的 ``embed_tokens`` 关键字参数（4.x 有，5.x 已移除）
2. MERT2 编码器 ``RotaryEmbedding.inv_freq`` buffer 被 5.x 的 meta 初始化
   写入垃圾数据——**这是最隐蔽的一项**：模型不会报错，只是转录结果里
   完全没有旋律（melody 字段恒为 0），听感上像"只识别出节奏和和弦"。

对照验证：transformers 4.45.2（官方锁定版本）输出 vocal=23 / ins=38 音符；
修复后 5.3.0 输出与之完全一致。
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading

import torch

from .transformers5 import apply_transformers5_compat


_LOAD_PATCH_LOCK = threading.RLock()
_INTENTIONALLY_TIED = {
    "decoder.embed_tokens.weight",
    "output_projection.weight",
}


@contextlib.contextmanager
def _ignore_intentionally_missing_tied_weights(model_cls, base_model_path=None):
    """Make transformers 5.x reproduce the 4.x tied-weight load report.

    SheetSage2's own ``from_pretrained`` asks the base Transformers loader for
    ``output_loading_info`` and rejects any missing key before it gets a chance
    to tie weights.  Transformers 4.x removed names listed in the legacy
    ``_tied_weights_keys`` list; 5.x no longer does so for this old checkpoint.
    Filter only the two known aliases, only while loading this exact class.
    """
    from transformers import PreTrainedModel

    with _LOAD_PATCH_LOCK:
        original_descriptor = PreTrainedModel.__dict__["from_pretrained"]
        original_func = original_descriptor.__func__

        def filtered(base_cls, *args, **kwargs):
            result = original_func(base_cls, *args, **kwargs)
            if base_cls is model_cls and kwargs.get("output_loading_info", False):
                model, info = result
                missing = info.get("missing_keys") or []
                if isinstance(missing, set):
                    info["missing_keys"] = missing - _INTENTIONALLY_TIED
                else:
                    info["missing_keys"] = [
                        name for name in missing if name not in _INTENTIONALLY_TIED
                    ]
                return model, info
            return result

        PreTrainedModel.from_pretrained = classmethod(filtered)

        # For a local MERT2 directory, bypass AutoModel's trust_remote_code
        # machinery.  Otherwise Transformers copies local Python files into
        # ~/.cache/huggingface/modules even though the weights are local.  The
        # verified MERT2 implementation is already imported alongside the
        # SheetSage2 class, so load the local weights with that class directly.
        owner_module = sys.modules.get(model_cls.__module__)
        original_auto_model = getattr(owner_module, "AutoModel", None)
        if base_model_path and original_auto_model is not None:
            local_base = os.path.normcase(os.path.abspath(base_model_path))
            mert2_cls = getattr(owner_module, "MERT2Model", None)

            class LocalParentAutoModel:
                @classmethod
                def from_pretrained(proxy_cls, name_or_path, *args, **kwargs):
                    candidate = os.path.normcase(os.path.abspath(str(name_or_path)))
                    if candidate == local_base and mert2_cls is not None:
                        for key in ("trust_remote_code", "code_revision", "revision",
                                    "token", "cache_dir", "force_download"):
                            kwargs.pop(key, None)
                        kwargs["local_files_only"] = True
                        return mert2_cls.from_pretrained(name_or_path, *args, **kwargs)
                    return original_auto_model.from_pretrained(
                        name_or_path, *args, **kwargs)

            owner_module.AutoModel = LocalParentAutoModel
        try:
            yield
        finally:
            if owner_module is not None and original_auto_model is not None:
                owner_module.AutoModel = original_auto_model
            PreTrainedModel.from_pretrained = original_descriptor


def _fix_bart_decoder() -> None:
    from transformers.models.bart import modeling_bart as mb

    if getattr(mb.BartDecoder, "_ss2_compat_patched", False):
        return
    orig_init = mb.BartDecoder.__init__

    def patched_init(self, config, embed_tokens=None):
        orig_init(self, config)
        if embed_tokens is not None:
            # 4.x 语义: decoder 与外部 embedding 共享同一权重
            self.embed_tokens.weight = embed_tokens.weight

    mb.BartDecoder.__init__ = patched_init
    mb.BartDecoder._ss2_compat_patched = True


def fix_rotary_inv_freq(model) -> int:
    """重算被 transformers 5.x meta 初始化污染的 inv_freq buffer（幂等）。"""
    fixed = 0
    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        head_dim = getattr(module, "head_dim", None)
        base = getattr(module, "base", None)
        if inv_freq is None or head_dim is None or base is None:
            continue
        if getattr(module, "_ss2_inv_freq_ok", False):
            continue
        expected = 1.0 / (base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        if not torch.isfinite(inv_freq).all() or not torch.allclose(
                inv_freq.float().cpu(), expected, atol=1e-6):
            with torch.no_grad():
                module.inv_freq = expected.to(inv_freq.device)
            fixed += 1
        module._ss2_inv_freq_ok = True
    return fixed


def apply_sheetsage2_compat(model_path: str):
    """应用补丁并返回 SheetSage2Model 类（幂等）。"""
    apply_transformers5_compat(verbose=False)
    _fix_bart_decoder()

    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module(
        "modeling_sheetsage2.SheetSage2Model", model_path)

    # transformers 4.x accepted a list of tied/missing weight names.  In 5.x
    # this became an explicit {target: source} mapping.  These two tensors are
    # intentionally absent from the adapter checkpoint: both share
    # token_embedding.weight and must not be reported as corrupt weights.
    cls._tied_weights_keys = {
        "decoder.embed_tokens.weight": "token_embedding.weight",
        "output_projection.weight": "token_embedding.weight",
    }
    if not getattr(cls, "_ss2_tie_patched", False):
        orig_tie = cls.tie_weights

        def patched_tie(self, *args, **kwargs):
            kwargs.pop("recompute_mapping", None)
            kwargs.pop("missing_keys", None)
            return orig_tie(self)

        cls.tie_weights = patched_tie
        cls._ss2_tie_patched = True
    return cls


def load_sheetsage2(model_path: str, device: str = "cuda", dtype=torch.bfloat16,
                    base_model_path: str | None = None):
    """加载 SheetSage2（处理 LoRA 合并、权重绑定、inv_freq 修复）。"""
    cls = apply_sheetsage2_compat(model_path)
    kwargs = {"trust_remote_code": True}
    if base_model_path:
        # 官方 SheetSage2 类原生支持 base_model_path。使用本地目录后，父模型
        # 权重和代码均从该目录读取，不会下载到 Hugging Face 缓存。
        kwargs["base_model_path"] = base_model_path
    with _ignore_intentionally_missing_tied_weights(cls, base_model_path):
        model = cls.from_pretrained(model_path, **kwargs).eval()
    # 5.x 下该绑定可能解绑，与模型自身 tie_weights 的语义保持一致
    model.decoder.embed_tokens.weight = model.token_embedding.weight
    model.output_projection.weight = model.token_embedding.weight
    fix_rotary_inv_freq(model)
    model = model.to(device=device, dtype=dtype)
    fix_rotary_inv_freq(model)
    return model

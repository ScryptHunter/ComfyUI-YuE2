# -*- coding: utf-8 -*-
"""transformers 5.x 兼容层。

transformers 5.0 起移除了若干 4.x 的公共 API，而 Qwen3-ASR、SheetSage2 等
模型代码仍按 4.x 编写。本模块必须在导入这些模型代码之前执行，注入等价实现。

已处理:
- ``utils.generic.check_model_inputs`` → ``utils.output_capturing.capture_outputs``（5.x 改名，
  语义相同：按 ``output_*`` 请求惰性安装 hook 采集中间层输出）
- ``modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"]``（5.x 移出公共注册表，
  改为各模型自带 ``compute_default_rope_parameters``）

调用 ``apply_transformers5_compat()`` 幂等，可重复调用。
"""
from __future__ import annotations

import torch

_applied: list[str] = []


def _patch_check_model_inputs() -> None:
    try:
        import transformers.utils.generic as generic
    except ImportError:
        return
    if hasattr(generic, "check_model_inputs"):
        return
    try:
        from transformers.utils.output_capturing import capture_outputs
    except ImportError:
        return
    generic.check_model_inputs = capture_outputs
    _applied.append("check_model_inputs→capture_outputs")


def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **rope_kwargs):
    """transformers 4.x 的默认 RoPE 实现（5.x 未在公共注册表中提供）。"""
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    else:
        base = getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (
        torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, 1.0


def patch_rotary_embedding_classes(namespace: dict | None = None, module=None) -> list[str]:
    """给命名空间/模块中所有 ``*RotaryEmbedding*`` 类补上 5.x 需要的
    ``compute_default_rope_parameters`` staticmethod。

    在导入按 4.x 编写的服务代码之后调用。返回被修补的类名列表。
    """
    import inspect as _inspect
    import torch.nn as nn

    targets = []
    if namespace:
        for name, obj in list(namespace.items()):
            if _inspect.isclass(obj):
                targets.append(obj)
    if module is not None:
        for name in dir(module):
            obj = getattr(module, name, None)
            if _inspect.isclass(obj):
                targets.append(obj)

    patched = []
    seen = set()
    for klass in targets:
        if klass in seen:
            continue
        seen.add(klass)
        if ("RotaryEmbedding" in klass.__name__
                and issubclass(klass, nn.Module)
                and "compute_default_rope_parameters" not in klass.__dict__):
            klass.compute_default_rope_parameters = staticmethod(_compute_default_rope_parameters)
            patched.append(klass.__name__)
    return patched


def patch_rope_default() -> None:
    """5.x 的 RoPE 有两处变更，都会打断按 4.x 编写的模型代码：

    1. ``ROPE_INIT_FUNCTIONS`` 不再提供 ``"default"`` 键（4.x 代码会 KeyError）。
    2. ``modeling_utils._init_weights`` 在 ``rope_type == "default"`` 时改为调用
       ``module.compute_default_rope_parameters``（5.x 模型定义为 staticmethod），
       4.x 的 RoPE 类没有这个方法。

    第 2 项依赖具体类，需在导入那些模块之后用
    :func:`patch_rotary_embedding_classes` 再补一次。
    """
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except ImportError:
        return
    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
        _applied.append("ROPE_INIT_FUNCTIONS['default']")


def _patch_config_defaults() -> None:
    """transformers 5.x 的 ``PretrainedConfig`` 不再为未显式设置的字段提供默认值
    （4.x 会返回 None/False）。大量按 4.x 编写的模型代码直接读取这些属性。"""
    try:
        from transformers import PretrainedConfig
    except ImportError:
        return
    if getattr(PretrainedConfig, "_yue2_defaults_patched", False):
        return

    # 4.x 的 PretrainedConfig.__init__ 会设置的属性及其默认值。
    # 注意：绝不要为 max_length/min_length/new_tokens 等生成控制字段提供默认值——
    # 5.x 会检测到它们被"设置"并拒绝生成。
    _defaults = {
        "pad_token_id": None, "bos_token_id": None, "eos_token_id": None,
        "decoder_start_token_id": None,
        "use_cache": True, "tie_word_embeddings": True,
        "add_cross_attention": False,
        "vocab_size": None, "hidden_size": None, "num_attention_heads": None,
        "num_hidden_layers": None, "num_key_value_heads": None,
        "intermediate_size": None, "head_dim": None,
        "hidden_act": None, "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02, "layer_norm_eps": 1e-12,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "rope_scaling": None,
        "partial_rotary_factor": 1.0, "max_position_embeddings": 512,
        "output_attentions": False, "output_hidden_states": False,
        "return_dict": True, "is_encoder_decoder": False,
        "torchscript": False, "pruned_heads": None,
        "problem_type": None, "chunk_size_feed_forward": 0,
        "architectures": None, "_name_or_path": "",
    }

    original_getattr = PretrainedConfig.__getattribute__

    def patched_getattr(self, key):
        try:
            return original_getattr(self, key)
        except AttributeError:
            if key in _defaults:
                # 首次访问时落到实例上，避免每次触发异常
                try:
                    object.__setattr__(self, key, _defaults[key])
                except Exception:
                    pass
                return _defaults[key]
            raise

    PretrainedConfig.__getattribute__ = patched_getattr
    PretrainedConfig._yue2_defaults_patched = True
    _applied.append("PretrainedConfig legacy defaults")


def _patch_tied_weights_keys() -> None:
    """transformers 5.x 要求 ``_tied_weights_keys`` 是 ``{target: source}`` 字典，
    4.x 惯用的 ``["key", ...]`` 列表会在 ``get_expanded_tied_weights_keys`` 里崩溃。

    该方法本身是 5.x 引入的；4.x（如 yue2-infer 依赖的 4.57.x）没有它，
    列表写法在 4.x 下本来就合法，无需补丁——探测不到该方法就直接跳过。
    """
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return
    if getattr(PreTrainedModel, "_yue2_tied_patched", False):
        return
    if not hasattr(PreTrainedModel, "get_expanded_tied_weights_keys"):
        return
    original = PreTrainedModel.get_expanded_tied_weights_keys

    def patched(self, all_submodels=False):
        for klass in type(self).__mro__:
            twk = klass.__dict__.get("_tied_weights_keys")
            if isinstance(twk, list):
                try:
                    klass._tied_weights_keys = {k: k for k in twk}
                except Exception:
                    pass
        return original(self, all_submodels)

    PreTrainedModel.get_expanded_tied_weights_keys = patched
    PreTrainedModel._yue2_tied_patched = True
    _applied.append("_tied_weights_keys list→dict")


def _patch_tie_weights_signature() -> None:
    """5.x 给 ``tie_weights`` 增加了 ``recompute_mapping``/``missing_keys`` 参数，
    在 ``init_weights``/``from_pretrained`` 内部以关键字传入。覆盖了该方法的 4.x 子类
    因签名不匹配而报 ``unexpected keyword argument``。

    在 ``__init_subclass__`` 上拦截：凡子类自定义的 4.x 风格 ``tie_weights(self)``
    一律包一层容忍版，覆盖所有调用点。
    """
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return
    if getattr(PreTrainedModel, "_yue2_tie_sig_patched", False):
        return

    import functools
    import inspect

    # Inspect the descriptor on the class itself.  Reading
    # ``PreTrainedModel.__init_subclass__`` directly can return an inherited,
    # already-bound builtin (notably with some Transformers/PyTorch pairs).
    # Calling that object as ``original(cls)`` passes cls twice and raises:
    #   __init_subclass__() takes no arguments (1 given)
    original_descriptor = PreTrainedModel.__dict__.get("__init_subclass__")
    if isinstance(original_descriptor, classmethod):
        original_func = original_descriptor.__func__
    else:
        original_func = original_descriptor

    def patched(cls, **kwargs):
        tie = cls.__dict__.get("tie_weights")
        if tie is not None and not getattr(tie, "_yue2_tolerant", False):
            try:
                params = inspect.signature(tie).parameters
            except (TypeError, ValueError):
                params = {}
            if "recompute_mapping" not in params:
                @functools.wraps(tie)
                def tolerant(self, *args, **kw):
                    kw.pop("recompute_mapping", None)
                    kw.pop("missing_keys", None)
                    return tie(self, *args, **kw)

                tolerant._yue2_tolerant = True
                cls.tie_weights = tolerant
        if original_func is not None:
            return original_func(cls, **kwargs)
        # PreTrainedModel did not define the hook. Preserve the inherited MRO
        # behavior and bind the next implementation to the new subclass.
        return super(PreTrainedModel, cls).__init_subclass__(**kwargs)

    PreTrainedModel.__init_subclass__ = classmethod(patched)
    PreTrainedModel._yue2_tie_sig_patched = True

    # 已在补丁前创建的既有子类也一并修复
    for klass in _iter_subclasses(PreTrainedModel):
        tie = klass.__dict__.get("tie_weights")
        if tie is not None and not getattr(tie, "_yue2_tolerant", False):
            try:
                params = inspect.signature(tie).parameters
            except (TypeError, ValueError):
                params = {}
            if "recompute_mapping" not in params:
                @functools.wraps(tie)
                def tolerant(self, *args, __tie=tie, **kw):
                    kw.pop("recompute_mapping", None)
                    kw.pop("missing_keys", None)
                    return __tie(self, *args, **kw)

                tolerant._yue2_tolerant = True
                klass.tie_weights = tolerant

    _applied.append("tie_weights signature compatibility")


def _iter_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_subclasses(sub)


def patch_fix_mistral_regex() -> None:
    """绕过 transformers 5.3.0 的一处调用 bug。

    ``TokenizersBackend.__init__`` 在 vocab>100000 且有 pre_tokenizer 时调用
    ``_patch_mistral_regex``，同时以显式关键字和 ``**kwargs`` 传入
    ``fix_mistral_regex``。只要调用方传了这个参数就会抛
    ``TypeError: got multiple values for keyword argument``——即该参数在
    5.3.0 下完全不可用（受影响模型一律崩溃）。

    这里在进入 ``__init__`` 前摘掉该键，使其退化为未传（走 ``None`` 分支，
    非 Mistral 模型本就如此）。相比上游的必然崩溃，这是严格的改进。
    仅影响 Mistral 系 tokenizer 的一个可选修正标志，与本插件模型无关。
    """
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
    except ImportError:
        return
    if getattr(TokenizersBackend, "_yue2_fmr_patched", False):
        return
    original_init = TokenizersBackend.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.pop("fix_mistral_regex", None)
        return original_init(self, *args, **kwargs)

    TokenizersBackend.__init__ = patched_init
    TokenizersBackend._yue2_fmr_patched = True
    _applied.append("fix_mistral_regex duplicate-argument workaround")


def patch_mask_utils() -> None:
    """4.x 的 ``create_causal_mask`` 参数名是 ``input_embeds``，5.x 改为
    ``inputs_embeds``。按 4.x 编写的模型代码会因此 TypeError，这里让两个名字都可用。"""
    try:
        from transformers import masking_utils
    except ImportError:
        return
    if getattr(masking_utils, "_yue2_mask_patched", False):
        return
    original = masking_utils.create_causal_mask

    def tolerant(*args, **kwargs):
        if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        return original(*args, **kwargs)

    masking_utils.create_causal_mask = tolerant
    masking_utils._yue2_mask_patched = True
    _applied.append("create_causal_mask(input_embeds) compatibility")


def patch_generate_model_kwargs(model_cls, forward_delegate: str = "thinker",
                                extra_keys: tuple = ()) -> list[str]:
    """让 5.x 的 ``GenerationMixin._validate_model_kwargs`` 接受多模态生成参数。

    5.x 会检查 ``prepare_inputs_for_generation`` 的签名；若其中含 ``**kwargs``，
    还会并上 ``self.forward`` 的参数表，然后丢弃"未被使用"的 model_kwargs。
    外层包装类（如 ``Qwen3ASRForConditionalGeneration``）只实现了 ``generate``，
    没有 ``forward``，于是 ``input_features`` / ``feature_attention_mask`` 被判定为
    未使用并被**在进入模型前丢弃**——音频因此永远到不了模型（表现为转录结果为空）。

    这里给该类补一个真实的 ``forward``，其签名显式声明委托目标的参数与
    ``extra_keys``，从而让这些键被识别为"已使用"。返回被修补的类名列表。
    """
    if getattr(model_cls, "_yue2_fwd_patched", False):
        return []

    import inspect

    # 从委托目标的 forward 取参数（类上可能没有，运行期由实例提供）
    params = []
    try:
        target = getattr(model_cls, forward_delegate + "_forward_template", None)
        if target is None:
            # 通过已有的 generate 推断：直接声明所有已知多模态键
            params = list(extra_keys)
        else:
            params = list(inspect.signature(target).parameters)
    except Exception:
        params = list(extra_keys)
    if not params:
        params = list(extra_keys) or ["input_features", "feature_attention_mask", "attention_mask"]

    def forward(self, *args, **kwargs):
        return getattr(self, forward_delegate)(*args, **kwargs)

    # 伪造签名：5.x 只做 inspect.signature 的名字集合比较
    forward.__name__ = "forward"
    forward.__signature__ = inspect.Signature(
        parameters=[inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)] +
                   [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
                    for name in params],
        return_annotation=inspect.Signature.empty,
    )
    model_cls.forward = forward
    model_cls._yue2_fwd_patched = True
    return [model_cls.__name__]


def apply_transformers5_compat(verbose: bool = True) -> list[str]:
    """注入 transformers 5.x 缺失的 4.x API（幂等）。返回本次新应用的补丁名。

    注意：依赖具体模型类的补丁（RoPE 方法、generate 的 forward 委托）
    需在导入对应服务代码后，另外调用
    :func:`patch_rotary_embedding_classes` 与 :func:`patch_generate_model_kwargs`。
    """
    global _applied
    _applied = []
    _patch_check_model_inputs()
    patch_rope_default()
    _patch_config_defaults()
    _patch_tied_weights_keys()
    _patch_tie_weights_signature()
    patch_fix_mistral_regex()
    patch_mask_utils()
    if verbose and _applied:
        try:
            import transformers
            version = transformers.__version__
        except Exception:
            version = "?"
        print(f"[compat] transformers {version}: " + ", ".join(_applied))
    return list(_applied)

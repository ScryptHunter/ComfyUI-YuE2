"""Request-scoped canonical adapter stacks; shared legacy weights are never merged."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass,field
import threading
import torch
from .adapters import Yue2Adapter
from .canonical import (AdapterStack,AdapterStackEntry,CanonicalAdapterBundle,
                        CanonicalPatch,stack_from_legacy)

_LOCK=threading.RLock()

class RuntimeDelta(torch.nn.Module):
    def __init__(self,pair,strength):
        super().__init__();self.register_buffer('A',pair.A.clone());self.register_buffer('B',pair.B.clone());self.strength=float(strength)
    def forward(self,module,args,output):
        delta=(args[0].float()@self.A.float().T)@self.B.float().T
        return output+(delta*self.strength).to(output.dtype)

class RuntimeDenseDelta(torch.nn.Module):
    def __init__(self,weight,bias,strength):
        super().__init__();self.register_buffer('weight',weight);self.register_buffer('bias',bias);self.strength=float(strength)
    def forward(self,module,args,output):
        delta=0
        if self.weight is not None:delta=args[0].float()@self.weight.float().T
        if self.bias is not None:delta=delta+self.bias.float()
        return output+(delta*self.strength).to(output.dtype)

@dataclass(frozen=True)
class LegacyLoraPipe:
    base:object
    adapter:Yue2Adapter|None=None
    stack:AdapterStack=field(default_factory=AdapterStack)
    def __post_init__(self):
        if self.adapter is not None and not self.stack.entries:
            object.__setattr__(self,'stack',stack_from_legacy(self.adapter))
    def decode(self,*args,**kwargs):
        with _LOCK:return self.base.decode(*args,**kwargs)

def _target_module(model,target):
    if target.startswith('model.layers.') or target in ('llm2vae','vae2llm','time_embedder.mlp.0','time_embedder.mlp.2'):
        try: return model.get_submodule(target)
        except (AttributeError, IndexError) as exc: raise ValueError(f'Adapter layer count/target is absent from YuE2 model: {target}') from exc
    raise ValueError(f'Unsupported canonical legacy YuE2 target: {target}')

def stack_bindings(model,stack):
    result=[]
    for entry in stack.entries:
        if not entry.enabled:continue
        for patch in entry.bundle.patches:
            strength=entry.strength_for(patch.branch)
            if not strength:continue
            module=_target_module(model,patch.target)
            if not isinstance(module,torch.nn.Linear):
                raise ValueError(f'Canonical adapter requires an unquantized legacy Linear: {patch.target}')
            if patch.algorithm=='lora':
                patch.validate();patch.validate_shape(module.weight.shape) if hasattr(patch,'validate_shape') else None
                if patch.down.shape[1]!=module.in_features or patch.up.shape[0]!=module.out_features:
                    raise ValueError(f'Adapter shape mismatch: {patch.target}')
                hook=RuntimeDelta(type('Pair',(),{'A':patch.down,'B':patch.up})(),strength*patch.intrinsic_scale)
            else:
                delta=patch.dense_delta(module.weight.shape)
                bias=patch.bias_delta
                if delta is not None and tuple(delta.shape)!=tuple(module.weight.shape):
                    raise ValueError(f'Adapter delta shape mismatch: {patch.target}')
                if bias is not None and (module.bias is None or tuple(bias.shape)!=tuple(module.bias.shape)):
                    raise ValueError(f'Adapter bias shape mismatch: {patch.target}')
                if delta is not None:delta=delta.float()
                if bias is not None:bias=bias.float()*patch.intrinsic_scale
                hook=RuntimeDenseDelta(delta,bias,strength)
            result.append((module,hook,patch.target))
    return result

def bindings(model,adapter):
    """Compatibility bridge for the previous two-branch FL API."""
    return [(m,h) for m,h,_ in stack_bindings(model,stack_from_legacy(adapter))]

def validate_runtime(pipe):
    if getattr(pipe,'quantization','none')!='none':raise ValueError('Legacy YuE2 adapters require quantization=none (FP8 is unsupported)')
    if getattr(pipe,'backend','torch') not in ('torch','torch-eager'):raise ValueError('Legacy YuE2 adapters require torch/torch-eager, not vLLM')

def _to_entry(value,ar_strength=1.,nar_strength=1.,shared_strength=None,enabled=True):
    if isinstance(value,Yue2Adapter):
        stack=stack_from_legacy(value)
        return stack.entries[0] if stack.entries else None
    if isinstance(value,AdapterStackEntry):return value
    if isinstance(value,CanonicalAdapterBundle):
        return AdapterStackEntry(value,float(ar_strength),float(nar_strength),
                                 None if shared_strength is None else float(shared_strength),bool(enabled))
    raise TypeError(f'Unsupported YuE2 adapter object: {type(value).__name__}')

def apply_legacy(pipe,adapter,ar_strength=1.,nar_strength=1.,enabled=True,*,replace_existing=False):
    base=pipe.base if isinstance(pipe,LegacyLoraPipe) else pipe
    if adapter is None:return base
    entry=_to_entry(adapter,ar_strength,nar_strength,enabled=enabled)
    if entry is None:return base
    stack=AdapterStack() if replace_existing else (pipe.stack if isinstance(pipe,LegacyLoraPipe) else AdapterStack())
    stack=stack.append(entry)
    if stack.is_empty:return base
    validate_runtime(base)
    with _LOCK:stack_bindings(base._load_model(),stack)
    print(f'[ComfyUI-YuE2] Adapter stack: legacy | entries={len(stack.entries)} | identity={hash(stack.identity)}')
    return LegacyLoraPipe(base,adapter if isinstance(adapter,Yue2Adapter) else None,stack)

@contextmanager
def installed(model,adapter):
    stack=adapter if isinstance(adapter,AdapterStack) else stack_from_legacy(adapter)
    modules=stack_bindings(model,stack);handles=[];attached=[]
    try:
        for module,hook,target in modules:
            hook.to(device=module.weight.device)
            name=f'_yue2_lora_delta_{len(attached)}'
            module.add_module(name,hook);attached.append((module,name))
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:handle.remove()
        for module,name in attached:delattr(module,name)

@contextmanager
def pipeline_context(pipe):
    with _LOCK:
        if not isinstance(pipe,LegacyLoraPipe):yield pipe;return
        validate_runtime(pipe.base);model=pipe.base._load_model()
        with installed(model,pipe.stack):
            original=getattr(pipe.base,'effective_config',None);sentinel=object();previous=vars(pipe.base).get('effective_config',sentinel)
            if callable(original):
                def effective_config(*args,**kwargs):return {**original(*args,**kwargs),'yue2_adapter_stack':pipe.stack.identity,'yue2_lora':(pipe.adapter.identity if pipe.adapter is not None else pipe.stack.identity)}
                pipe.base.effective_config=effective_config
            try:yield pipe.base
            finally:
                if callable(original):
                    if previous is sentinel:del pipe.base.effective_config
                    else:pipe.base.effective_config=previous

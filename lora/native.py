"""ComfyUI native patcher application for canonical adapter stacks."""
from __future__ import annotations
import re
import torch
from .adapters import LoraPair, BranchAdapter, Yue2Adapter
from .canonical import (AdapterStack,AdapterStackEntry,CanonicalAdapterBundle,
                        CanonicalPatch,stack_from_legacy)

_LAYER=re.compile(r'^model\.layers\.(\d+)\.(nar_self_attn|self_attn|nar_mlp|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$')

def fuse(pairs,sizes,in_features):
    if len(pairs)!=len(sizes):raise ValueError('Fusion requires output widths per target')
    rank=sum(p.A.shape[0] for p in pairs if p is not None)
    if not rank:return None
    A=torch.zeros(rank,in_features);B=torch.zeros(sum(sizes),rank);row=col=0
    for pair,size in zip(pairs,sizes):
        if pair is not None:
            pair.validate((size,in_features));r=pair.A.shape[0]
            A[col:col+r]=pair.A;B[row:row+size,col:col+r]=pair.B;col+=r
        row+=size
    return LoraPair(A,B)

def _module_for(root,target,branch):
    if target.startswith('model.layers.'):
        native=target
        if branch=='nar':native=native.replace('.nar_self_attn.','.self_attn.').replace('.nar_mlp.','.mlp.')
        else:native=native.replace('.self_attn.','.self_attn.').replace('.mlp.','.mlp.')
        return root.get_submodule(native)
    return root.get_submodule(target)

def _pair_map(root,patches,branch,prefix=''):
    selected=[p for p in patches if p.branch==branch]
    targets=[p.target for p in selected if p.algorithm=='lora']
    if len(targets)!=len(set(targets)):raise ValueError('Duplicate/overlapping canonical native LoRA targets')
    result={};groups={};consumed=set()
    for p in selected:
        if p.algorithm!='lora':continue
        m=_LAYER.fullmatch(p.target)
        if m:
            layer,group,name=m.groups()
            groups.setdefault((layer,group),{})[name]=p
        else:
            mod=_module_for(root,p.target,branch)
            if tuple(mod.weight.shape)!=(p.up.shape[0],p.down.shape[1]):raise ValueError(f'Native adapter shape mismatch: {p.target}')
            result[prefix+p.target+'.weight']=LoraPair(p.down,p.up*p.intrinsic_scale)
            consumed.add(p.target)
    for (index,group),targets in groups.items():
        try: layer=root.get_submodule(f'model.layers.{index}')
        except (AttributeError,IndexError) as exc: raise ValueError(f'Native adapter target layer missing: model.layers.{index}') from exc
        nar=group.startswith('nar_'); native_group=group[4:] if nar else group
        if nar != (branch=='nar'):raise ValueError(f'Canonical branch/target mismatch: {group}')
        module=getattr(layer,native_group)
        if native_group=='self_attn' and hasattr(module,'qkv_proj'):
            names=('q_proj','k_proj','v_proj');sizes=(module.inner_size,module.kv_size,module.kv_size)
            pair_values=[]
            for name in names:
                p=targets.get(name);pair_values.append(LoraPair(p.down,p.up*p.intrinsic_scale) if p else None)
            pair=fuse(pair_values,sizes,module.qkv_proj.in_features)
            if pair:result[prefix+f'model.layers.{index}.self_attn.qkv_proj.weight']=pair
            consumed.update(p.target for p in targets.values() if p is not None and p.target.rsplit('.',1)[-1] in names)
            for name,p in targets.items():
                if name in names or p is None: continue
                mod=getattr(module,name); p2=LoraPair(p.down,p.up*p.intrinsic_scale); p2.validate(mod.weight.shape)
                result[prefix+f'model.layers.{index}.self_attn.{name}.weight']=p2
                consumed.add(p.target)
        elif native_group=='mlp' and hasattr(module,'gate_up_proj'):
            names=('gate_proj','up_proj');size=module.down_proj.in_features
            pair_values=[]
            for name in names:
                p=targets.get(name);pair_values.append(LoraPair(p.down,p.up*p.intrinsic_scale) if p else None)
            pair=fuse(pair_values,(size,size),module.gate_up_proj.in_features)
            if pair:result[prefix+f'model.layers.{index}.mlp.gate_up_proj.weight']=pair
            consumed.update(p.target for p in targets.values() if p is not None and p.target.rsplit('.',1)[-1] in names)
            for name,p in targets.items():
                if name in names or p is None: continue
                mod=getattr(module,name); p2=LoraPair(p.down,p.up*p.intrinsic_scale); p2.validate(mod.weight.shape)
                result[prefix+f'model.layers.{index}.mlp.{name}.weight']=p2
                consumed.add(p.target)
        else:
            for name,p in targets.items():
                if p is None:continue
                native=f'{native_group}.{name}'
                mod=getattr(module,name)
                p2=LoraPair(p.down,p.up*p.intrinsic_scale);p2.validate(mod.weight.shape)
                result[prefix+f'model.layers.{index}.{native}.weight']=p2
                consumed.add(p.target)
    expected={p.target for p in selected if p.algorithm=='lora'}
    if consumed!=expected:raise ValueError(f'Native backend dropped canonical LoRA patches: {sorted(expected-consumed)}')
    return result

def _dense_patch_map(root,patches,branch,prefix=''):
    """Map canonical delta/LoKr patches to native dense weight patches."""
    selected=[p for p in patches if p.branch==branch and p.algorithm!='lora']
    targets=[p.target for p in selected]
    if len(targets)!=len(set(targets)):raise ValueError('Duplicate/overlapping canonical native dense targets')
    result={}; grouped={}; consumed=set()
    for p in selected:
        match=_LAYER.fullmatch(p.target)
        if match:
            layer,group,name=match.groups()
            grouped.setdefault((layer,group),{})[name]=p
        else:
            mod=_module_for(root,p.target,branch)
            delta=p.dense_delta(mod.weight.shape)
            if delta is not None: result[prefix+p.target+'.weight']=(delta,)
            if p.bias_delta is not None: result[prefix+p.target+'.bias']=(p.bias_delta*p.intrinsic_scale,)
            consumed.add(p.target)
    for (index,group),targets in grouped.items():
        try: layer=root.get_submodule(f'model.layers.{index}')
        except (AttributeError,IndexError) as exc: raise ValueError(f'Native adapter target layer missing: model.layers.{index}') from exc
        nar=group.startswith('nar_'); native_group=group[4:] if nar else group
        if nar != (branch=='nar'): raise ValueError(f'Canonical branch/target mismatch: {group}')
        module=getattr(layer,native_group)
        if native_group=='self_attn' and hasattr(module,'qkv_proj'):
            names=('q_proj','k_proj','v_proj'); sizes=(module.inner_size,module.kv_size,module.kv_size)
            key=prefix+f'model.layers.{index}.self_attn.qkv_proj.weight'
            fused=torch.zeros_like(module.qkv_proj.weight,dtype=torch.float32); row=0
            for name,size in zip(names,sizes):
                patch=targets.get(name)
                if patch is not None:
                    delta=patch.dense_delta((size,module.qkv_proj.in_features))
                    fused[row:row+size].add_(delta); consumed.add(patch.target)
                row+=size
            direct={'o_proj'}
            unknown=targets.keys()-set(names)-direct
            if unknown: raise ValueError(f'Unsupported attention dense targets: {unknown}')
            if key in result: raise ValueError(f'Mixed LoRA and dense patches collide at {key}')
            if set(targets)&set(names): result[key]=(fused,)
            for name in direct:
                patch=targets.get(name)
                if patch is None: continue
                mod=getattr(module,name); delta=patch.dense_delta(mod.weight.shape)
                direct_key=prefix+f'model.layers.{index}.self_attn.{name}.weight'
                if delta is not None: result[direct_key]=(delta,)
                if patch.bias_delta is not None: result[prefix+f'model.layers.{index}.self_attn.{name}.bias']=(patch.bias_delta*patch.intrinsic_scale,)
                consumed.add(patch.target)
        elif native_group=='mlp' and hasattr(module,'gate_up_proj'):
            names=('gate_proj','up_proj'); size=module.down_proj.in_features
            key=prefix+f'model.layers.{index}.mlp.gate_up_proj.weight'
            fused=torch.zeros_like(module.gate_up_proj.weight,dtype=torch.float32); row=0
            for name in names:
                patch=targets.get(name)
                if patch is not None:
                    delta=patch.dense_delta((size,module.gate_up_proj.in_features))
                    fused[row:row+size].add_(delta); consumed.add(patch.target)
                row+=size
            direct={'down_proj'}
            unknown=targets.keys()-set(names)-direct
            if unknown: raise ValueError(f'Unsupported MLP dense targets: {unknown}')
            if key in result: raise ValueError(f'Mixed LoRA and dense patches collide at {key}')
            if set(targets)&set(names): result[key]=(fused,)
            for name in direct:
                patch=targets.get(name)
                if patch is None: continue
                mod=getattr(module,name); delta=patch.dense_delta(mod.weight.shape)
                direct_key=prefix+f'model.layers.{index}.mlp.{name}.weight'
                if delta is not None: result[direct_key]=(delta,)
                if patch.bias_delta is not None: result[prefix+f'model.layers.{index}.mlp.{name}.bias']=(patch.bias_delta*patch.intrinsic_scale,)
                consumed.add(patch.target)
        else:
            for name,patch in targets.items():
                mod=getattr(module,name); delta=patch.dense_delta(mod.weight.shape)
                if delta is not None: result[prefix+f'model.layers.{index}.{native_group}.{name}.weight']=(delta,)
                if patch.bias_delta is not None: result[prefix+f'model.layers.{index}.{native_group}.{name}.bias']=(patch.bias_delta*patch.intrinsic_scale,)
                consumed.add(patch.target)
    expected={p.target for p in selected}
    if consumed!=expected: raise ValueError(f'Native backend dropped canonical patches: {sorted(expected-consumed)}')
    return result

def _make_comfy_lora(patcher,root,patches,branch,prefix=''):
    import comfy.lora
    pairs=_pair_map(root,patches,branch,prefix)
    tensors={};mapping={}
    for key,pair in pairs.items():
        mapping[key]=key;tensors[key+'.lora_down.weight']=pair.A;tensors[key+'.lora_up.weight']=pair.B
        # Factors already contain canonical intrinsic scale; neutralize Comfy alpha/rank.
        tensors[key+'.alpha']=torch.tensor(float(pair.A.shape[0]))
    result=comfy.lora.load_lora(tensors,mapping,log_missing=False)
    if set(result)!=set(pairs):raise ValueError('ComfyUI rejected one or more canonical native LoRAs')
    dense=_dense_patch_map(root,patches,branch,prefix)
    overlap=set(result)&set(dense)
    if overlap: raise ValueError(f'Mixed LoRA and dense patches collide: {sorted(overlap)}')
    result.update(dense)
    return result

def _legacy_bundle(adapter):
    if isinstance(adapter,Yue2Adapter):return stack_from_legacy(adapter)
    return adapter

def apply_native(pipe,adapter,ar_strength=1.,nar_strength=1.,enabled=True,*,replace_existing=False):
    if not isinstance(pipe,dict) or pipe.get('kind')!='YUE2_NATIVE_PIPE':raise ValueError('Expected YUE2_NATIVE_PIPE')
    base=pipe.get('_yue2_adapter_base',pipe.get('_yue2_lora_base',pipe))
    incoming=_legacy_bundle(adapter)
    if isinstance(incoming,CanonicalAdapterBundle):
        incoming=AdapterStack((AdapterStackEntry(incoming,float(ar_strength),float(nar_strength),enabled=enabled),))
    if isinstance(incoming,AdapterStackEntry):incoming=AdapterStack((incoming,))
    if not isinstance(incoming,AdapterStack):raise TypeError(f'Unsupported canonical adapter {type(adapter).__name__}')
    old=(AdapterStack() if replace_existing else (pipe.get('_yue2_adapter_stack',AdapterStack()) if '_yue2_adapter_base' in pipe else AdapterStack()))
    if not incoming.entries:
        import comfy.model_prefetch
        comfy.model_prefetch.cleanup_prefetch_queues()
        return base
    stack=AdapterStack((*old.entries,*incoming.entries))
    if stack.is_empty:
        import comfy.model_prefetch
        comfy.model_prefetch.cleanup_prefetch_queues()
        return base
    model,clip=base['model'].clone(),base['clip'].clone()
    try:
        for entry in stack.entries:
            if not entry.enabled:continue
            for branch in ('ar','nar','shared'):
                strength=entry.strength_for(branch)
                if not strength:continue
                if branch=='shared':
                    if not any(p.branch=='shared' for p in entry.bundle.patches): continue
                    raise ValueError('Native shared-branch adapters are not implemented')
                patcher=clip.patcher if branch=='ar' else model
                root=patcher.model if branch=='ar' else patcher.model.get_submodule('diffusion_model')
                prefix='' if branch=='ar' else 'diffusion_model.'
                patches=_make_comfy_lora(patcher,root,entry.bundle.patches,branch,prefix)
                if not patches:continue
                if set(patcher.add_patches(patches,strength))!=set(patches):
                    raise ValueError(f'Adapter contains targets absent from native {branch.upper()} model')
        import comfy.model_prefetch
        comfy.model_prefetch.cleanup_prefetch_queues()
    except Exception:
        # Keep construction transactional: unpublished clones are released on
        # any failure, while the source pipe/base patchers remain untouched.
        for patcher in (getattr(clip,'patcher',None),model):
            if patcher is not None:
                try: patcher.detach()
                except Exception: pass
        try:
            import comfy.model_prefetch
            comfy.model_prefetch.cleanup_prefetch_queues()
        except Exception: pass
        raise
    print(f'[ComfyUI-YuE2] Adapter stack: native | entries={len(stack.entries)} | patches={len(stack.patches)}')
    return {**base,'model':model,'clip':clip,'_yue2_adapter_base':base,
            '_yue2_adapter_stack':stack,'_yue2_lora_base':base,'lora_identity':stack.identity}

# Backward-compatible helper consumed by tests and existing integrations.
def mapped_pairs(root,adapter,prefix=''):
    if isinstance(adapter,BranchAdapter):
        class Proxy:pass
        proxy=Proxy();proxy.branch=adapter.branch;proxy.metadata=adapter.metadata;proxy.fingerprint=adapter.fingerprint
        proxy.ar=proxy.nar=None
        if adapter.branch=='ar':proxy.ar=adapter
        else:proxy.nar=adapter
        canonical=stack_from_legacy(Yue2Adapter(proxy.ar,proxy.nar))
        patches=canonical.patches
        return _pair_map(root,patches,adapter.branch,prefix)
    return _pair_map(root,adapter.patches if isinstance(adapter,CanonicalAdapterBundle) else tuple(adapter),adapter.branch,prefix)

def make_patches(patcher,adapter,prefix=''):
    if isinstance(adapter,BranchAdapter):
        branch=adapter.branch;root=patcher.model.get_submodule(prefix.rstrip('.')) if prefix else patcher.model
        pairs=mapped_pairs(root,adapter,prefix)
        import comfy.lora
        tensors={};mapping={}
        for key,pair in pairs.items():mapping[key]=key;tensors[key+'.lora_down.weight']=pair.A;tensors[key+'.lora_up.weight']=pair.B;tensors[key+'.alpha']=torch.tensor(float(pair.A.shape[0]))
        return comfy.lora.load_lora(tensors,mapping,log_missing=False)
    branch=adapter.branch
    root=patcher.model.get_submodule(prefix.rstrip('.')) if prefix else patcher.model
    return _make_comfy_lora(patcher,root,adapter.patches,branch,prefix)

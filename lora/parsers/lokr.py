"""LyCORIS/HOT-Step LoKr parser. LoKr remains a distinct canonical algorithm."""
from __future__ import annotations
import re
import torch
from ..canonical import CanonicalPatch,CanonicalAdapterBundle
from ..detect import file_digest
from .common import load_with_meta,_canonical_target

_FACTOR=re.compile(r'^(.*?)[.]?(lokr_w1_a|lokr_w1_b|lokr_w2_a|lokr_w2_b|lokr_w1|lokr_w2)(?:[.]weight)?$')

def parse(path,meta,keys,sidecar,stack):
    meta,keys,values=load_with_meta(path)
    fmt=meta.get('format','')
    factors={}; alphas={}
    for key,value in values.items():
        if not torch.isfinite(value).all(): raise ValueError(f'Non-finite LoKr tensor: {key}')
        if key.endswith('.alpha'):
            alphas[key[:-6]]=float(value.item());continue
        m=_FACTOR.fullmatch(key)
        if not m: raise ValueError(f'Unsupported LoKr tensor key: {key}')
        prefix,name=m.groups()
        factors.setdefault(prefix,{})[name]=value.float()
    patches=[]
    forced='nar' if fmt=='yue2-nar-lokr-v1' else 'ar' if fmt=='yue2-ar-lokr-v1' else meta.get('branch')
    for prefix,f in factors.items():
        branch,target,fused=_canonical_target(prefix,meta,forced)
        if branch is None: raise ValueError(f'Cannot determine LoKr branch: {prefix}')
        if fused: raise ValueError(f'Fused LoKr needs an explicit factor row-slice layout; unsupported: {target}')
        # HOT-Step's native-split exporter documents monolithic LyCORIS w2
        # as already-scaled (scale=1); only factorized w2_a/w2_b uses alpha/dim.
        scale=1.
        alpha=alphas.get(prefix,meta.get('alpha'))
        if alpha is not None and not (meta.get('yue2_adapter_layout')=='native_split_v1' and 'lokr_w2' in f):
            dim=int(meta.get('rank',meta.get('lokr_dim',0)))
            if not dim: raise ValueError('LoKr alpha scaling requires rank or lokr_dim metadata')
            scale=float(alpha)/dim
        patch=CanonicalPatch(branch,target,'lokr',scale,w1=f.get('lokr_w1'),w2=f.get('lokr_w2'),
            w1a=f.get('lokr_w1_a'),w1b=f.get('lokr_w1_b'),
            w2a=f.get('lokr_w2_a'),w2b=f.get('lokr_w2_b')).validate()
        patches.append(patch)
    if not patches: raise ValueError('LoKr adapter contains no supported targets')
    if len({(p.branch,p.target) for p in patches})!=len(patches):raise ValueError('Duplicate LoKr targets')
    source=str(path.resolve())
    return CanonicalAdapterBundle((source,),(file_digest(source),),fmt,dict(meta),(),tuple(patches),
        {'branches':sorted({p.branch for p in patches}),'algorithm':'lokr'})

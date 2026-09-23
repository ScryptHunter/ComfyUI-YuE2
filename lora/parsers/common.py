"""Shared strict tensor to canonical YuE2 patch conversion helpers."""
from __future__ import annotations
import re
import torch
from safetensors import safe_open
from ..canonical import CanonicalPatch, CanonicalAdapterBundle, AdapterDependency
from ..detect import file_digest

_SUFFIXES=(('.lora_down.weight','down'),('.lora_up.weight','up'),
           ('.lora_A.weight','down'),('.lora_B.weight','up'),
           ('.lora_A.default.weight','down'),('.lora_B.default.weight','up'),
           ('.lora_A','down'),('.lora_B','up'))
_FUSED=re.compile(r'(?:(?:diffusion_model\.)?model\.)?layers\.(\d+)\.(self_attn|mlp)\.(qkv_proj|gate_up_proj)')
_SPLIT=re.compile(r'(?:(?:diffusion_model\.)?model\.)?layers\.(\d+)\.(?:(nar_)?)(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)')
_HOT_SPLIT=re.compile(r'^yue2\.blk\.(\d+)\.(nar_)?(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)$')


def tensors(path):
    with safe_open(str(path),framework='pt',device='cpu') as sf:
        # Detach canonical payloads from safetensors' mmap so files can safely
        # be replaced/removed on Windows after parsing.
        return {k:sf.get_tensor(k).to(dtype=torch.float32).clone() for k in sf.keys()}

def _factor_map(values):
    factors={}; alphas={}; deltas={}
    for key,value in values.items():
        if not torch.isfinite(value).all(): raise ValueError(f'Non-finite adapter tensor: {key}')
        if key.endswith('.alpha'):
            alphas[key[:-6]]=float(value.item()); continue
        suffix=next((x for x in _SUFFIXES if key.endswith(x[0])),None)
        if suffix is None:
            # PEFT can name adapters other than `default` (e.g. .production.).
            named=re.search(r'\.lora_([AB])\.[^.]+\.weight$',key)
            if named: suffix=(named.group(0),'down' if named.group(1)=='A' else 'up')
        if suffix:
            prefix=key[:-len(suffix[0])]
            factors.setdefault(prefix,{})[suffix[1]]=value
        elif key.endswith('.diff_b'):
            deltas.setdefault(key[:-7],{})['bias']=value
        elif key.endswith('.diff'):
            deltas.setdefault(key[:-5],{})['weight']=value
        elif 'lokr_' in key.lower():
            continue
        else:
            raise ValueError(f'Unsupported adapter tensor key: {key}')
    return factors,alphas,deltas

def _branch_for(prefix,meta,force=None):
    lower=prefix.lower()
    if lower.startswith('yue2.blk.'):
        return 'nar' if re.search(r'\.nar_(?:attn|ffn)_',lower) else 'ar'
    if lower.startswith(('nar.','nar_','nar_experts.','nar_expert.','model.nar.')): return 'nar'
    if lower.startswith(('text_encoder.','text_encoders.','clip.','ar.','ar_experts.','ar_expert.','model.ar.')): return 'ar'
    if lower.startswith(('ar.','ar_experts.','ar_expert.','model.ar.')): return 'ar'
    if lower.startswith(('diffusion_model.','model.diffusion_model.')): return 'nar'
    if force in ('ar','nar'): return force
    if 'branch' in meta and meta['branch'] in ('ar','nar'): return meta['branch']
    return None

def _strip_namespace(prefix):
    for start in ('base_model.model.','model.base_model.model.','ar_experts.','nar_experts.',
                  'ar_expert.','nar_expert.','model.ar.','model.nar.','text_encoders.','text_encoder.','clip.','diffusion_model.',
                  'model.diffusion_model.','ar.','nar.'):
        if prefix.startswith(start): return prefix[len(start):]
    return prefix

def _canonical_target(prefix,meta,force=None):
    branch=_branch_for(prefix,meta,force)
    target=_strip_namespace(prefix).replace('.default.','.').replace('_', '_')
    hot=_HOT_SPLIT.fullmatch(prefix)
    if hot:
        layer,nar,name=hot.groups(); branch=branch or ('nar' if nar else 'ar')
        names={'attn_q':'self_attn.q_proj','attn_k':'self_attn.k_proj','attn_v':'self_attn.v_proj',
               'attn_output':'self_attn.o_proj','ffn_gate':'mlp.gate_proj','ffn_up':'mlp.up_proj','ffn_down':'mlp.down_proj',
               'nar_attn_q':'nar_self_attn.q_proj','nar_attn_k':'nar_self_attn.k_proj','nar_attn_v':'nar_self_attn.v_proj',
               'nar_attn_output':'nar_self_attn.o_proj','nar_ffn_gate':'nar_mlp.gate_proj','nar_ffn_up':'nar_mlp.up_proj','nar_ffn_down':'nar_mlp.down_proj'}
        token=('nar_' if nar else '')+name
        return branch,f'model.layers.{layer}.{names[token]}',False
    match=_SPLIT.search(target)
    if match:
        layer,nar,group,module=match.groups()
        branch=branch or ('nar' if nar else 'ar')
        if branch=='nar': module_group='nar_self_attn' if group=='self_attn' else 'nar_mlp'
        else: module_group=group
        return branch,f'model.layers.{layer}.{module_group}.{module}',False
    fused=_FUSED.search(target)
    if fused:
        layer,group,module=fused.groups()
        branch=branch or ('nar' if meta.get('branch')=='nar' else 'ar')
        return branch,f'model.layers.{layer}.{group}.{module}',True
    # Top-level linear modules occur in Starnodes' acoustic preset.
    for name in ('llm2vae','vae2llm','time_embedder.mlp.0','time_embedder.mlp.2'):
        if target.endswith(name): return branch or 'nar',name,False
    raise ValueError(f'Unrecognized YuE2 LoRA target: {prefix}')

def _rank_scale(prefix,down,meta,alphas,fmt):
    alpha=alphas.get(prefix,meta.get('alpha'))
    if fmt=='fl-yue2-lora-v1': return 1.0
    if alpha is None: return 1.0
    alpha=float(alpha)
    rank=int(meta.get('rank',down.shape[0]))
    if rank < 1: raise ValueError(f'Invalid rank metadata for {prefix}')
    return alpha/rank

def _split_fused(target,branch,down,up,scale,meta,fmt=''):
    """Undo exact fused factor layout used by native YuE2 exporters."""
    match=re.fullmatch(r'model\.layers\.(\d+)\.(self_attn|mlp)\.(qkv_proj|gate_up_proj)',target)
    if not match: raise ValueError(f'Invalid fused YuE2 target {target}')
    layer,group,kind=match.groups(); count=3 if kind=='qkv_proj' else 2
    if up.shape[1] != down.shape[0]: raise ValueError(f'Fused LoRA rank mismatch for {target}')
    if kind=='qkv_proj':
        inp=down.shape[1]
        if inp % 2: raise ValueError(f'Invalid YuE2 QKV input width for {target}')
        sizes=(inp,inp//2,inp//2); names=('q_proj','k_proj','v_proj')
        out_parts=[]; cursor=0
        for size in sizes: out_parts.append(up[cursor:cursor+size]); cursor+=size
        groups=None
    else:
        if up.shape[0] % 2: raise ValueError(f'Invalid YuE2 gate/up output width for {target}')
        half=up.shape[0]//2; sizes=(half,half); names=('gate_proj','up_proj')
        out_parts=(up[:half],up[half:]); groups=None
    if sum(sizes)!=up.shape[0]: raise ValueError(f'Fused output shape mismatch for {target}')
    # HOT-Step fused export stores one shared A for the whole site. Starnodes
    # native conversion instead expands rank and writes block-diagonal factors.
    declared=str(meta.get('yue2_fused_layout',meta.get('fused_lora_layout',''))).lower()
    if declared in ('shared_a','shared-a','shared'):
        blockdiag=False
    elif declared in ('block_diagonal','block-diagonal','starnodes_block_diagonal'):
        blockdiag=True
    else:
        blockdiag=fmt!='yue2-aitk-fused-lora-v1' and _looks_block_diagonal(up,sizes,down.shape[0])
    if blockdiag:
        if down.shape[0] % count: raise ValueError(f'Invalid block-diagonal fused rank for {target}')
        rank=down.shape[0]//count; groups=[(r*rank,(r+1)*rank) for r in range(count)]
    else:
        rank=down.shape[0]; groups=[(0,rank)]*count
    patches=[]
    for name,rows,(lo,hi) in zip(names,out_parts,groups):
        canonical_group=('nar_self_attn' if branch=='nar' else 'self_attn') if group=='self_attn' else ('nar_mlp' if branch=='nar' else 'mlp')
        patches.append(CanonicalPatch(branch,f'model.layers.{layer}.{canonical_group}.{name}',
            'lora',scale,down[lo:hi],rows[:,lo:hi]).validate())
    return patches

def _looks_block_diagonal(up,sizes,rank):
    count=len(sizes)
    if rank % count:return False
    r=rank//count
    # Require exact zero off-block support; this matches Starnodes' converter
    # contract and avoids mistaking a dense shared-A B for split factors.
    row=0
    for i,size in enumerate(sizes):
        block=up[row:row+size]
        off=torch.cat((block[:,:i*r],block[:,(i+1)*r:]),dim=1)
        if off.numel() and torch.count_nonzero(off):return False
        row+=size
    return True

def _branch_has_tensors(branches):
    found={p.branch for p in branches}
    return found

def make_bundle(path,fmt,meta,values,*,force_branch=None,dependencies=(),info=None,
                alpha_default=None,peft_config=None):
    factors,alphas,deltas=_factor_map(values)
    metadata=dict(meta)
    if peft_config: metadata['_peft_config']=peft_config
    if alpha_default is not None and 'alpha' not in metadata: metadata['alpha']=str(alpha_default)
    patches=[]; seen=set(); branch_seen=set()
    for prefix,parts in factors.items():
        if set(parts)!={'down','up'}: raise ValueError(f'Missing A/B pair for {prefix}')
        down,up=parts['down'],parts['up']
        if down.ndim!=2 or up.ndim!=2 or down.shape[0]!=up.shape[1]: raise ValueError(f'Invalid LoRA dimensions for {prefix}')
        branch,target,fused=_canonical_target(prefix,metadata,force_branch)
        if branch is None: raise ValueError(f'Cannot determine AR/NAR branch for {prefix}')
        target_key=(branch,target)
        if target_key in seen: raise ValueError(f'Duplicate canonical adapter target: {branch}:{target}')
        seen.add(target_key); branch_seen.add(branch)
        scale=_rank_scale(prefix,down,metadata,alphas,fmt)
        if not torch.isfinite(torch.tensor(scale)): raise ValueError(f'Non-finite scaling for {prefix}')
        patches.extend(_split_fused(target,branch,down,up,scale,metadata,fmt) if fused else
                       [CanonicalPatch(branch,target,'lora',scale,down,up).validate()])
    for prefix,parts in deltas.items():
        branch,target,_=_canonical_target(prefix,metadata,force_branch)
        if branch is None: branch='nar'
        patches.append(CanonicalPatch(branch,target,'delta',1.,weight_delta=parts.get('weight'),bias_delta=parts.get('bias')).validate())
        branch_seen.add(branch)
    if not patches: raise ValueError(f'{fmt} adapter contains no recognized YuE2 patches')
    # Detect conflicting spelling aliases after normalization, including deltas.
    keys=[(p.branch,p.target) for p in patches]
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate/overlapping canonical targets within one adapter')
    source=str(path.resolve())
    from ..detect import file_digest
    return CanonicalAdapterBundle((source,), (file_digest(source),), fmt, metadata,
        tuple(dependencies),tuple(patches),{'branches':sorted(branch_seen),**(info or {})})


def load_with_meta(path):
    from ..detect import inspect_file
    meta,keys=inspect_file(path)
    values=tensors(path)
    return meta,keys,values

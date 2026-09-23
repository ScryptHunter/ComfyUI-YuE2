"""Strict parser for Mothersuperior's raw and ComfyUI YuE2 AR exports."""
import re
from .common import load_with_meta,make_bundle

def parse(path,meta,keys,sidecar,stack):
    meta,keys,values=load_with_meta(path)
    if str(meta.get('format','')).lower()!='pt':
        raise ValueError('Mothersuperior adapter must declare format=pt')
    is_raw='lora_scale' in meta or any(re.fullmatch(r'layers\.\d+\..*\.lora_[AB]',key) for key in keys)
    if is_raw:
        if 'yue2' not in str(meta.get('base_model','')).lower():
            raise ValueError('Mothersuperior raw adapter base_model must identify YuE2')
        if str(meta.get('intended_cot','')).lower()!='full' or 'ar branch' not in str(meta.get('delta','')).lower():
            raise ValueError('Mothersuperior raw adapter must identify the YuE2 AR/full-CoT contract')
        try:
            rank=int(meta['rank']); scale=float(meta['lora_scale'])
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError('Mothersuperior raw adapter requires numeric rank and lora_scale metadata') from exc
        if rank<1 or scale!=1.0:
            raise ValueError('Mothersuperior raw adapter requires positive rank and lora_scale=1.0')
        if 'alpha' in meta or 'lora_alpha' in meta or any(key.endswith('.alpha') for key in keys):
            raise ValueError('Mothersuperior raw export uses lora_scale, not alpha/rank scaling')
        target_re=re.compile(r'^layers\.(\d+)\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.lora_[AB]$')
        layers={int(m.group(1)) for key in keys if (m:=target_re.fullmatch(key))}
        expected={f'layers.{layer}.{module}.lora_{factor}' for layer in layers
            for module in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
                           'mlp.gate_proj','mlp.up_proj','mlp.down_proj') for factor in ('A','B')}
        if not layers or set(keys)!=expected:
            raise ValueError('Mothersuperior raw adapter must contain complete split q/k/v/o and gate/up/down A/B targets per layer')
        if any(values[key].shape[0]!=rank for key in keys if key.endswith('.lora_A')):
            raise ValueError('Mothersuperior raw A tensor rank disagrees with rank metadata')
        normalized={'model.'+key:value for key,value in values.items()}
        bundle=make_bundle(path,'mothersuperior-yue2-lora-v1',meta,normalized,
            force_branch='ar',info={'family':'Mothersuperior YuE2 instrumental AR','layout':'raw split','intended_cot':'full'})
        if bundle.branches!={'ar'}:raise ValueError('Mothersuperior raw adapter must contain AR patches only')
        return bundle

    branch=str(meta.get('yue2_lora_branch','')).lower()
    if branch not in ('ar','nar'):
        raise ValueError('Mothersuperior adapter requires yue2_lora_branch=ar or nar')
    layout=str(meta.get('layout','')).lower()
    if not ('qkv_proj' in layout and 'gate_up_proj' in layout and ('block' in layout or '3r' in layout)):
        raise ValueError('Mothersuperior adapter requires an explicit YuE2 fused/block-diagonal layout declaration')
    if not any('yue2' in str(meta.get(field,'')).lower() for field in ('base_model','base')):
        raise ValueError('Mothersuperior adapter base_model must identify YuE2')
    if 'scale' in meta and float(str(meta['scale']).split()[0])!=1.:
        raise ValueError('Mothersuperior no-alpha export requires scale=1.0')
    for key in values:
        # `format=pt` alone is not sufficient; require model-specific targets.
        if not key.startswith('text_encoders.model.layers.'):
            raise ValueError(f'Unsupported Mothersuperior target namespace: {key}')
    # This community ComfyUI export stores block-diagonal fused factors. Force
    # that layout instead of relying on a sparsity heuristic; no alpha/rank
    # scale is applied when the artifact declares scale=1.0 (no alpha).
    adjusted={**meta,'fused_lora_layout':'block_diagonal'}
    bundle=make_bundle(path,'mothersuperior-yue2-lora-v1',adjusted,values,
        force_branch=branch,info={'family':'Mothersuperior YuE2 ComfyUI AR','layout':'block-diagonal'})
    if bundle.branches!={branch}:
        raise ValueError('Mothersuperior adapter tensors disagree with yue2_lora_branch')
    if 'alpha' in meta or any(key.endswith('.alpha') for key in keys):
        raise ValueError('Mothersuperior ComfyUI export declares scale=1.0 (no alpha); alpha metadata is unsupported')
    return bundle

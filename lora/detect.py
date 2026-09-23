"""Safetensors inspection and ambiguity-safe YuE2 adapter parser registry."""
from __future__ import annotations
from pathlib import Path
from safetensors import safe_open
import hashlib,json
import re

_REGISTRY={}
def file_digest(path):
    with Path(path).resolve().open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def inspect_file(path):
    with safe_open(str(path),framework='pt',device='cpu') as stream:
        return dict(stream.metadata() or {}),tuple(stream.keys())
def _sidecar(path,name):
    target=Path(path).resolve().parent/name
    if not target.is_file():return None
    try:return json.loads(target.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError) as exc:raise ValueError(f'Invalid {name} sidecar for {path}: {exc}') from exc
def _peft_config(path):return _sidecar(path,'adapter_config.json')

def _is_mothersuperior_yue2(meta,keys):
    """Require an explicit YuE2 branch/layout and recognizable model targets."""
    if str(meta.get('format','')).lower()!='pt':return False
    if str(meta.get('yue2_lora_branch','')).lower() not in ('ar','nar'):return False
    layout=str(meta.get('layout','')).lower()
    if not all(token in layout for token in ('qkv_proj','gate_up_proj')):return False
    if 'block' not in layout and '3r' not in layout:return False
    target_re=re.compile(r'^text_encoders\.model\.layers\.\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.(?:gate_up_proj|down_proj))\.lora_(?:down|up)\.weight$')
    target_keys=[key for key in keys if key.endswith(('.lora_down.weight','.lora_up.weight'))]
    return bool(target_keys) and len(target_keys)==len(keys) and all(target_re.fullmatch(key) for key in target_keys)

def _is_mothersuperior_raw(meta,keys):
    """Strict signature for the community split/PEFT-like raw `format=pt` files."""
    if str(meta.get('format','')).lower()!='pt':return False
    if 'yue2' not in str(meta.get('base_model','')).lower():return False
    if str(meta.get('intended_cot','')).lower()!='full':return False
    if 'ar branch' not in str(meta.get('delta','')).lower():return False
    if 'lora_b @ lora_a' not in str(meta.get('delta','')).lower():return False
    try:
        if float(meta.get('lora_scale','nan'))!=1. or int(meta.get('rank','0'))<1:return False
    except (TypeError,ValueError):return False
    target_re=re.compile(r'^layers\.(\d+)\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.lora_[AB]$')
    if not keys or any(not target_re.fullmatch(key) for key in keys):return False
    layers={int(m.group(1)) for key in keys if (m:=target_re.fullmatch(key))}
    expected={f'layers.{layer}.{module}.lora_{factor}' for layer in layers
        for module in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
                       'mlp.gate_proj','mlp.up_proj','mlp.down_proj') for factor in ('A','B')}
    return set(keys)==expected

def register_parser(name,parser):
    if name in _REGISTRY and _REGISTRY[name] is not parser:raise ValueError(f'Duplicate YuE2 parser {name}')
    _REGISTRY[name]=parser
def registered_parsers():return tuple(_REGISTRY)

def detect_format(path):
    meta,keys=inspect_file(path)
    fmt=str(meta.get('format','')).strip().lower()
    layout=str(meta.get('yue2_adapter_layout','')).strip().lower()
    source=str(meta.get('source_format','')).strip().lower()
    config=_peft_config(path)
    is_lokr=any('lokr_' in key.lower() for key in keys)
    if is_lokr:
        candidate='hotstep_lokr' if fmt in ('yue2-ar-lokr-v1','yue2-nar-lokr-v1','yue2-aitk-lokr-v1') else None
    elif fmt=='fl-yue2-lora-v1':candidate='fl'
    elif fmt=='yue2-artist-ar-v1':candidate='artist_bundle'
    elif fmt=='comfyui-native-lora':candidate='comfy_native'
    elif _is_mothersuperior_yue2(meta,keys):candidate='mothersuperior'
    elif _is_mothersuperior_raw(meta,keys):candidate='mothersuperior'
    elif fmt=='yue2-aitk-fused-lora-v1':candidate='hotstep_fused'
    elif fmt in ('yue2-ar-lora-v1','yue2-nar-lora-v1') or layout=='native_split_v1':candidate='hotstep_native'
    elif fmt in ('yue2-studio-lora-v1',):candidate='starnodes'
    elif fmt=='yue2-lora-v1' and source=='yue2-lora-v1' and any(k.startswith('diffusion_model.') for k in keys):candidate='comfy_native'
    elif fmt=='yue2-lora-v1':candidate='starnodes' # same raw dialect used by Yue2 Studio
    elif config is not None:candidate='peft'
    elif any('.lora_A' in k or '.lora_B' in k for k in keys):
        # Metadata-less PEFT accepted only when key targets unmistakably map YuE2.
        target_keys=[k for k in keys if '.lora_A' in k or '.lora_B' in k]
        recognizable=all(any(token in k for token in ('model.layers.','llm2vae.','vae2llm.','time_embedder.')) for k in target_keys)
        candidate='peft' if recognizable else None
    else:candidate=None
    if candidate is None:
        raise ValueError(f'Unsupported or ambiguous YuE2 adapter: {path}; format={fmt!r}, layout={layout!r}, '
                         f'source_format={source!r}, PEFT config={config is not None}, sample keys={keys[:5]}')
    parser=_REGISTRY.get(candidate)
    if parser is None:raise ValueError(f'Detected {candidate}, but its parser is not registered')
    return candidate,meta,keys,config

def parse_adapter(path,*,_stack=()):
    path=Path(path).expanduser().resolve()
    if str(path) in _stack:raise ValueError(f'Adapter dependency cycle detected: {path}')
    name,meta,keys,config=detect_format(path)
    return _REGISTRY[name](path,meta,keys,config,(*_stack,str(path)))

# Registration occurs after all registry APIs exist; parser imports may call back into this module.
from . import parsers as _registered_parsers  # noqa: E402,F401

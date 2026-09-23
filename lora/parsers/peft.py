"""PEFT/HF safetensors adapter parser with YuE2-target allowlisting."""
from .common import load_with_meta,make_bundle
from ..canonical import AdapterDependency
from ..detect import file_digest
from pathlib import Path

def parse(path,meta,keys,sidecar,stack):
    if not isinstance(sidecar,dict): raise ValueError('PEFT adapter requires a valid adapter_config.json')
    model=str(sidecar.get('base_model_name_or_path','')).lower()
    if model and 'yue2' not in model and 'm-a-p' not in model:
        raise ValueError(f'PEFT base model is not clearly YuE2: {sidecar.get("base_model_name_or_path")!r}')
    if sidecar.get('fan_in_fan_out',False): raise ValueError('PEFT fan_in_fan_out is unsupported for YuE2 Linear targets')
    if sidecar.get('use_dora',False): raise ValueError('PEFT DoRA is not yet supported')
    rank=sidecar.get('r'); alpha=sidecar.get('lora_alpha',rank)
    if rank is None: raise ValueError('PEFT adapter_config.json must declare r')
    meta,keys,values=load_with_meta(path)
    cfg={**meta,'rank':str(rank),'alpha':str(alpha)}
    targets=sidecar.get('target_modules')
    if targets:
        targets={targets} if isinstance(targets,str) else set(targets)
        for key in keys:
            if '.lora_A' in key or '.lora_B' in key:
                if not any(key.split('.lora_')[0].endswith(t) for t in targets):
                    raise ValueError(f'PEFT tensor target is not declared in config: {key}')
    bundle=make_bundle(path,'peft-hf-lora',cfg,values,
        force_branch=meta.get('branch'),peft_config=sidecar,
        info={'family':'PEFT/HF','base_model':sidecar.get('base_model_name_or_path','')})
    config_path=Path(path).resolve().parent/'adapter_config.json'
    config_hash=file_digest(config_path)
    return type(bundle)((*bundle.source_files,str(config_path)),(*bundle.sha256s,config_hash),bundle.format_name,bundle.metadata,(AdapterDependency('peft-config',str(config_path),config_hash,True),*bundle.dependencies),bundle.patches,bundle.info)

"""ComfyUI-native fused YuE2 LoRA exporter parser."""
from .common import load_with_meta,make_bundle

def parse(path,meta,keys,sidecar,stack):
    meta,keys,values=load_with_meta(path)
    # Native trainer converter bakes source alpha/rank into B and writes
    # tensor alpha=fused_rank. Standard unbaked native files use alpha/rank.
    # Both consequently flow through the per-module alpha value here.
    branches={}
    for key in values:
        if key.endswith('.lora_down.weight') and key.startswith('diffusion_model.'):
            branches['nar']=True
        elif key.endswith('.lora_down.weight') and key.startswith('text_encoder.'):
            branches['ar']=True
    force=meta.get('branch')
    if force is None and len(branches)==1: force=next(iter(branches))
    return make_bundle(path,'comfyui-native-lora',meta,values,force_branch=force,
                       info={'family':'ComfyUI native fused'})

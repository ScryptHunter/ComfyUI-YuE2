"""HOT-Step / AI-Toolkit split and fused LoRA parser."""
from .common import load_with_meta,make_bundle

def parse_fused(path,meta,keys,sidecar,stack):
    meta,keys,values=load_with_meta(path)
    return make_bundle(path,'yue2-aitk-fused-lora-v1',meta,values,
                       info={'family':'HOT-Step/AI-Toolkit fused'})

def parse_native(path,meta,keys,sidecar,stack):
    meta,keys,values=load_with_meta(path)
    branch=meta.get('branch')
    if branch is None:
        branch='nar' if meta.get('format')=='yue2-nar-lora-v1' else 'ar'
    return make_bundle(path,meta.get('format','yue2-native-split-v1'),meta,values,
                       force_branch=branch,info={'family':'HOT-Step native split'})

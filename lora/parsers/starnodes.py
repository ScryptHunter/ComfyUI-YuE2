"""Starnodes raw trainer format; Yue2 Studio-compatible raw targets use same dialect."""
from .common import load_with_meta,make_bundle

def parse(path,meta,keys,sidecar,stack):
    meta,keys,values=load_with_meta(path)
    # Verified trainer implementation: default nar preset, metadata alpha/rank.
    branch=meta.get('branch','nar')
    if branch not in ('ar','nar'): raise ValueError(f'Invalid Starnodes branch: {branch!r}')
    return make_bundle(path,'yue2-lora-v1',meta,values,force_branch=branch,
        info={'family':'Starnodes raw / Studio-compatible'})

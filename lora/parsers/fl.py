"""FL-YuE2 parser with legacy-compatible strength semantics and paired NAR bundles."""
from pathlib import Path
from ..adapters import paired_nar
from ..canonical import CanonicalAdapterBundle,AdapterDependency
from ..detect import file_digest
from .common import load_with_meta,make_bundle

def parse(path,meta,keys,sidecar,stack):
    meta,keys,values=load_with_meta(path)
    if meta.get('format')!='fl-yue2-lora-v1': raise ValueError('Not an FL-YuE2 fl-yue2-lora-v1 file')
    branch=meta.get('branch')
    if branch not in ('ar','nar'): raise ValueError(f'Invalid FL branch {branch!r}')
    result=make_bundle(path,'fl-yue2-lora-v1',meta,values,force_branch=branch)
    if 'rank' in meta:
        expected=int(meta['rank'])
        for patch in result.patches:
            if patch.algorithm=='lora' and patch.rank!=expected:
                raise ValueError(f'FL rank metadata disagrees with {patch.target}')
    ref=meta.get('acoustic_adapter') if branch=='ar' else None
    if not ref:return result
    companion=paired_nar(path,(path.parent,path.parent.parent))
    if not companion:return result  # historical AR-only warning/fallback
    from ..detect import parse_adapter
    nar=parse_adapter(companion,_stack=stack)
    if nar.branches!={'nar'}: raise ValueError('FL acoustic_adapter must resolve only to an FL NAR adapter')
    digest=file_digest(companion)
    dep=AdapterDependency('nar-companion',str(Path(companion).resolve()),digest,True)
    return CanonicalAdapterBundle((*result.source_files,*nar.source_files),
        (*result.sha256s,*nar.sha256s),result.format_name,dict(meta),
        (*result.dependencies,dep,*nar.dependencies),(*result.patches,*nar.patches),
        {'branches':['ar','nar'],'paired_nar':str(companion)})

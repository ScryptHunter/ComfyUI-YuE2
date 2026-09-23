"""Yue2 Studio Artist contract validator.

Artist's NAR companion is a pinned decoder state (.pt), not an LoRA. Until the
runtime can transactionally swap/restore that state, reject the bundle rather
than misrepresent it as an independently-scalable NAR adapter.
"""
from pathlib import Path
import json
from ..detect import file_digest

def _safe_relative(root, name):
    if not isinstance(name, str) or not name: raise ValueError('Artist manifest requires companion.path')
    rel=Path(name.replace('\\','/'))
    if rel.is_absolute() or rel.drive or '..' in rel.parts: raise ValueError(f'Unsafe Artist companion path: {name}')
    path=(root/rel).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError(f'Artist companion file missing: {name}')
    return path

def parse(path,meta,keys,sidecar,stack):
    path=Path(path).resolve(); root=path.parent
    if meta.get('format')!='yue2-artist-ar-v1': raise ValueError('Artist adapter metadata format must be yue2-artist-ar-v1')
    try: scale=float(meta.get('scale','nan'))
    except (TypeError,ValueError): scale=float('nan')
    if scale!=1.: raise ValueError('Artist adapter metadata scale must be exactly 1')
    if not meta.get('encoder_revision'): raise ValueError('Artist adapter metadata requires encoder_revision')
    if not meta.get('companion_sha256'): raise ValueError('Artist adapter metadata requires companion_sha256')
    manifest_name=meta.get('manifest','manifest.json')
    manifest_path=_safe_relative(root,manifest_name)
    try: manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError) as exc: raise ValueError(f'Invalid Artist manifest: {exc}') from exc
    if not isinstance(manifest,dict) or manifest.get('format')!='yue2-artist-ar-v1':
        raise ValueError('Artist manifest format must be yue2-artist-ar-v1')
    if not isinstance(manifest.get('base'),(str,dict)) or not manifest.get('base'):
        raise ValueError('Artist manifest requires base identity')
    companion=manifest.get('companion')
    if not isinstance(companion,dict): raise ValueError('Artist manifest requires companion {path, sha256}')
    companion_path=_safe_relative(root,companion.get('path'))
    if companion_path.name!='nar_lora_joint_v4.pt':
        raise ValueError('Artist companion must be the upstream nar_lora_joint_v4.pt pinned decoder state')
    digest=file_digest(companion_path)
    expected=str(companion.get('sha256','')).lower()
    if not expected or digest.lower()!=expected: raise ValueError('Artist companion manifest SHA-256 mismatch')
    if str(meta['companion_sha256']).lower()!=digest.lower(): raise ValueError('Artist adapter companion_sha256 mismatch')
    # Upstream applies this state as a pinned NAR decoder companion at full
    # strength whenever Artist strength is nonzero; it is not a LoRA delta.
    raise ValueError('Yue2 Studio Artist bundles are unsupported by YuE2 Universal Loader: upstream companion is a pinned NAR decoder state, not a LoRA patch')

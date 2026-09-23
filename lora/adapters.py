"""Strict FL reader and backend-independent adapter representation."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import math
import re
import warnings
import torch
from safetensors import safe_open

FORMAT = "fl-yue2-lora-v1"

@dataclass(frozen=True)
class LoraPair:
    A: torch.Tensor
    B: torch.Tensor

    def validate(self, shape=None):
        if (self.A.ndim != 2 or self.B.ndim != 2 or
                min(*self.A.shape, *self.B.shape) < 1 or self.A.shape[0] != self.B.shape[1]):
            raise ValueError("Invalid LoRA A/B dimensions")
        actual = (self.B.shape[0], self.A.shape[1])
        if shape is not None and tuple(shape) != actual:
            raise ValueError(f"LoRA shape mismatch: expected {tuple(shape)}, got {actual}")

@dataclass(frozen=True)
class BranchAdapter:
    branch: str
    layers: dict[int, dict[str, LoraPair]]
    metadata: dict[str, str]
    path: str
    fingerprint: tuple
    projection_deltas: dict[str, torch.Tensor] = field(default_factory=dict)

@dataclass(frozen=True)
class Yue2Adapter:
    ar: BranchAdapter | None = None
    nar: BranchAdapter | None = None
    ar_strength: float = 1.0
    nar_strength: float = 1.0

    def __post_init__(self):
        for value in (self.ar_strength, self.nar_strength):
            if not math.isfinite(value) or not 0 <= value <= 3:
                raise ValueError("LoRA strengths must be finite and between 0 and 3")

    @property
    def identity(self):
        return tuple((a.fingerprint if a else None, s) for a, s in self.branches())

    def branches(self):
        return ((self.ar, self.ar_strength), (self.nar, self.nar_strength))

def fingerprint(path):
    """Detect same-size replacements even when mtime is preserved."""
    path = Path(path).resolve()
    stat = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return str(path), stat.st_mtime_ns, stat.st_size, digest

def metadata(path, branch=None):
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        meta = stream.metadata() or {}
    if meta.get("format") != FORMAT:
        raise ValueError(f"Unsupported YuE2 LoRA format. Detected metadata: {meta}. "
                         f"Currently supported: FL-YuE2 {FORMAT}.")
    if meta.get("branch") not in ("ar", "nar") or (branch and meta["branch"] != branch):
        raise ValueError(f"YuE2 LoRA branch error: expected {branch or 'ar/nar'}, "
                         f"got {meta.get('branch')!r} in {path}")
    return meta

def read_adapter(path, branch):
    meta = metadata(path, branch)  # Detect before materializing any weights.
    before = fingerprint(path)
    prefix = "nar_" if branch == "nar" else ""
    pattern = re.compile(r"model\.layers\.(\d+)\." + prefix +
                         r"(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj\.lora_(down|up)\.weight")
    pairs, deltas = {}, {}
    projection_keys = {f"{name}.{suffix}": f"{name}.{kind}"
                       for name in ("vae2llm", "llm2vae")
                       for suffix, kind in (("diff", "weight"), ("diff_b", "bias"))}
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        for key in stream.keys():
            match = pattern.fullmatch(key)
            if not match and not (branch == "nar" and key in projection_keys):
                raise ValueError(f"Unsupported YuE2 LoRA tensor: {key}")
            # safetensors returns CPU tensors backed by a mapped file region.
            # Copy before leaving the context: keeping the mapping alive can
            # prevent checkpoint replacement on Windows and couple canonical
            # patches to the lifetime of the source file handle.
            value = stream.get_tensor(key).clone()
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"LoRA tensor must contain finite floating-point values: {key}")
            value = value.float()
            if match:
                layer, group, target, factor = match.groups()
                if (group == "self_attn") != (target in ("q", "k", "v", "o")):
                    raise ValueError(f"Unsupported YuE2 LoRA target: {key}")
                pairs.setdefault((int(layer), target), {})[factor] = value
            else:
                deltas[projection_keys[key]] = value
    layers = {}
    for (layer, target), factors in pairs.items():
        if set(factors) != {"down", "up"}:
            raise ValueError(f"Incomplete LoRA A/B pair: layer {layer}, {target}")
        pair = LoraPair(factors["down"], factors["up"])
        pair.validate()
        if "rank" in meta and int(meta["rank"]) != pair.A.shape[0]:
            raise ValueError(f"LoRA rank metadata disagrees with layer {layer}, {target}")
        layers.setdefault(layer, {})[target] = pair
    if not layers and not deltas:
        raise ValueError("YuE2 adapter has no applicable tensors")
    if fingerprint(path) != before:
        raise ValueError("LoRA changed while loading; retry with a completed checkpoint")
    return BranchAdapter(branch, layers, meta, str(Path(path).resolve()), before, deltas)

def paired_nar(ar_path, search_roots=()):
    """Resolve metadata, including FL exports relative to the YuE2 LoRA root.

    Never guess a '-nar' filename or allow metadata to escape a search root.
    """
    ar_path = Path(ar_path).resolve()
    reference = metadata(ar_path, "ar").get("acoustic_adapter")
    if not reference:
        return None
    relative = Path(reference.replace("\\", "/"))
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError(f"Unsafe acoustic_adapter path: {reference!r}")
    roots = [ar_path.parent]
    if relative.parts and relative.parts[0].casefold() == ar_path.parent.name.casefold():
        roots.append(ar_path.parent.parent)
    roots.extend(Path(root).resolve() for root in search_roots)
    for root in roots:
        candidate = (root / relative).resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            metadata(candidate, "nar")
            return str(candidate)
    warnings.warn(f"Paired NAR adapter missing: {reference}; continuing with AR only", stacklevel=2)
    return None

def load_pair(ar_path=None, nar_override=None, auto=True, ar_strength=1.0,
              nar_strength=1.0, search_roots=()):
    nar_path = nar_override
    if not nar_path and ar_path and auto:
        nar_path = paired_nar(ar_path, search_roots)
    adapter = Yue2Adapter(read_adapter(ar_path, "ar") if ar_path else None,
                          read_adapter(nar_path, "nar") if nar_path else None,
                          float(ar_strength), float(nar_strength))
    for branch, strength in adapter.branches():
        if branch:
            ranks = sorted({p.A.shape[0] for layer in branch.layers.values() for p in layer.values()})
            print(f"[ComfyUI-YuE2] {FORMAT} {branch.branch.upper()}: {branch.path} | "
                  f"layers={len(branch.layers)}, ranks={ranks}, strength={strength:g}")
    return adapter

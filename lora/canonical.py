"""Backend-independent YuE2 patches and dependency-capable bundles."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import torch

Branch = Literal["ar", "nar", "shared"]

@dataclass(frozen=True)
class CanonicalPatch:
    branch: Branch
    target: str
    algorithm: Literal["lora", "lokr", "delta"]
    intrinsic_scale: float = 1.0
    down: torch.Tensor | None = None
    up: torch.Tensor | None = None
    weight_delta: torch.Tensor | None = None
    bias_delta: torch.Tensor | None = None
    w1: torch.Tensor | None = None
    w2: torch.Tensor | None = None
    w1a: torch.Tensor | None = None
    w1b: torch.Tensor | None = None
    w2a: torch.Tensor | None = None
    w2b: torch.Tensor | None = None

    def validate(self):
        if self.branch not in ("ar", "nar", "shared"):
            raise ValueError(f"Invalid adapter branch {self.branch!r}")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("Canonical adapter target cannot be empty")
        if self.algorithm == "lora":
            if self.down is None or self.up is None:
                raise ValueError(f"Missing LoRA down/up for {self.target}")
            if self.down.ndim != 2 or self.up.ndim != 2 or self.down.shape[0] != self.up.shape[1]:
                raise ValueError(f"Invalid LoRA dimensions for {self.target}")
        elif self.algorithm == "delta":
            if self.weight_delta is None and self.bias_delta is None:
                raise ValueError(f"Empty delta for {self.target}")
        elif self.algorithm == "lokr":
            if (self.w1 is None) != (self.w2 is None):
                raise ValueError(f"LoKr w1/w2 factors must be paired for {self.target}")
            if (self.w1a is None) != (self.w1b is None) or (self.w2a is None) != (self.w2b is None):
                raise ValueError(f"LoKr low-rank factors must be paired for {self.target}")
            if self.w1 is None and self.w1a is None:
                raise ValueError(f"LoKr factors are missing for {self.target}")
        else:
            raise ValueError(f"Unsupported adapter algorithm {self.algorithm!r}")
        if not torch.isfinite(torch.tensor(float(self.intrinsic_scale))):
            raise ValueError(f"Non-finite intrinsic scale for {self.target}")
        for name in ("down", "up", "weight_delta", "bias_delta", "w1", "w2", "w1a", "w1b", "w2a", "w2b"):
            value=getattr(self,name)
            if value is not None and (not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid/non-finite {name} tensor for {self.target}")
        return self

    @property
    def rank(self):
        return int(self.down.shape[0]) if self.down is not None else None

    def dense_delta(self, shape=None):
        """Materialize a 2D exact delta for parity tests and fused row slicing."""
        if self.algorithm == "lora":
            delta = self.up.float() @ self.down.float()
        elif self.algorithm == "delta":
            delta = self.weight_delta.float() if self.weight_delta is not None else None
        else:
            a = self.w1.float() if self.w1 is not None else self.w1a.float() @ self.w1b.float()
            b = self.w2.float() if self.w2 is not None else self.w2a.float() @ self.w2b.float()
            delta = torch.kron(a, b)
            if shape is not None and delta.numel() != int(torch.tensor(shape).prod()):
                raise ValueError(f"LoKr shape {tuple(delta.shape)} does not match {tuple(shape)} for {self.target}")
            if shape is not None:
                delta = delta.reshape(shape)
        if delta is not None and shape is not None and tuple(delta.shape) != tuple(shape):
            raise ValueError(f"Adapter delta shape {tuple(delta.shape)} does not match {tuple(shape)} for {self.target}")
        return None if delta is None else delta * float(self.intrinsic_scale)

@dataclass(frozen=True)
class AdapterDependency:
    role: str
    path: str
    sha256: str | None = None
    required: bool = True

@dataclass(frozen=True)
class CanonicalAdapterBundle:
    source_files: tuple[str, ...]
    sha256s: tuple[str, ...]
    format_name: str
    metadata: dict
    dependencies: tuple[AdapterDependency, ...]
    patches: tuple[CanonicalPatch, ...]
    info: dict = field(default_factory=dict)

    @property
    def branches(self):
        return frozenset(p.branch for p in self.patches)

    @property
    def algorithms(self):
        return frozenset(p.algorithm for p in self.patches)

@dataclass(frozen=True)
class AdapterStackEntry:
    bundle: CanonicalAdapterBundle
    ar_strength: float = 1.0
    nar_strength: float = 1.0
    shared_strength: float | None = None
    enabled: bool = True

    def __post_init__(self):
        for value in (self.ar_strength,self.nar_strength):
            if not torch.isfinite(torch.tensor(float(value))) or not 0 <= float(value) <= 3:
                raise ValueError("Adapter strengths must be finite and between 0 and 3")
        if self.shared_strength is not None and (not torch.isfinite(torch.tensor(float(self.shared_strength))) or not 0 <= float(self.shared_strength) <= 3):
            raise ValueError("Shared adapter strength must be finite and between 0 and 3")

    def strength_for(self, branch):
        if not self.enabled:
            return 0.0
        if branch == "ar":
            return float(self.ar_strength)
        if branch == "nar":
            return float(self.nar_strength)
        return float(self.shared_strength if self.shared_strength is not None else self.nar_strength)

@dataclass(frozen=True)
class AdapterStack:
    entries: tuple[AdapterStackEntry, ...] = ()

    @property
    def identity(self):
        return tuple((entry.bundle.source_files, entry.bundle.sha256s, entry.bundle.format_name,
                      entry.ar_strength, entry.nar_strength, entry.shared_strength, entry.enabled)
                     for entry in self.entries)

    def append(self, entry):
        if not isinstance(entry, AdapterStackEntry):
            raise TypeError("AdapterStack.append expects AdapterStackEntry")
        return AdapterStack((*self.entries, entry))

    @property
    def patches(self):
        return tuple(p for entry in self.entries if entry.enabled for p in entry.bundle.patches)

    @property
    def is_empty(self):
        for entry in self.entries:
            if not entry.enabled:continue
            if any(entry.strength_for(branch)!=0 for branch in entry.bundle.branches):return False
        return True

def stack_from_legacy(adapter):
    """Bridge the established FL API into the universal representation."""
    if not adapter.ar and not adapter.nar:
        return AdapterStack()
    patches, sources, hashes, metadata, dependencies = [], [], [], {}, []
    branches = []
    for branch, strength in adapter.branches():
        if branch is None:
            continue
        branches.append(branch.branch)
        sources.append(branch.path)
        hashes.append(branch.fingerprint[-1] if branch.fingerprint else "")
        metadata[branch.branch] = dict(branch.metadata)
        for index, targets in branch.layers.items():
            group = "nar_self_attn" if branch.branch == "nar" else "self_attn"
            mlp = "nar_mlp" if branch.branch == "nar" else "mlp"
            for name, pair in targets.items():
                module = group if name in ("q", "k", "v", "o") else mlp
                patches.append(CanonicalPatch(branch.branch,
                    f"model.layers.{index}.{module}.{name}_proj", "lora", 1., pair.A, pair.B).validate())
        for key, value in branch.projection_deltas.items():
            patches.append(CanonicalPatch(branch.branch, key.rsplit(".", 1)[0], "delta", 1.,
                weight_delta=value if key.endswith(".weight") else None,
                bias_delta=value if key.endswith(".bias") else None).validate())
    bundle = CanonicalAdapterBundle(tuple(sources), tuple(hashes), "fl-yue2-lora-v1", metadata,
                                   tuple(dependencies), tuple(patches), {"legacy_fl_api": True})
    return AdapterStack((AdapterStackEntry(bundle, adapter.ar_strength, adapter.nar_strength),))

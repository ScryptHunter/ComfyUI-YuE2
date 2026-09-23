"""FL-YuE2 loaders for the two intentionally distinct pipeline sockets."""
from pathlib import Path
import folder_paths
from ..lora.adapters import fingerprint, load_pair, paired_nar
from ..lora.legacy import apply_legacy
from ..lora.native import apply_native
from ..lora.detect import parse_adapter, inspect_file

def _files():
    files = [n.replace("\\", "/") for n in folder_paths.get_filename_list("loras")
             if n.lower().endswith(".safetensors")]
    return ["none", *sorted(files, key=lambda n: ("yue2" not in n.lower(), n.lower()))]

def _path(name):
    return None if not name or name == "none" else folder_paths.get_full_path_or_raise("loras", name)

def _roots():
    roots = []
    for root in folder_paths.get_folder_paths("loras"):
        roots.append(root)
        path = Path(root)
        if path.is_dir():
            roots.extend(str(p) for p in path.iterdir() if p.is_dir() and p.name.casefold() == "yue2")
    return roots

def _require_fl_pair(ar_name,nar_name,auto_load_paired_nar,backend):
    ar=_path(ar_name); nar=_path(nar_name)
    # Validate the explicitly selected AR file before companion discovery. The
    # companion resolver parses candidate adapters, so without this guard a
    # non-FL selection leaks a low-level parser error instead of useful node
    # migration guidance.
    def require_fl(selected):
        if not selected: return
        try: detected=str(inspect_file(selected)[0].get('format','unknown'))
        except Exception: detected='unknown'
        if detected!='fl-yue2-lora-v1':
            raise ValueError('This node supports FL-YuE2 fl-yue2-lora-v1 only. '
                f'Detected format: {detected}. Use "YuE2 Universal Adapter Loader (LoRA / LoKr) — {backend}".')
    require_fl(ar)
    require_fl(nar)
    if ar and not nar and auto_load_paired_nar:
        nar=paired_nar(ar,_roots())
    require_fl(nar)

def _algorithm_labels(bundle):
    names={'lora':'LoRA','lokr':'LoKr','delta':'Dense delta'}
    return ', '.join(names.get(value,value) for value in sorted(bundle.algorithms))

class YuE2LoraLoader:
    PIPE_TYPE = "YUE2_PIPE"
    RETURN_TYPES = ("YUE2_PIPE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "YuE2/Legacy HF Runtime"
    DESCRIPTION = ("This is the legacy FL-YuE2-specific loader. For Starnodes, PEFT, HOT-Step, LoKr and community adapters use YuE2 Universal Adapter Loader. "
                   "Accepts fl-yue2-lora-v1 only; replaces its earlier contribution.")

    @classmethod
    def INPUT_TYPES(cls):
        options = _files()
        return {"required": {
            "pipeline": (cls.PIPE_TYPE,), "ar_lora": (options, {"tooltip":"FL-YuE2 fl-yue2-lora-v1 AR adapter. This FL-only node does not load other YuE2 formats."}),
            "ar_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,"tooltip":"Strength applied only to the FL AR branch."}),
            "nar_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,"tooltip":"Strength applied only to the FL NAR branch."}),
            "auto_load_paired_nar": ("BOOLEAN", {"default": True,"tooltip":"Find the paired FL NAR adapter referenced by the AR file's acoustic_adapter metadata."}),
        }, "optional": {"nar_lora_override": (options, {"tooltip":"Optional FL-YuE2 fl-yue2-lora-v1 NAR adapter. Overrides automatic acoustic_adapter pairing."})}}

    @classmethod
    def IS_CHANGED(cls, ar_lora="none", ar_strength=1.0, nar_strength=1.0,
                   auto_load_paired_nar=True, nar_lora_override="none", **kwargs):
        ar, nar = _path(ar_lora), _path(nar_lora_override)
        if ar and not nar and auto_load_paired_nar:
            nar = paired_nar(ar, _roots())
        return (fingerprint(ar) if ar else None, fingerprint(nar) if nar else None,
                ar_strength, nar_strength, auto_load_paired_nar)

    def load(self, pipeline, ar_lora="none", ar_strength=1.0, nar_strength=1.0,
             auto_load_paired_nar=True, nar_lora_override="none"):
        _require_fl_pair(ar_lora,nar_lora_override,auto_load_paired_nar,'Legacy')
        adapter = load_pair(_path(ar_lora), _path(nar_lora_override),
                            auto_load_paired_nar, ar_strength, nar_strength, _roots())
        return (apply_legacy(pipeline, adapter, replace_existing=True),)

class YuE2NativeLoraLoader(YuE2LoraLoader):
    PIPE_TYPE = "YUE2_NATIVE_PIPE"
    RETURN_TYPES = ("YUE2_NATIVE_PIPE",)
    CATEGORY = "YuE2/Native ComfyUI"
    DESCRIPTION = ("This is the legacy FL-YuE2-specific loader. For Starnodes, PEFT, HOT-Step, LoKr and community adapters use YuE2 Universal Adapter Loader. "
                   "Accepts fl-yue2-lora-v1 only; Native ComfyUI patchers are cloned before applying patches.")

    def load(self, pipeline, ar_lora="none", ar_strength=1.0, nar_strength=1.0,
             auto_load_paired_nar=True, nar_lora_override="none"):
        _require_fl_pair(ar_lora,nar_lora_override,auto_load_paired_nar,'Native')
        adapter = load_pair(_path(ar_lora), _path(nar_lora_override),
                            auto_load_paired_nar, ar_strength, nar_strength, _roots())
        return (apply_native(pipeline, adapter, replace_existing=True),)


class YuE2UniversalAdapterLoader:
    """Append a parsed canonical adapter to a pipeline's adapter stack."""
    PIPE_TYPE = "YUE2_PIPE"
    RETURN_TYPES = (PIPE_TYPE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "YuE2/Adapters"
    DESCRIPTION = "YuE2 Universal Adapter Loader: automatically detects and applies supported YuE2 LoRA or LoKr formats. Chain Universal loaders to stack; changing this node replaces its own previous contribution."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipeline": (cls.PIPE_TYPE,), "adapter": (_files(), {"tooltip":"YuE2 LoRA/LoKr file. Format is detected automatically."}),
            "ar_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,"tooltip":"Strength applied to AR/semantic adapter patches. Has no effect for NAR-only adapters."}),
            "nar_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,"tooltip":"Strength applied to NAR/acoustic adapter patches. Has no effect for AR-only adapters."}),
            "enabled": ("BOOLEAN", {"default": True,"tooltip":"Disable to return the clean input pipeline without this loader's adapter contribution."})}}

    @classmethod
    def IS_CHANGED(cls, adapter="none", ar_strength=1.0, nar_strength=1.0, enabled=True, **kwargs):
        path = _path(adapter)
        if not path: return (None, ar_strength, nar_strength, enabled)
        bundle = parse_adapter(path)
        return (bundle.source_files, bundle.sha256s, tuple((d.path,d.sha256) for d in bundle.dependencies), ar_strength, nar_strength, enabled)

    def load(self, pipeline, adapter="none", ar_strength=1.0, nar_strength=1.0, enabled=True):
        path = _path(adapter)
        if not path: return (pipeline,)
        bundle = parse_adapter(path)
        print(f"[ComfyUI-YuE2] Universal Adapter\nFormat: {bundle.format_name}\nAlgorithm: {_algorithm_labels(bundle)}\nBranches: {', '.join(b.upper() for b in sorted(bundle.branches))}\nTargets: {len(bundle.patches)}\nStrength: AR={ar_strength} NAR={nar_strength}")
        if bundle.dependencies: print(f"[ComfyUI-YuE2] Adapter dependencies: {bundle.dependencies}")
        return (apply_legacy(pipeline, bundle, ar_strength, nar_strength, enabled),)

class YuE2NativeUniversalAdapterLoader(YuE2UniversalAdapterLoader):
    PIPE_TYPE = "YUE2_NATIVE_PIPE"
    RETURN_TYPES = (PIPE_TYPE,)
    CATEGORY = "YuE2/Native ComfyUI"
    def load(self, pipeline, adapter="none", ar_strength=1.0, nar_strength=1.0, enabled=True):
        path = _path(adapter)
        if not path: return (pipeline,)
        bundle = parse_adapter(path)
        print(f"[ComfyUI-YuE2] Universal Adapter\nFormat: {bundle.format_name}\nAlgorithm: {_algorithm_labels(bundle)}\nBranches: {', '.join(b.upper() for b in sorted(bundle.branches))}\nTargets: {len(bundle.patches)}\nStrength: AR={ar_strength} NAR={nar_strength}")
        return (apply_native(pipeline, bundle, ar_strength, nar_strength, enabled),)
NODE_CLASS_MAPPINGS = {"YuE2LoraLoader": YuE2LoraLoader, "YuE2NativeLoraLoader": YuE2NativeLoraLoader, "YuE2UniversalAdapterLoader": YuE2UniversalAdapterLoader, "YuE2NativeUniversalAdapterLoader": YuE2NativeUniversalAdapterLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"YuE2LoraLoader": "FL-YuE2 LoRA Pair Loader — Legacy",
                              "YuE2NativeLoraLoader": "FL-YuE2 LoRA Pair Loader — Native",
                              "YuE2UniversalAdapterLoader": "YuE2 Universal Adapter Loader (LoRA / LoKr) — Legacy",
                              "YuE2NativeUniversalAdapterLoader": "YuE2 Universal Adapter Loader (LoRA / LoKr) — Native"}

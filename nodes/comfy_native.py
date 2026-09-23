"""Native ComfyUI YuE2 pipeline and modular generation nodes.

``YUE2_NATIVE_PIPE`` is intentionally a small container around ComfyUI's own
MODEL / CLIP / VAE / AUDIO_ENCODER objects.  It is not the isolated
``yue2_infer`` runtime used by the legacy nodes and never needs a second copy
of the model weights.
"""
from __future__ import annotations

import folder_paths


PIPE_TYPE = "YUE2_NATIVE_PIPE"


def _preferred(options, marker: str):
    """Keep every selectable file, but put likely YuE2 files first."""
    return sorted(options, key=lambda value: (marker not in value.lower(), value.lower()))


def _loader_inputs():
    checkpoints = _preferred(folder_paths.get_filename_list("checkpoints"), "yue2")
    encoders = _preferred(folder_paths.get_filename_list("audio_encoders"), "sheetsage")
    if not checkpoints:
        checkpoints = ["Place yue2_3b_int8_convrot.safetensors in models/checkpoints"]
    encoder_options = ["none", *encoders]
    checkpoint_default = next(
        (name for name in checkpoints if name.lower() == "yue2_3b_int8_convrot.safetensors"),
        checkpoints[0],
    )
    encoder_default = next(
        (name for name in encoders if name.lower() == "sheetsage2_bf16.safetensors"),
        "none",
    )
    return {
        "required": {
            "checkpoint": (checkpoints, {
                "default": checkpoint_default,
                "tooltip": "Comfy-Org YuE2 checkpoint. Native INT8 and BF16 are supported.",
            }),
            "audio_encoder": (encoder_options, {
                "default": encoder_default,
                "tooltip": "Optional SheetSage2 model from models/audio_encoders.",
            }),
        },
        "optional": {
            "memory_budget_gib": ("INT", {
                "default": 0, "min": 0, "max": 192, "step": 1,
                "tooltip": "Connect memory_budget_gib from YuE2 GPU Memory Preset. Zero uses normal ComfyUI memory management.",
            }),
            "offload_ar": ("BOOLEAN", {
                "default": True,
                "tooltip": "Connect offload_ar from YuE2 GPU Memory Preset. Disable to force the native autoregressive text encoder fully into VRAM.",
            }),
        },
    }


def _load_native(checkpoint, audio_encoder="none", memory_budget_gib=0, offload_ar=True):
    """Load the same model objects used by ComfyUI's stock YuE2 nodes."""
    import comfy.sd

    checkpoint_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint)
    loaded = comfy.sd.load_checkpoint_guess_config(
        checkpoint_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    model, clip, vae = loaded[:3]
    if model is None or clip is None or vae is None:
        raise ValueError(
            f"{checkpoint!r} did not load as a complete native YuE2 checkpoint "
            "(MODEL, CLIP, and VAE are required)"
        )

    encoder = None
    encoder_path = None
    if audio_encoder != "none":
        import comfy.audio_encoders.audio_encoders
        import comfy.utils

        encoder_path = folder_paths.get_full_path_or_raise("audio_encoders", audio_encoder)
        state_dict = comfy.utils.load_torch_file(encoder_path, safe_load=True)
        encoder = comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd(state_dict)
        if encoder is None or not callable(getattr(encoder, "generate_abc", None)):
            raise ValueError(f"{audio_encoder!r} is not a native SheetSage2 audio encoder")

    pipe = {
        "kind": PIPE_TYPE,
        "version": 1,
        "model": model,
        "clip": clip,
        "vae": vae,
        "audio_encoder": encoder,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "audio_encoder_name": audio_encoder,
        "audio_encoder_path": encoder_path,
        "memory_budget_gib": int(memory_budget_gib or 0),
        "offload_ar": bool(offload_ar),
    }
    return pipe, model, clip, vae, encoder


def _require_pipe(pipe, *, audio_encoder=False):
    if not isinstance(pipe, dict) or pipe.get("kind") != PIPE_TYPE:
        raise ValueError("Expected a YUE2_NATIVE_PIPE from YuE2 Native Pipeline Loader")
    required = ("model", "clip", "vae")
    missing = [name for name in required if pipe.get(name) is None]
    if missing:
        raise ValueError(f"Native YuE2 pipe is missing: {', '.join(missing)}")
    if audio_encoder and pipe.get("audio_encoder") is None:
        raise ValueError(
            "This workflow needs SheetSage2. Select sheetsage2_bf16.safetensors "
            "in YuE2 Native Pipeline Loader."
        )
    return pipe


def _prepare_text_encoder(pipe):
    """Honor the shared GPU preset using ComfyUI's public model loader API."""
    pipe = _require_pipe(pipe)
    if pipe.get("offload_ar", True):
        return pipe["clip"]

    patcher = getattr(pipe["clip"], "patcher", None)
    if patcher is None:
        raise RuntimeError("Native YuE2 CLIP object does not expose a ComfyUI ModelPatcher")

    import comfy.model_management
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    return pipe["clip"]


class YuE2NativeModelsLoader:
    """Backward-compatible loader exposing only standard ComfyUI objects."""

    DESCRIPTION = (
        "Loads native YuE2 MODEL, CLIP, VAE, and optional AUDIO_ENCODER objects "
        "from the standard ComfyUI model folders. Use Native Pipeline Loader for "
        "the modular pipe workflow."
    )

    INPUT_TYPES = classmethod(lambda cls: _loader_inputs())
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "AUDIO_ENCODER")
    RETURN_NAMES = ("model", "clip", "vae", "audio_encoder")
    FUNCTION = "load"
    CATEGORY = "YuE2/Native ComfyUI"

    def load(self, checkpoint, audio_encoder="none", memory_budget_gib=0, offload_ar=True):
        _pipe, model, clip, vae, encoder = _load_native(
            checkpoint, audio_encoder, memory_budget_gib, offload_ar
        )
        return model, clip, vae, encoder


class YuE2NativePipelineLoader:
    """Load native Comfy objects and bundle them into a reusable light pipe."""

    DESCRIPTION = (
        "Loads YuE2 from models/checkpoints and SheetSage2 from "
        "models/audio_encoders. The pipe only wraps native ComfyUI objects; it "
        "does not use yue2_infer or duplicate weights. Standard outputs remain "
        "available for stock ComfyUI nodes."
    )

    INPUT_TYPES = classmethod(lambda cls: _loader_inputs())
    RETURN_TYPES = (PIPE_TYPE, "MODEL", "CLIP", "VAE", "AUDIO_ENCODER")
    RETURN_NAMES = ("pipe", "model", "clip", "vae", "audio_encoder")
    FUNCTION = "load"
    CATEGORY = "YuE2/Native ComfyUI"

    def load(self, checkpoint, audio_encoder="none", memory_budget_gib=0, offload_ar=True):
        return _load_native(checkpoint, audio_encoder, memory_budget_gib, offload_ar)


class YuE2NativePipeComponents:
    """Expose standard ComfyUI objects contained in a native YuE2 pipe."""

    DESCRIPTION = (
        "Unpacks a YUE2_NATIVE_PIPE into MODEL, CLIP, VAE, and AUDIO_ENCODER "
        "for interoperability with stock ComfyUI nodes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": (PIPE_TYPE,)}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "AUDIO_ENCODER", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "audio_encoder", "info")
    FUNCTION = "unpack"
    CATEGORY = "YuE2/Native ComfyUI"

    def unpack(self, pipe):
        pipe = _require_pipe(pipe)
        info = (
            f"checkpoint={pipe.get('checkpoint')} | "
            f"audio_encoder={pipe.get('audio_encoder_name')} | "
            f"AR_offload={pipe.get('offload_ar')} | native ComfyUI"
        )
        return pipe["model"], pipe["clip"], pipe["vae"], pipe.get("audio_encoder"), info


class YuE2NativeGenerateABC:
    """Pipe-input equivalent of ComfyUI's stock YuE2 Generate ABC node."""

    DESCRIPTION = (
        "Generates an ABC score with the native CLIP contained in the pipe. "
        "Sampling controls match ComfyUI's stock YuE2 Generate ABC node."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipe": (PIPE_TYPE,),
            "style": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "lyrics": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                              "control_after_generate": True}),
            "mode": (["full", "melody"],),
            "max_abc_tokens": ("INT", {"default": 8192, "min": 1, "max": 20000}),
            "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 5.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 30, "min": 1, "max": 32768}),
            "repetition_penalty": ("FLOAT", {"default": 1.005, "min": 0.01, "max": 10.0, "step": 0.005}),
            "penalty_window": ("INT", {"default": 100, "min": 1, "max": 20000}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("abc",)
    FUNCTION = "generate"
    CATEGORY = "YuE2/Native ComfyUI"

    def generate(self, pipe, style, lyrics, seed, mode, max_abc_tokens=8192,
                 temperature=0.7, top_p=0.9, top_k=30,
                 repetition_penalty=1.005, penalty_window=100):
        clip = _prepare_text_encoder(pipe)
        tokens = clip.tokenize(
            style,
            lyrics=lyrics,
            cot=mode,
            seed=int(seed),
            max_tokens=int(max_abc_tokens),
            penalty_window=int(penalty_window),
        )
        ids = clip.generate(
            tokens,
            max_length=int(max_abc_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            seed=int(seed),
        )
        return (clip.decode(ids),)


class YuE2NativeGenerateMusic:
    """Build native YuE2 conditioning and expose the model needed to sample it."""

    DESCRIPTION = (
        "Uses the native pipe to generate music conditioning from style, lyrics, "
        "and optional ABC. Connect MODEL/CONDITIONING/seconds/VAE to standard "
        "KSampler, Empty YuE2 Latent Audio, and VAE Decode Audio nodes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipe": (PIPE_TYPE,),
            "style": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "lyrics": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "abc": ("STRING", {"default": "", "multiline": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                              "control_after_generate": True}),
            "mode": (["full", "melody"],),
            "max_duration": ("FLOAT", {"default": 360.0, "min": 0.04, "max": 900.0, "step": 0.04}),
            "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 100, "min": 1, "max": 32768}),
            "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.01, "max": 10.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL", "VAE", "CONDITIONING", "FLOAT")
    RETURN_NAMES = ("model", "vae", "conditioning", "seconds")
    FUNCTION = "generate"
    CATEGORY = "YuE2/Native ComfyUI"

    def generate(self, pipe, style, lyrics, abc, seed, mode, max_duration=360.0,
                 temperature=1.0, top_p=0.95, top_k=100,
                 repetition_penalty=1.2):
        from comfy.text_encoders.yue2 import FRAMES_PER_SECOND

        pipe = _require_pipe(pipe)
        clip = _prepare_text_encoder(pipe)
        abc = abc or ""
        cot = mode if abc.strip() else "off"
        tokens = clip.tokenize(
            style,
            lyrics=lyrics,
            cot=cot,
            seed=int(seed),
            abc=abc,
            max_tokens=max(1, round(float(max_duration) * FRAMES_PER_SECOND)),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        seconds = conditioning[0][1]["yue2_frames"] / FRAMES_PER_SECOND
        return pipe["model"], pipe["vae"], conditioning, float(seconds)


class YuE2NativeAudioToABC:
    """Transcribe reference audio through SheetSage2 stored in the native pipe."""

    DESCRIPTION = (
        "Converts reference audio to ABC with the SheetSage2 encoder selected in "
        "the native pipeline loader. Output is compatible with all YuE2 ABC tools."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipe": (PIPE_TYPE,),
            "audio": ("AUDIO",),
            "mode": (["melody", "full"],),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("abc",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "transcribe"
    CATEGORY = "YuE2/Native ComfyUI"

    def transcribe(self, pipe, audio, mode):
        encoder = _require_pipe(pipe, audio_encoder=True)["audio_encoder"]
        abc = encoder.generate_abc(
            audio["waveform"],
            audio["sample_rate"],
            melody_only=mode == "melody",
        )
        return (abc if isinstance(abc, list) else [abc],)


NODE_CLASS_MAPPINGS = {
    "YuE2NativeModelsLoader": YuE2NativeModelsLoader,
    "YuE2NativePipelineLoader": YuE2NativePipelineLoader,
    "YuE2NativePipeComponents": YuE2NativePipeComponents,
    "YuE2NativeGenerateABC": YuE2NativeGenerateABC,
    "YuE2NativeGenerateMusic": YuE2NativeGenerateMusic,
    "YuE2NativeAudioToABC": YuE2NativeAudioToABC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2NativeModelsLoader": "YuE2 Native Models Loader (ComfyUI)",
    "YuE2NativePipelineLoader": "YuE2 Native Pipeline Loader (ComfyUI)",
    "YuE2NativePipeComponents": "YuE2 Native Pipe Components",
    "YuE2NativeGenerateABC": "YuE2 Native Generate ABC",
    "YuE2NativeGenerateMusic": "YuE2 Native Generate Music",
    "YuE2NativeAudioToABC": "YuE2 Native Audio to ABC",
}

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nodes" / "comfy_native.py"


def load_module(folder_paths):
    name = "_test_comfy_native"
    sys.modules.pop(name, None)
    with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
        spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


class NativeLoaderTests(unittest.TestCase):
    def setUp(self):
        self.folder_paths = types.SimpleNamespace(
            get_filename_list=lambda kind: {
                "checkpoints": ["other.safetensors", "yue2_3b_bf16.safetensors",
                                "yue2_3b_int8_convrot.safetensors"],
                "audio_encoders": ["other_encoder.safetensors", "sheetsage2_bf16.safetensors"],
            }[kind],
            get_full_path_or_raise=lambda kind, name: f"/{kind}/{name}",
            get_folder_paths=lambda kind: [f"/{kind}"],
        )
        self.module = load_module(self.folder_paths)

    def test_int8_and_sheetsage_are_the_defaults(self):
        required = self.module.YuE2NativeModelsLoader.INPUT_TYPES()["required"]
        self.assertEqual(required["checkpoint"][1]["default"],
                         "yue2_3b_int8_convrot.safetensors")
        self.assertEqual(required["audio_encoder"][1]["default"],
                         "sheetsage2_bf16.safetensors")

    def test_loads_standard_comfy_objects_without_encoder(self):
        model, clip, vae = object(), object(), object()
        load_checkpoint = mock.Mock(return_value=(model, clip, vae, {"ignored": True}))
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy_sd = types.ModuleType("comfy.sd")
        comfy_sd.load_checkpoint_guess_config = load_checkpoint
        comfy.sd = comfy_sd
        with mock.patch.dict(sys.modules, {"comfy": comfy, "comfy.sd": comfy_sd}):
            result = self.module.YuE2NativeModelsLoader().load(
                "yue2_3b_int8_convrot.safetensors", "none")
        self.assertEqual(result, (model, clip, vae, None))
        load_checkpoint.assert_called_once_with(
            "/checkpoints/yue2_3b_int8_convrot.safetensors",
            output_vae=True,
            output_clip=True,
            embedding_directory=["/embeddings"],
        )

    def test_loads_native_sheetsage_encoder(self):
        model, clip, vae = object(), object(), object()
        encoder = types.SimpleNamespace(generate_abc=lambda *args, **kwargs: ["X:1"])
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy_sd = types.ModuleType("comfy.sd")
        comfy_sd.load_checkpoint_guess_config = mock.Mock(return_value=(model, clip, vae, None))
        comfy_utils = types.ModuleType("comfy.utils")
        comfy_utils.load_torch_file = mock.Mock(return_value={"weights": object()})
        comfy_audio = types.ModuleType("comfy.audio_encoders")
        comfy_audio.__path__ = []
        comfy_audio_impl = types.ModuleType("comfy.audio_encoders.audio_encoders")
        comfy_audio_impl.load_audio_encoder_from_sd = mock.Mock(return_value=encoder)
        comfy.sd = comfy_sd
        comfy.utils = comfy_utils
        comfy.audio_encoders = comfy_audio
        comfy.audio_encoders.audio_encoders = comfy_audio_impl
        modules = {
            "comfy": comfy,
            "comfy.sd": comfy_sd,
            "comfy.utils": comfy_utils,
            "comfy.audio_encoders": comfy_audio,
            "comfy.audio_encoders.audio_encoders": comfy_audio_impl,
        }
        with mock.patch.dict(sys.modules, modules):
            result = self.module.YuE2NativeModelsLoader().load(
                "yue2_3b_bf16.safetensors", "sheetsage2_bf16.safetensors")
        self.assertIs(result[3], encoder)
        comfy_utils.load_torch_file.assert_called_once_with(
            "/audio_encoders/sheetsage2_bf16.safetensors", safe_load=True)

    def test_pipeline_loader_wraps_native_objects_and_keeps_standard_outputs(self):
        model, clip, vae = object(), object(), object()
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy_sd = types.ModuleType("comfy.sd")
        comfy_sd.load_checkpoint_guess_config = mock.Mock(
            return_value=(model, clip, vae, None))
        comfy.sd = comfy_sd
        with mock.patch.dict(sys.modules, {"comfy": comfy, "comfy.sd": comfy_sd}):
            pipe, output_model, output_clip, output_vae, encoder = (
                self.module.YuE2NativePipelineLoader().load(
                    "yue2_3b_int8_convrot.safetensors", "none")
            )
        self.assertEqual(pipe["kind"], "YUE2_NATIVE_PIPE")
        self.assertIs(pipe["model"], model)
        self.assertIs(pipe["clip"], clip)
        self.assertIs(pipe["vae"], vae)
        self.assertIs(output_model, model)
        self.assertIs(output_clip, clip)
        self.assertIs(output_vae, vae)
        self.assertIsNone(encoder)

    def test_pipe_abc_generation_uses_native_clip(self):
        clip = mock.Mock()
        clip.tokenize.return_value = {"tokens": True}
        clip.generate.return_value = [1, 2, 3]
        clip.decode.return_value = "X:1\nK:C\nC D E F|"
        pipe = {
            "kind": "YUE2_NATIVE_PIPE",
            "model": object(),
            "clip": clip,
            "vae": object(),
        }
        result = self.module.YuE2NativeGenerateABC().generate(
            pipe, "trip hop", "[verse]\nHello", 7, "melody")
        self.assertEqual(result, ("X:1\nK:C\nC D E F|",))
        clip.tokenize.assert_called_once_with(
            "trip hop", lyrics="[verse]\nHello", cot="melody", seed=7,
            max_tokens=8192, penalty_window=100)
        clip.generate.assert_called_once()

    def test_gpu_preset_can_force_native_text_encoder_full_load(self):
        patcher = object()
        clip = mock.Mock()
        clip.patcher = patcher
        clip.tokenize.return_value = {"tokens": True}
        clip.generate.return_value = [1]
        clip.decode.return_value = "X:1"
        pipe = {
            "kind": "YUE2_NATIVE_PIPE",
            "model": object(), "clip": clip, "vae": object(),
            "memory_budget_gib": 48, "offload_ar": False,
        }
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        management = types.ModuleType("comfy.model_management")
        management.load_models_gpu = mock.Mock()
        comfy.model_management = management
        with mock.patch.dict(sys.modules, {
            "comfy": comfy, "comfy.model_management": management,
        }):
            self.module.YuE2NativeGenerateABC().generate(
                pipe, "trip hop", "[verse]\nHello", 7, "melody"
            )
        management.load_models_gpu.assert_called_once_with(
            [patcher], force_full_load=True
        )
    def test_pipe_audio_transcription_uses_native_encoder(self):
        encoder = mock.Mock()
        encoder.generate_abc.return_value = ["X:1\nK:C\nC|"]
        pipe = {
            "kind": "YUE2_NATIVE_PIPE",
            "model": object(),
            "clip": object(),
            "vae": object(),
            "audio_encoder": encoder,
        }
        audio = {"waveform": object(), "sample_rate": 48000}
        result = self.module.YuE2NativeAudioToABC().transcribe(
            pipe, audio, "melody")
        self.assertEqual(result, (["X:1\nK:C\nC|"],))
        encoder.generate_abc.assert_called_once_with(
            audio["waveform"], 48000, melody_only=True)


if __name__ == "__main__":
    unittest.main()

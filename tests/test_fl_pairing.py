"""FL joint-checkpoint dependency and loader opt-out regressions."""
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
import types
from unittest import mock
from uuid import uuid4

import pytest
import torch
from safetensors.torch import save_file

from lora.detect import parse_adapter


ROOT = Path(__file__).resolve().parents[1]


def write_fl(path, branch, companion=None):
    prefix = "nar_" if branch == "nar" else ""
    key = f"model.layers.0.{prefix}self_attn.q_proj"
    metadata = {"format": "fl-yue2-lora-v1", "branch": branch}
    if companion is not None:
        metadata["acoustic_adapter"] = companion
    save_file({key + ".lora_down.weight": torch.ones((2, 4)),
               key + ".lora_up.weight": torch.ones((4, 2))}, str(path), metadata=metadata)
    return path


@contextmanager
def load_nodes(root):
    name = "_fl_pair_features_" + uuid4().hex
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT)]
    folder = types.ModuleType("folder_paths")
    folder.get_filename_list = lambda kind: [p.name for p in root.glob("*.safetensors")]
    folder.get_full_path_or_raise = lambda kind, filename: str(root / filename)
    folder.get_folder_paths = lambda kind: [str(root)]
    sys.modules[name] = package
    previous = sys.modules.get("folder_paths")
    sys.modules["folder_paths"] = folder
    try:
        yield importlib.import_module(name + ".nodes.yue2_lora")
    finally:
        for module_name in list(sys.modules):
            if module_name == name or module_name.startswith(name + "."):
                sys.modules.pop(module_name, None)
        if previous is None:
            sys.modules.pop("folder_paths", None)
        else:
            sys.modules["folder_paths"] = previous


def test_universal_parser_requires_metadata_companion_and_reports_pair(tmp_path, capsys):
    ar = write_fl(tmp_path / "ar.safetensors", "ar", "nar.safetensors")
    with pytest.raises(ValueError, match="required.*nar.safetensors"):
        parse_adapter(ar)
    write_fl(tmp_path / "nar.safetensors", "nar")
    bundle = parse_adapter(ar)
    assert bundle.branches == {"ar", "nar"}
    assert Path(bundle.info["paired_nar"]).name == "nar.safetensors"
    assert any(dep.required for dep in bundle.dependencies)
    with load_nodes(tmp_path) as nodes:
        with mock.patch.object(nodes, "apply_legacy", return_value="legacy"), mock.patch.object(
            nodes, "apply_native", return_value="native"
        ):
            assert nodes.YuE2UniversalAdapterLoader().load("base", "ar.safetensors") == ("legacy",)
            assert nodes.YuE2NativeUniversalAdapterLoader().load("base", "ar.safetensors") == ("native",)
    diagnostic = capsys.readouterr().out
    assert "Format: fl-yue2-lora-v1" in diagnostic
    assert "Branches: AR, NAR" in diagnostic
    assert "Paired NAR: nar.safetensors" in diagnostic


def test_universal_ar_without_companion_metadata_is_legal(tmp_path):
    ar = write_fl(tmp_path / "ar.safetensors", "ar")
    assert parse_adapter(ar).branches == {"ar"}


def test_fl_specific_auto_required_but_explicit_opt_out_and_override_work(tmp_path):
    write_fl(tmp_path / "ar.safetensors", "ar", "missing-nar.safetensors")
    with load_nodes(tmp_path) as nodes:
        for loader_name, applier_name in (("YuE2LoraLoader", "apply_legacy"),
                                          ("YuE2NativeLoraLoader", "apply_native")):
            loader = getattr(nodes, loader_name)()
            with pytest.raises(ValueError, match="missing-nar.safetensors"):
                loader.load("base", "ar.safetensors", auto_load_paired_nar=True)
            with mock.patch.object(nodes, applier_name, return_value="patched") as apply:
                assert loader.load("base", "ar.safetensors", auto_load_paired_nar=False) == ("patched",)
                assert apply.call_args.args[1].nar is None
            write_fl(tmp_path / "chosen.safetensors", "nar")
            with mock.patch.object(nodes, applier_name, return_value="patched") as apply:
                assert loader.load("base", "ar.safetensors", nar_lora_override="chosen.safetensors") == ("patched",)
                assert apply.call_args.args[1].nar.path.endswith("chosen.safetensors")


def test_unsafe_metadata_reference_remains_rejected(tmp_path):
    ar = write_fl(tmp_path / "ar.safetensors", "ar", "../outside.safetensors")
    with pytest.raises(ValueError, match="Unsafe acoustic_adapter"):
        parse_adapter(ar)


@pytest.mark.parametrize("loader_name,applier_name", (
    ("YuE2UniversalAdapterLoader", "apply_legacy"),
    ("YuE2NativeUniversalAdapterLoader", "apply_native"),
))
def test_disabled_universal_loader_does_not_resolve_parse_or_apply(
    tmp_path, loader_name, applier_name,
):
    write_fl(tmp_path / "ar.safetensors", "ar", "missing-nar.safetensors")
    with load_nodes(tmp_path) as nodes:
        loader_type = getattr(nodes, loader_name)
        pipeline = object()
        with mock.patch.object(nodes, "_path") as path, mock.patch.object(
            nodes, "parse_adapter"
        ) as parse, mock.patch.object(nodes, applier_name) as apply:
            assert loader_type().load(pipeline, "ar.safetensors", enabled=False)[0] is pipeline
            assert loader_type.IS_CHANGED(adapter="ar.safetensors", enabled=False)[0] is None
            path.assert_not_called()
            parse.assert_not_called()
            apply.assert_not_called()
        with pytest.raises(ValueError, match="missing-nar.safetensors"):
            loader_type().load(pipeline, "ar.safetensors", enabled=True)


def test_load_nodes_removes_only_its_namespace_and_restores_folder_paths(tmp_path):
    before = sys.modules.get("folder_paths")
    sentinel = types.ModuleType("_fl_pair_features_unrelated")
    with mock.patch.dict(sys.modules, {sentinel.__name__: sentinel}):
        with load_nodes(tmp_path) as nodes:
            prefix = nodes.__name__.split(".nodes.", 1)[0]
            assert prefix in sys.modules
            assert any(key.startswith(prefix + ".") for key in sys.modules)
        assert not any(key == prefix or key.startswith(prefix + ".") for key in sys.modules)
        assert sys.modules[sentinel.__name__] is sentinel
    assert sys.modules.get("folder_paths") is before


def test_load_nodes_cleans_namespace_when_import_fails(tmp_path):
    before = sys.modules.get("folder_paths")
    created = []

    def fail_import(name):
        prefix = name.split(".nodes.", 1)[0]
        created.append(prefix)
        sys.modules[prefix + ".partial"] = types.ModuleType(prefix + ".partial")
        raise RuntimeError("synthetic import failure")

    with mock.patch.object(importlib, "import_module", side_effect=fail_import):
        with pytest.raises(RuntimeError, match="synthetic import failure"):
            with load_nodes(tmp_path):
                pass
    assert created
    assert not any(key == created[0] or key.startswith(created[0] + ".") for key in sys.modules)
    assert sys.modules.get("folder_paths") is before

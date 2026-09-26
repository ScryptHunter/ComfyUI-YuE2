import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    return module


def load_abc_tools():
    package = _package("_duration_features", ROOT)
    nodes = _package("_duration_features.nodes", ROOT / "nodes")
    utils = types.ModuleType("_duration_features.nodes.utils")
    utils.safe_stem = lambda value: value
    utils.timestamp_dir = lambda value: value
    with patch.dict(sys.modules, {
        "_duration_features": package,
        "_duration_features.nodes": nodes,
        "_duration_features.nodes.utils": utils,
    }):
        name = "_duration_features.nodes.abc_tools"
        spec = importlib.util.spec_from_file_location(name, ROOT / "nodes" / "abc_tools.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def load_advanced_nodes():
    package = _package("_duration_features", ROOT)
    nodes = _package("_duration_features.nodes", ROOT / "nodes")
    models = _package("_duration_features.models", ROOT / "models")
    fake_yue2 = types.ModuleType("_duration_features.models.yue2")
    utils = types.ModuleType("_duration_features.nodes.utils")
    utils.audio_dict = lambda *args, **kwargs: {}
    utils.timestamp_dir = lambda value: value
    with patch.dict(sys.modules, {
        "_duration_features": package,
        "_duration_features.nodes": nodes,
        "_duration_features.models": models,
        "_duration_features.models.yue2": fake_yue2,
        "_duration_features.nodes.utils": utils,
    }):
        name = "_duration_features.nodes.yue2_advanced"
        spec = importlib.util.spec_from_file_location(name, ROOT / "nodes" / "yue2_advanced.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def load_native_nodes():
    package = _package("_duration_native_test", ROOT)
    nodes = _package("_duration_native_test.nodes", ROOT / "nodes")
    folder_paths = types.ModuleType("folder_paths")
    with patch.dict(sys.modules, {
        "_duration_native_test": package,
        "_duration_native_test.nodes": nodes,
        "folder_paths": folder_paths,
    }):
        name = "_duration_native_test.nodes.comfy_native"
        spec = importlib.util.spec_from_file_location(name, ROOT / "nodes" / "comfy_native.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def test_duration_budget_native_legacy_and_headroom():
    abc_tools = load_abc_tools()
    score = "M:4/4\nL:1/4\nQ:1/4=120\nV:Vocal\n" + "CCCC|\n" * 5
    score_seconds, native, legacy, report = abc_tools.calculate_duration_budget(score)
    assert score_seconds == 10.0
    assert native == 13.0  # 10 * 1.10 + 2 seconds
    assert legacy == 325   # ceil(13 * 25)
    assert "hard ceiling" in report
    assert "target duration" not in report.lower()
    node_outputs = abc_tools.YuE2DurationBudget().calculate(score)
    assert node_outputs[:3] == (score_seconds, native, legacy)


def test_duration_budget_clamps_to_backend_ranges_and_legacy_minimum():
    abc_tools = load_abc_tools()
    short_score = "M:4/4\nL:1/4\nQ:1/4=120\nV:Vocal\nC|\n"
    _, native, legacy, report = abc_tools.calculate_duration_budget(
        short_score, headroom_percent=0, tail_seconds=0,
    )
    assert native == 0.5
    assert legacy == 200
    assert "Legacy semantic max_tokens clamped" in report

    long_score = "M:4/4\nL:1/4\nQ:1/4=120\nV:Vocal\nZ1000|\n"
    _, native, legacy, report = abc_tools.calculate_duration_budget(long_score)
    assert native == 900.0
    assert legacy == 16384
    assert "WARNING:" in report


def test_duration_budget_counts_bracket_chord_as_one_event():
    abc_tools = load_abc_tools()
    header = "M:4/4\nL:1/4\nQ:1/4=120\nK:C\n"
    score, native, legacy, _ = abc_tools.calculate_duration_budget(header + "[CEG]4|\n")
    assert score == pytest.approx(2.0)
    assert native == pytest.approx(4.2)
    assert legacy == 200  # Legacy's configured minimum is higher than 4.2s * 25.

    for body, expected in (("[CEG]2 [DFA]2|", 2.0),
                           ("[C2E2G2]|", 1.0), ("[C2E4G]|", 2.0)):
        duration = abc_tools.analyze_abc(header + body + "\n")[3]
        assert duration == pytest.approx(expected)


def test_duration_budget_distinguishes_inline_fields_from_note_chords():
    abc_tools = load_abc_tools()
    score = ("M:4/4\nL:1/4\nQ:1/4=120\nK:C\n"
             "[K:G][M:4/4][L:1/4][Q:1/4=120][V:Vocal][CEG]4|\n")
    assert abc_tools.analyze_abc(score)[3] == pytest.approx(2.0)


@pytest.mark.parametrize("ending", ("[1", "[2", "[12", "[1,3"))
def test_duration_budget_counts_chord_after_numbered_repeat_ending(ending):
    abc_tools = load_abc_tools()
    score = ("M:4/4\nL:1/4\nQ:1/4=120\nK:C\n"
             f"{ending} C D | [CEG]2|\n")
    assert abc_tools.analyze_abc(score)[3] == pytest.approx(2.0)


def test_duration_budget_rejects_unsupported_bracket_chord_instead_of_undercounting():
    abc_tools = load_abc_tools()
    score = "M:4/4\nL:1/4\nQ:1/4=120\nK:C\n[C~EG]4|\n"
    with pytest.raises(ValueError, match="Unsupported ABC bracket chord"):
        abc_tools.calculate_duration_budget(score)


def test_sampling_settings_override_only_semantic_max_tokens():
    module = load_advanced_nodes()
    settings, _ = module.YuE2SamplingSettings().build(
        "stable", semantic_max_tokens_override=1234,
    )
    semantic = settings["semantic"]
    assert semantic["max_tokens"] == 1234
    assert semantic["temperature"] == 0.85
    assert semantic["top_p"] == 0.9
    assert semantic["top_k"] == 100
    assert semantic["min_tokens"] == 200

    unchanged, _ = module.YuE2SamplingSettings().build(
        "official", semantic_max_tokens_override=0,
    )
    assert unchanged["semantic"]["max_tokens"] == 9000


def test_native_limit_reached_detection_with_mock_conditioning():
    module = load_native_nodes()
    assert module.YuE2NativeGenerateMusic.RETURN_NAMES[:4] == (
        "model", "vae", "conditioning", "seconds",
    )
    assert module.YuE2NativeGenerateMusic.RETURN_NAMES[4:] == (
        "limit_reached", "info",
    )
    # Explicit True wins even when a context-limited generation ends below the
    # requested ceiling (100 frames versus a requested 1000).
    conditioning = [["mocked", {"yue2_frames": 100, "yue2_truncated": True}]]
    reached, info = module.native_limit_diagnostic(conditioning, 20.0, 50)
    assert reached is True
    assert "generated=2.0s requested_ceiling=20.0s limit_reached=True" in info
    assert "may be truncated" in info
    assert "effective budget may have been reduced by the YuE2 context limit" in info

    # Explicit False overrides the old frame-count heuristic at the boundary.
    explicit_false, false_info = module.native_limit_diagnostic(
        [["mocked", {"yue2_frames": 100, "yue2_truncated": False}]], 2.0, 50,
    )
    assert explicit_false is False
    assert "limit_reached=False" in false_info

    # Older ComfyUI metadata without yue2_truncated uses the frame fallback.
    fallback_true, _ = module.native_limit_diagnostic(
        [["mocked", {"yue2_frames": 100}]], 2.0, 50,
    )
    fallback_false, fallback_info = module.native_limit_diagnostic(
        [["mocked", {"yue2_frames": 99}]], 2.0, 50,
    )
    assert fallback_true is True
    assert fallback_false is False
    assert "ended below" in fallback_info

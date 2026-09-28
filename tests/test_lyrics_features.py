"""Focused regressions for lyrics section tags versus production directions."""
import importlib.util
from pathlib import Path
import sys
import types
from unittest import mock
import pytest


ROOT = Path(__file__).resolve().parents[1]
VOYAGE_DIRECTIVES = (
    "[Delicate female soprano, intimate and airy]",
    "[Instrumental grows darker]",
    "[Wide ethereal female soprano]",
    "[Haunting echoes]",
    "[Slow half-time percussion]",
    "[Very soft female soprano]",
    "[Fragile, almost whispered delivery]",
    "[Slow cinematic rise]",
    "[Full emotional soprano]",
    "[Dark witch house beat enters fully]",
    "[Higher soprano register]",
    "[Beat fades away]",
    "[Breathy layered soprano echoes]",
)


def load_module():
    package = types.ModuleType("_lyrics_features")
    package.__path__ = [str(ROOT)]
    nodes = types.ModuleType("_lyrics_features.nodes")
    nodes.__path__ = [str(ROOT / "nodes")]
    models = types.ModuleType("_lyrics_features.models")
    models.__path__ = [str(ROOT / "models")]
    paths = types.ModuleType("_lyrics_features.models.paths")
    paths.list_snapshots = mock.Mock(return_value=[])
    utils = types.ModuleType("_lyrics_features.nodes.utils")
    utils.mono_waveform = mock.Mock()
    utils.timestamp_dir = mock.Mock()
    modules = {
        "_lyrics_features": package,
        "_lyrics_features.nodes": nodes,
        "_lyrics_features.models": models,
        "_lyrics_features.models.sheetsage2": types.ModuleType("_lyrics_features.models.sheetsage2"),
        "_lyrics_features.models.paths": paths,
        "_lyrics_features.nodes.utils": utils,
    }
    with mock.patch.dict(sys.modules, modules):
        name = "_lyrics_features.nodes.sheetsage2"
        spec = importlib.util.spec_from_file_location(name, ROOT / "nodes" / "sheetsage2.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def test_production_brackets_removed_without_losing_lyrics_or_sections():
    module = load_module()
    source = ("[Verse]\nreal lyric\n[Deep witch house drums, huge reverb]\n"
              "more lyrics\n[Chorus]\nfinal lyric\n")
    result, report = module.LyricsStructurer().format(source, "", 4, "")
    assert result == "[Verse]\nreal lyric\nmore lyrics\n[Chorus]\nfinal lyric\n"
    assert "Production/style instructions were removed from Lyrics; move them to Style" in report
    assert "[Deep witch house drums, huge reverb]" in report


def test_structural_case_suffix_unknown_and_crlf():
    module = load_module()
    source = ("[vErSe 2]\r\nline one\r\n[Instrumental Build]\r\n"
              "[Soft reverberant piano]\r\n[Mystery]\r\nline two\r\n")
    result, report = module.LyricsStructurer().format(source, "", 4, "")
    assert result == ("[vErSe 2]\r\nline one\r\n[Instrumental Build]\r\n"
                      "[Mystery]\r\nline two\r\n")
    assert "Unknown bracket tags preserved" in report
    assert "[Mystery]" in report
    assert "[Soft reverberant piano]" in report


def test_production_only_does_not_disable_plain_lyric_structuring():
    module = load_module()
    result, report = module.LyricsStructurer().format(
        "[Low distorted bass pulse]\none\ntwo\n", "", 2, "verse:2")
    assert "[Low distorted bass pulse]" not in result
    assert "[verse]" in result
    assert "one" in result and "two" in result
    assert "Production/style instructions" in report


@pytest.mark.parametrize("directive", VOYAGE_DIRECTIVES)
def test_voyage_production_directive_is_removed(directive):
    module = load_module()
    assert module._bracket_line_kind(directive) == "production"
    result, report = module.LyricsStructurer().format(
        f"[Verse]\nfirst lyric\n{directive}\nsecond lyric\n", "", 4, "")
    assert directive not in result
    assert "first lyric" in result and "second lyric" in result
    assert directive in report


def test_voyage_fragment_keeps_sections_but_no_production_directives():
    module = load_module()
    sections = ("[Verse]", "[Pre-Chorus]", "[Chorus]", "[Final Chorus]",
                "[Outro]", "[Instrumental Build]")
    lines = []
    for index, directive in enumerate(VOYAGE_DIRECTIVES):
        if index < len(sections):
            lines.append(sections[index])
        lines.extend(("a real lyric line", directive))
    source = "\n".join(lines) + "\n"
    result, report = module.LyricsStructurer().format(source, "", 4, "")
    for section in sections:
        assert module._bracket_line_kind(section) == "section"
        assert section in result
    for directive in VOYAGE_DIRECTIVES:
        assert directive not in result
    assert "Production/style instructions were removed" in report
    assert module._bracket_line_kind("[Mystery]") == "unknown"

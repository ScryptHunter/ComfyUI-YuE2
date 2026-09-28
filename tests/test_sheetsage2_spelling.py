"""Lightweight SheetSage2 chord-spelling compatibility regressions."""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

from compat.sheetsage2_spelling import correct_abc_chord_spellings


def _load_installed_core():
    root = os.environ.get("COMFYUI_ROOT")
    if not root:
        return None
    path = Path(root) / "comfy" / "audio_encoders" / "sheetsage2_abc.py"
    if not path.is_file():
        return None
    name = "_test_installed_sheetsage2_abc"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sheetsage2_core():
    module = _load_installed_core()
    if module is None:
        pytest.skip("Set COMFYUI_ROOT to validate the installed ComfyUI SheetSage2 helper")
    if not all(callable(getattr(module, name, None)) for name in
               ("normalize_key_name", "correct_chord_spelling")):
        pytest.skip("Installed ComfyUI does not expose both SheetSage2 spelling helpers")
    return module


@pytest.mark.parametrize(
    ("chord", "key", "expected"),
    [
        ("C:maj", "E:major", "C:maj"),
        ("A#:maj", "G:major", "Bb:maj"),
        ("D#:maj", "G:major", "Eb:maj"),
        ("G#:7", "C:major", "Ab:7"),
        ("A#:min", "C:major", "Bb:min"),
        ("C#:hdim7", "G:major", "C#:hdim7"),
        ("F:hdim7", "G#:minor", "E#:hdim7"),
    ],
)
def test_installed_core_upstream_chord_spelling_cases(sheetsage2_core, chord, key, expected):
    corrected = sheetsage2_core.correct_chord_spelling(chord, key)
    assert corrected == expected

    source_root, source_descriptor = chord.split(":", 1)
    corrected_root, corrected_descriptor = corrected.split(":", 1)
    assert sheetsage2_core._pitch_class(source_root)[0] == sheetsage2_core._pitch_class(corrected_root)[0]
    assert corrected_descriptor == source_descriptor


def test_installed_core_preserves_double_accidentals_and_row_timing(sheetsage2_core):
    source = [[1.25, 4.75, "D:dim7"]]
    corrected = sheetsage2_core.correct_chord_rows(source, [[0.0, 5.0, "E:major"]])
    assert corrected == [[1.25, 4.75, "C##:dim7"]]
    assert sheetsage2_core._pitch_class("D")[0] == sheetsage2_core._pitch_class("C##")[0]
    assert corrected[0][:2] == source[0][:2]

    slash_chord = sheetsage2_core.correct_chord_spelling("A#:7/E", "G:major")
    assert slash_chord == "Bb:7/E"
    assert slash_chord.split(":", 1)[1] == "7/E"
    assert sheetsage2_core._pitch_class("A#")[0] == sheetsage2_core._pitch_class("Bb")[0]


def test_installed_core_normalizes_enharmonic_keys(sheetsage2_core):
    assert sheetsage2_core.normalize_key_name("A#:major") == "Bb:major"
    assert sheetsage2_core.normalize_key_name("D#:major") == "Eb:major"
    assert sheetsage2_core.normalize_key_name("G#:minor") == "G#:minor"


def test_legacy_abc_adapter_uses_header_and_inline_key_without_touching_timeline(sheetsage2_core):
    source = (
        "X:1\r\n"
        "K:G\r\n"
        'V: Vocal\r\n'
        '"D#7/E" C2|[K:E]"D#" D2| % "D#" in comment\r\n'
    )
    corrected = correct_abc_chord_spellings(
        source, corrector=sheetsage2_core.correct_chord_spelling,
        normalizer=sheetsage2_core.normalize_key_name,
    )

    assert corrected == (
        "X:1\r\n"
        "K:G\r\n"
        'V: Vocal\r\n'
        '"Eb7/E" C2|[K:E]"D#" D2| % "D#" in comment\r\n'
    )
    assert re.sub(r'"[^"\r\n]*"', '"CHORD"', corrected) == re.sub(
        r'"[^"\r\n]*"', '"CHORD"', source
    )


def test_legacy_abc_adapter_preserves_unsupported_chords_and_no_key_scores(sheetsage2_core):
    source = '"A#unsupported" C|\nK:G\n"A#" D|\n'
    assert correct_abc_chord_spellings(
        source, corrector=sheetsage2_core.correct_chord_spelling,
        normalizer=sheetsage2_core.normalize_key_name,
    ) == '"A#unsupported" C|\nK:G\n"Bb" D|\n'


def test_header_and_inline_keys_stay_source_preserving_with_canonical_chord_context():
    normalizer = Mock(side_effect=lambda key: {"D#:major": "Eb:major"}.get(key, key))
    corrector = Mock(side_effect=lambda chord, key: "Eb:maj" if (
        chord == "D#:maj" and key == "Eb:major"
    ) else chord)
    source = ('X:1\r\nK:  D#  % original key\r\nV:Vocal\r\n'
              '"D#" C2|[K: D# ]"D#" D2| % "D#" in comment\r\n')
    expected = ('X:1\r\nK:  D#  % original key\r\nV:Vocal\r\n'
                '"Eb" C2|[K: D# ]"Eb" D2| % "D#" in comment\r\n')
    assert correct_abc_chord_spellings(
        source, corrector=corrector, normalizer=normalizer,
    ) == expected
    assert normalizer.call_count == 2
    assert all(call.args == ("D#:major",) for call in normalizer.call_args_list)
    assert all(call.args == ("D#:maj", "Eb:major") for call in corrector.call_args_list)
    assert corrector.call_count == 2


@pytest.mark.parametrize("label,mode", [
    ("A", "major"), ("Amaj", "major"), ("Amajor", "major"),
    ("Am", "minor"), ("Amin", "minor"), ("Aminor", "minor"),
    ("A major", "major"), ("A maj", "major"),
    ("A minor", "minor"), ("A min", "minor"), ("A m", "minor"),
])
def test_common_abc_key_aliases_use_the_correct_core_mode(label, mode):
    normalizer = Mock(side_effect=lambda key: key)
    corrector = Mock(side_effect=lambda chord, key: chord)
    source = f'K:{label}\n"C" C|\n'
    assert correct_abc_chord_spellings(
        source, corrector=corrector, normalizer=normalizer,
    ) == source
    normalizer.assert_called_once_with(f"A:{mode}")
    corrector.assert_called_once_with("C:maj", f"A:{mode}")


@pytest.mark.parametrize("label,mode", [
    ("A major", "major"), ("A maj", "major"),
    ("A minor", "minor"), ("A min", "minor"), ("A m", "minor"),
])
def test_spaced_inline_key_alias_preserves_source_and_updates_chord_context(label, mode):
    normalizer = Mock(side_effect=lambda key: key)
    corrector = Mock(side_effect=lambda chord, key: "Eb:maj" if key == f"A:{mode}" else chord)
    source = f'K:C\r\n"D#" C|[K:{label}]"D#" D| % comment "D#"\r\n'
    expected = f'K:C\r\n"D#" C|[K:{label}]"Eb" D| % comment "D#"\r\n'
    assert correct_abc_chord_spellings(
        source, corrector=corrector, normalizer=normalizer,
    ) == expected
    assert [call.args[1] for call in corrector.call_args_list] == ["C:major", f"A:{mode}"]


def test_flat_key_root_is_passed_to_core_with_its_accidental_intact():
    normalizer = Mock(side_effect=lambda key: key)
    corrector = Mock(side_effect=lambda chord, key: chord)
    source = 'K:Bb\n"C" C|\n'
    assert correct_abc_chord_spellings(
        source, corrector=corrector, normalizer=normalizer,
    ) == source
    normalizer.assert_called_once_with("Bb:major")
    corrector.assert_called_once_with("C:maj", "Bb:major")


def test_inline_key_change_updates_the_key_used_for_following_chords():
    normalizer = Mock(side_effect=lambda key: {"D#:major": "Eb:major"}.get(key, key))
    corrector = Mock(side_effect=lambda chord, key: "Eb:maj" if key == "Eb:major" else chord)
    source = 'K:A\n"D#" C|[K:D#]"D#" D|\n'
    expected = 'K:A\n"D#" C|[K:D#]"Eb" D|\n'
    assert correct_abc_chord_spellings(
        source, corrector=corrector, normalizer=normalizer,
    ) == expected
    assert [call.args[1] for call in corrector.call_args_list] == ["A:major", "Eb:major"]


def test_enharmonic_header_key_does_not_reinterpret_bare_melody_notes():
    normalizer = Mock(side_effect=lambda key: {"C#:major": "Db:major"}.get(key, key))
    corrector = Mock(side_effect=lambda chord, key: chord)
    source = "K:C#\r\nV:Vocal\r\nC D E F|\r\n"
    assert correct_abc_chord_spellings(
        source, corrector=corrector, normalizer=normalizer,
    ) == source
    normalizer.assert_called_once_with("C#:major")
    corrector.assert_not_called()


def test_unsupported_modal_key_does_not_reuse_the_previous_key():
    normalizer = Mock(side_effect=lambda key: key)
    corrector = Mock(side_effect=lambda chord, key: chord)
    source = ('K:A\n"C"|[K:Ddor]"D#"|[K:D dor]"D#"|'
              '[K:Emix]"D#"|[K:E mixolydian]"D#"|\n'
              'K:F lyd\n"D#"|\nK:A minor\n"C"|\n')
    assert correct_abc_chord_spellings(
        source, corrector=corrector, normalizer=normalizer,
    ) == source
    assert [call.args for call in normalizer.call_args_list] == [
        ("A:major",), ("A:minor",),
    ]
    assert [call.args for call in corrector.call_args_list] == [
        ("C:maj", "A:major"), ("C:maj", "A:minor"),
    ]


@pytest.mark.parametrize("available", ["correct_chord_spelling", "normalize_key_name"])
def test_abc_adapter_is_a_noop_if_a_core_helper_is_unavailable(monkeypatch, available):
    import compat.sheetsage2_spelling as spelling

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    audio_encoders = types.ModuleType("comfy.audio_encoders")
    audio_encoders.__path__ = []
    old_core = types.ModuleType("comfy.audio_encoders.sheetsage2_abc")
    setattr(old_core, available, lambda *args: args[0])
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.audio_encoders", audio_encoders)
    monkeypatch.setitem(sys.modules, "comfy.audio_encoders.sheetsage2_abc", old_core)
    source = 'K:D#\n"A#" C|\n'
    assert spelling._load_core_helpers() is None
    assert spelling.correct_abc_chord_spellings(source) == source

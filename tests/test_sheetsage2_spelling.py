"""Lightweight SheetSage2 chord-spelling compatibility regressions."""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

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
        source, corrector=sheetsage2_core.correct_chord_spelling
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
        source, corrector=sheetsage2_core.correct_chord_spelling
    ) == '"A#unsupported" C|\nK:G\n"Bb" D|\n'


def test_abc_adapter_is_a_noop_if_core_corrector_is_unavailable(monkeypatch):
    import compat.sheetsage2_spelling as spelling

    source = 'K:G\n"A#" C|\n'
    monkeypatch.setattr(spelling, "_load_core_corrector", lambda: None)
    assert spelling.correct_abc_chord_spellings(source) == source

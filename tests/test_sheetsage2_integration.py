"""Lightweight tests for Legacy SheetSage2 ABC post-processing."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

import compat.sheetsage2_spelling as spelling


ROOT = Path(__file__).resolve().parents[1]


class FakeWaveform:
    ndim = 1

    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def reshape(self, *_shape): return self
    def contiguous(self): return self


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parameters(self):
        return iter([types.SimpleNamespace(dtype="fp32")])

    def transcribe(self, waveform, **kwargs):
        self.calls.append((waveform, kwargs))
        return self.result


@pytest.fixture
def sheetsage_module():
    name = "_legacy_sheetsage_test"
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT)]
    compat = types.ModuleType(name + ".compat")
    compat.__path__ = [str(ROOT / "compat")]
    models = types.ModuleType(name + ".models")
    models.__path__ = [str(ROOT / "models")]
    loader = types.ModuleType(name + ".compat.sheetsage2")
    loader.load_sheetsage2 = lambda *args, **kwargs: None
    paths = types.ModuleType(name + ".models.paths")
    paths.model_dirs = lambda *_args: []
    paths.resolve = lambda *_args: ""
    torch = types.ModuleType("torch")
    torch.bfloat16 = object()
    torch.float32 = object()
    modules = {
        name: package,
        name + ".compat": compat,
        name + ".compat.sheetsage2": loader,
        name + ".compat.sheetsage2_spelling": spelling,
        name + ".models": models,
        name + ".models.paths": paths,
        "torch": torch,
    }
    module_name = name + ".models.sheetsage2"
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(module_name, ROOT / "models" / "sheetsage2.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


def test_transcribe_corrects_returned_abc_and_saved_score_without_changing_events(
    sheetsage_module, monkeypatch, tmp_path,
):
    normalizer = mock.Mock(side_effect=lambda key: {"D#:major": "Eb:major"}.get(key, key))
    corrector = mock.Mock(side_effect=lambda chord, key: "Eb:maj" if (
        chord == "D#:maj" and key == "Eb:major"
    ) else chord)
    monkeypatch.setattr(spelling, "_load_core_helpers", lambda: (corrector, normalizer))

    source = 'K:D#\r\n"D#" C2|\r\n'
    expected = 'K:Eb\r\n"Eb" C2|\r\n'
    score_path = tmp_path / "score.abc"
    with score_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(source)
    events = [{"time": 0.0, "values": {"structure": "verse"}},
              {"time": 2.5, "values": {"structure": "chorus"}}]
    chord_rows = [[0.25, 1.75, "D#:maj"]]
    result = {"abc": source, "events": events, "chords": chord_rows, "midi": b"raw-midi"}
    model = FakeModel(result)

    abc_text, _structure, returned, midi = sheetsage_module.transcribe(
        model, FakeWaveform(), 24000, output_dir=str(tmp_path),
    )
    assert abc_text == expected == result["abc"]
    assert returned is result
    assert score_path.read_bytes() == expected.encode("utf-8")
    assert result["events"] is events
    assert result["chords"] is chord_rows
    assert chord_rows == [[0.25, 1.75, "D#:maj"]]
    assert result["midi"] == midi == b"raw-midi"
    assert model.calls[0][1]["sampling_rate"] == 24000
    assert corrector.call_args.args == ("D#:maj", "Eb:major")


def test_transcribe_does_not_rewrite_unchanged_score(sheetsage_module, monkeypatch, tmp_path):
    monkeypatch.setattr(spelling, "_load_core_helpers", lambda: (
        lambda chord, key: chord, lambda key: key,
    ))
    source = 'K:C\n"C" C|\n'
    score_path = tmp_path / "score.abc"
    score_path.write_text(source, encoding="utf-8")
    model = FakeModel({"abc": source, "midi": b"raw-midi"})

    with mock.patch.object(sheetsage_module.io, "open", side_effect=AssertionError("score rewritten")):
        abc_text, _structure, result, _midi = sheetsage_module.transcribe(
            model, FakeWaveform(), 24000, output_dir=str(tmp_path),
        )
    assert abc_text == result["abc"] == source
    assert score_path.read_text(encoding="utf-8") == source


def test_transcribe_midi_recovery_still_writes_abc(sheetsage_module, tmp_path):
    pretty_midi = types.ModuleType("pretty_midi")

    class FakeMidi:
        def __init__(self, data):
            assert data.read() == b"raw-midi"
            note = types.SimpleNamespace(start=0.0, end=0.5, pitch=60)
            self.instruments = [types.SimpleNamespace(is_drum=False, name="Vocal", notes=[note])]

        def get_tempo_changes(self):
            return [0.0], [120.0]

    pretty_midi.PrettyMIDI = FakeMidi
    result = {"abc": "", "midi": b"raw-midi", "events": []}
    model = FakeModel(result)
    with mock.patch.dict(sys.modules, {"pretty_midi": pretty_midi}):
        abc_text, _structure, returned, midi = sheetsage_module.transcribe(
            model, FakeWaveform(), 24000, output_dir=str(tmp_path),
            abc_error_mode="snap_invalid_notes",
        )

    assert abc_text == returned["abc"]
    assert "V:Vocal" in abc_text and "C8" in abc_text
    assert returned["abc_recovery"] == "snap_invalid_notes"
    assert (tmp_path / "score_recovered.abc").read_text(encoding="utf-8") == abc_text
    assert midi == result["midi"] == b"raw-midi"

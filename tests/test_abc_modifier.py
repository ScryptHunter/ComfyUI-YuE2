"""Regression coverage for key-aware, source-preserving ABC transposition."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nodes" / "abc_tools.py"


def load_module():
    package = types.ModuleType("_abc_modifier_test")
    package.__path__ = [str(ROOT)]
    nodes = types.ModuleType("_abc_modifier_test.nodes")
    nodes.__path__ = [str(ROOT / "nodes")]
    utils = types.ModuleType("_abc_modifier_test.nodes.utils")
    utils.safe_stem = lambda value: value
    utils.timestamp_dir = lambda value: value
    with patch.dict(sys.modules, {
        "_abc_modifier_test": package,
        "_abc_modifier_test.nodes": nodes,
        "_abc_modifier_test.nodes.utils": utils,
    }):
        name = "_abc_modifier_test.nodes.abc_tools"
        spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class ABCModifierTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_pitch_invariant_across_contiguous_notes_accidentals_voices_and_keys(self):
        source = (
            "X:1\nT:Keep spacing\nM:4/4\nL:1/8\nK: C\n"
            "V:Vocal\nCDEF ^F F| F =B B| [K:G] F F|[K:Db] B, B,|\n"
            "V:Ins\nC D _E E|E|\n"
        )
        before = self.mod.score_notes(source)
        output, _ = self.mod.ABCModifier().modify(
            source, "keep", 120, 1, "whole_score", 2, False, "", octave_shift=-1,
        )
        after = self.mod.score_notes(output)
        self.assertEqual(len(before), len(after))
        self.assertTrue(before)
        for a, b in zip(before, after):
            self.assertEqual(b[2] - a[2], -10, (a, b, output))
        self.assertIn("[K:A]", output)
        self.assertIn("[K:Eb]", output)

    def test_barline_resets_accidental_and_inline_fields_are_not_notes(self):
        source = "K:C\nV:Vocal\n_E E|E [K:G] F [V:Ins] C !accent! D\n"
        notes = self.mod.score_notes(source)
        pitches = [pitch for _, _, pitch in notes]
        self.assertEqual(pitches[:3], [63, 63, 64])
        self.assertEqual(pitches[3:], [66, 60, 62])
        self.assertEqual(len(pitches), 6)

    def test_inline_meter_and_unit_fields_are_preserved_and_not_notes(self):
        source = "K:C\nV:Vocal\nC [M:4/4] D [L:1/16] E\n"
        before = self.mod.score_notes(source)
        output, _ = self.mod.ABCModifier().modify(
            source, "keep", 120, 1, "whole_score", 2, False, "",
        )
        after = self.mod.score_notes(output)
        self.assertEqual([pitch for _, _, pitch in before], [60, 62, 64])
        self.assertEqual([pitch - original[2] for (voice, section, pitch), original
                          in zip(after, before)], [2, 2, 2])
        self.assertIn("[M:4/4]", output)
        self.assertIn("[L:1/16]", output)

    def test_normal_key_change_does_not_reinterpret_source_as_destination(self):
        source = "K:C\nV:Vocal\nF F|\nK:G\nF F|\n"
        before = self.mod.score_notes(source)
        output, _ = self.mod.ABCModifier().modify(
            source, "keep", 120, 1, "whole_score", 2, False, "",
        )
        after = self.mod.score_notes(output)
        self.assertEqual([b[2] - a[2] for a, b in zip(before, after)], [2] * 4)
        self.assertIn("K:D", output)
        self.assertIn("K:A", output)

    def test_chord_roots_and_slash_basses_transpose_in_both_directions(self):
        cases = {
            '"E/G#"': ('"F#/A#"', '"D/Gb"'),
            '"D#maj7/F##"': ('"Fmaj7/A"', '"C#maj7/F"'),
            '"Bb7/Db"': ('"C7/Eb"', '"Ab7/B"'),
            '"Cb"': ('"Db"', '"A"'),
            '"Bbb"': ('"B"', '"G"'),
            '"C6/9"': ('"D6/9"', '"Bb6/9"'),
        }
        for chord, (up, down) in cases.items():
            with self.subTest(chord=chord):
                self.assertEqual(self.mod._transpose_chords(chord, 2), up)
                self.assertEqual(self.mod._transpose_chords(chord, -2), down)

    def test_noop_is_byte_for_byte_source_preserving(self):
        source = "X:1\r\nK: C  \r\nV:Vocal\r\nCDEF  |  [K:G] F  % keep spacing\r\n\r\n"
        output, _ = self.mod.ABCModifier().modify(
            source, "keep", 120, 1, "none", 0, False, "",
            vocal_transpose=0, instrumental_transpose=0, octave_shift=0,
        )
        self.assertEqual(output, source)

    def test_transposition_only_changes_pitch_spans_and_key_fields(self):
        source = "K: C\nV:Vocal\nC2  D- !accent! E | [K:G] F % keep CDEF and \"quotes\"\n% keep this comment\n"
        output, _ = self.mod.ABCModifier().modify(
            source, "keep", 120, 1, "whole_score", 2, False, "",
        )
        self.assertIn("D2  E- !accent! F | [K:A] G", output)
        self.assertIn('% keep CDEF and "quotes"\n', output)
        self.assertIn("% keep this comment\n", output)

    def test_current_main_node_contracts_remain(self):
        self.assertEqual(self.mod.ABCAnalyzer().analyze("")[0],
                         "ABC input is empty (score planning is disabled).")
        style_extra = self.mod.YuE2StyleBuilder.INPUT_TYPES()["required"]["extra"][1]
        self.assertTrue(style_extra["multiline"])
        expected_names = {
            "YuE2ABCAnalyzer": "YuE2 ABC Analyzer",
            "YuE2ABCDurationBudget": "YuE2 ABC Duration Budget",
            "YuE2ABCModifier": "YuE2 ABC Modifier",
            "YuE2VocalRangeRetarget": "YuE2 Vocal Range Retarget",
            "YuE2MelodyCleanup": "YuE2 Melody Cleanup",
            "YuE2LoadABCFile": "YuE2 ABC File Loader",
            "YuE2SaveABCFile": "YuE2 ABC File Saver",
            "YuE2StyleBuilder": "YuE2 Style Prompt Builder",
            "YuE2LyricsMelodyFit": "YuE2 Lyrics / Melody Fit Analyzer",
        }
        self.assertEqual(self.mod.NODE_DISPLAY_NAME_MAPPINGS, expected_names)
        self.assertEqual(set(self.mod.NODE_CLASS_MAPPINGS), set(expected_names))

    def test_analyzer_multimeasure_rests_use_meter_timeline(self):
        for rest, expected in (("Z", 2.0), ("Z2", 4.0), ("Z4", 8.0)):
            with self.subTest(rest=rest):
                source = f"M:4/4\nL:1/32\nQ:1/4=120\n\nV:Vocal\n{rest}|\n"
                self.assertAlmostEqual(self.mod.analyze_abc(source)[3], expected)

    def test_analyzer_distinguishes_multimeasure_Z4_from_ordinary_z4(self):
        prefix = "M:4/4\nL:1/32\nQ:1/4=120\n\nV:Vocal\n"
        multimeasure = self.mod.analyze_abc(prefix + "Z4|\n")[3]
        ordinary_rest = self.mod.analyze_abc(prefix + "z4|\n")[3]
        self.assertAlmostEqual(multimeasure, 8.0)
        self.assertAlmostEqual(ordinary_rest, 0.25)

    def test_analyzer_voice_timelines_are_parallel_and_continue_on_switch(self):
        source = (
            "M:4/4\nL:1/4\nQ:1/4=120\n\n"
            "V:Vocal\nZ2|\n\n"
            "V:Ins\nCCCC|CCCC|CCCC|CCCC|\n"
            "V:Vocal\nZ2|\n"
        )
        # Both voices span four bars; Vocal's timeline continues after switching away.
        self.assertAlmostEqual(self.mod.analyze_abc(source)[3], 8.0)

    def test_analyzer_note_and_rest_lengths_ignore_non_music_text(self):
        source = (
            "M:4/4\nL:1/8\nQ:1/4=120\n\nV:Vocal\n"
            'C C4 C/2 C/ C3/2 z24 "quoted chord" !trill! [K:G] | % C4\n'
        )
        # (1 + 4 + .5 + .5 + 1.5 + 24) eighth-note units = 31.5 units.
        self.assertAlmostEqual(self.mod.analyze_abc(source)[3], 31.5 * 0.25)

    def test_analyzer_supports_inline_voice_switch_and_tempo_beat_unit(self):
        source = "M:2/4\nL:1/4\nQ:1/8=120\nV:Vocal\nC [V:Ins] Z|\n"
        # The Ins one-bar rest lasts four eighth-note beats = two seconds.
        self.assertAlmostEqual(self.mod.analyze_abc(source)[3], 2.0)

    def test_analyzer_integrates_tempo_changes_only_after_each_Q(self):
        source = "M:4/4\nL:1/4\nQ:1/4=120\nV:Vocal\nCCCC|\nQ:1/4=60\nCCCC|\n"
        self.assertAlmostEqual(self.mod.analyze_abc(source)[3], 6.0)

    def test_abc_modifier_override_updates_analyzed_duration(self):
        source = "M:4/4\nL:1/4\nQ:1/4=120\nV:Vocal\nCCCC|\n"
        before = self.mod.analyze_abc(source)[3]
        modified, _ = self.mod.ABCModifier().modify(
            source, "override", 60, 1, "none", 0, False, "",
        )
        self.assertIn("Q:1/4=60", modified)
        self.assertAlmostEqual(self.mod.analyze_abc(modified)[3], before * 2)

    def test_tempo_multiplier_preserves_Q_beat_unit(self):
        source = "M:4/4\nL:1/4\nQ:1/8=120\nV:Vocal\nCCCC|\n"
        modified, _ = self.mod.ABCModifier().modify(
            source, "multiplier", 120, 2, "none", 0, False, "",
        )
        self.assertIn("Q:1/8=240", modified)
        self.assertNotIn("Q:1/4=240", modified)

    def test_tempo_override_rewrites_header_and_inline_Q_fields(self):
        source = "M:4/4\nL:1/4\nQ:1/4=120\nV:Vocal\nCCCC|[Q:1/8=80]CCCC|\n"
        modified, _ = self.mod.ABCModifier().modify(
            source, "override", 100, 1, "none", 0, False, "",
        )
        self.assertIn("Q:1/4=100", modified)
        self.assertIn("[Q:1/8=100]", modified)

    def test_tempo_multiplier_rewrites_each_Q_from_its_own_source_bpm(self):
        source = "M:4/4\nL:1/4\nQ:1/4=120\nV:Vocal\nC[Q:1/8=80]CCC|\n"
        modified, _ = self.mod.ABCModifier().modify(
            source, "multiplier", 120, 2, "none", 0, False, "",
        )
        self.assertIn("Q:1/4=240", modified)
        self.assertIn("[Q:1/8=160]", modified)

    def test_inline_only_Q_is_rewritten_without_duplicate_header(self):
        source = "M:4/4\nL:1/4\nV:Vocal\nC[Q:1/4=120]CCC|\n"
        modified, _ = self.mod.ABCModifier().modify(
            source, "override", 100, 1, "none", 0, False, "",
        )
        self.assertIn("[Q:1/4=100]", modified)
        self.assertFalse(any(line.startswith("Q:") for line in modified.splitlines()))

    def test_tempo_keep_preserves_inline_Q_and_source_text(self):
        source = "M:4/4\r\nL:1/4\r\nV:Vocal\r\nC [Q:1/8 = 80] CCC|  % source\r\n"
        modified, _ = self.mod.ABCModifier().modify(
            source, "keep", 100, 2, "none", 0, False, "",
        )
        self.assertEqual(modified, source)

    def test_inline_meter_length_and_tempo_fields_apply_at_position(self):
        source = "M:4/4\nL:1/4\nQ:1/4=120\nV:Vocal\nCCCC|[Q:1/4=60]CCCC|\n"
        self.assertAlmostEqual(self.mod.analyze_abc(source)[3], 6.0)
        inline_length = "M:4/4\nL:1/4\nQ:1/4=120\nV:Vocal\nC [L:1/8]C\n"
        self.assertAlmostEqual(self.mod.analyze_abc(inline_length)[3], 0.75)
        inline_meter = "M:4/4\nL:1/4\nQ:1/4=120\nV:Vocal\nC [M:2/4]Z\n"
        self.assertAlmostEqual(self.mod.analyze_abc(inline_meter)[3], 1.5)

    def test_plus_decorations_are_preserved_and_not_counted_or_transposed(self):
        source = "M:4/4\nL:1/8\nQ:1/4=120\nV:Vocal\n+fermata+C +trill+D|\n"
        before = self.mod.score_notes(source)
        duration_before = self.mod.analyze_abc(source)[3]
        modified, _ = self.mod.ABCModifier().modify(
            source, "keep", 120, 1, "whole_score", 2, False, "",
        )
        self.assertIn("+fermata+D +trill+E|", modified)
        self.assertEqual([p for _, _, p in self.mod.score_notes(modified)],
                         [p + 2 for _, _, p in before])
        self.assertAlmostEqual(self.mod.analyze_abc(modified)[3], duration_before)

    def test_vocal_range_retarget_keeps_already_in_range_melody(self):
        abc = "K:C\nV:Vocal\nC D E F G A|\n"
        _, shift, _ = self.mod.VocalRangeRetarget().retarget(
            abc, "female_soprano", "nearest_key_safe", 0,
        )
        self.assertEqual(shift, 0)

    def test_manual_shift_takes_priority_over_original_target(self):
        abc = "K:C\nV:Vocal\nC D E|\n"
        before = [pitch for _, _, pitch in self.mod.score_notes(abc)]
        for requested in (5, -7):
            with self.subTest(requested=requested):
                output, shift, _ = self.mod.VocalRangeRetarget().retarget(
                    abc, "original", "manual", requested,
                )
                self.assertEqual(shift, requested)
                after = [pitch for _, _, pitch in self.mod.score_notes(output)]
                self.assertEqual([pitch - old for pitch, old in zip(after, before)],
                                 [requested] * len(before))

    def test_original_target_keeps_automatic_mode_unshifted(self):
        abc = "K:C\nV:Vocal\nC D E|\n"
        output, shift, _ = self.mod.VocalRangeRetarget().retarget(
            abc, "original", "nearest_key_safe", 12,
        )
        self.assertEqual(shift, 0)
        self.assertEqual(output, abc)

    def test_manual_vocal_shift_is_independent_of_whole_score_transpose(self):
        abc = "K:C\nV:Vocal\nC D|\nV:Ins\nE F|\n"
        vocal_shifted, shift, _ = self.mod.VocalRangeRetarget().retarget(
            abc, "original", "manual", 5,
        )
        self.assertEqual(shift, 5)
        final, _ = self.mod.ABCModifier().modify(
            vocal_shifted, "keep", 120, 1, "whole_score", 2, False, "",
        )
        before = self.mod.score_notes(abc)
        after = self.mod.score_notes(final)
        self.assertEqual([(new[0], new[2] - old[2]) for old, new in zip(before, after)],
                         [("Vocal", 7), ("Vocal", 7), ("Ins", 2), ("Ins", 2)])

    def test_major_and_minor_key_aliases_have_distinct_signatures(self):
        self.assertEqual(self.mod._key_accidentals("Amajor")["F"], 1)
        self.assertEqual(self.mod._key_accidentals("Amaj")["F"], 1)
        self.assertEqual(self.mod._key_accidentals("Amin")["F"], 0)
        self.assertEqual(self.mod._key_accidentals("Aminor")["F"], 0)

    def test_vocal_range_retarget_moves_octave_below_melody_up(self):
        abc = "K:C\nV:Vocal\nC, D, E, F, G, A,|\n"
        _, shift, _ = self.mod.VocalRangeRetarget().retarget(
            abc, "female_soprano", "nearest_key_safe", 0,
        )
        self.assertEqual(shift, 12)

    def test_vocal_range_retarget_moves_high_melody_only_when_coverage_improves(self):
        abc = "K:C\nV:Vocal\nc' d' e'|\n"
        _, shift, _ = self.mod.VocalRangeRetarget().retarget(
            abc, "female_soprano", "nearest_key_safe", 0,
        )
        self.assertEqual(shift, -12)
        before = self.mod.score_notes(abc)
        after = self.mod.score_notes(self.mod.VocalRangeRetarget().retarget(
            abc, "female_soprano", "nearest_key_safe", 0,
        )[0])
        self.assertGreater(sum(60 <= p <= 84 for _, _, p in after),
                           sum(60 <= p <= 84 for _, _, p in before))

    def test_vocal_range_safe_candidate_prefers_zero_shift_on_exact_tie(self):
        abc = "K:C\nV:Vocal\nC c' c' c''|\n"
        _, shift, _ = self.mod.VocalRangeRetarget().retarget(
            abc, "female_soprano", "nearest_key_safe", 0,
        )
        self.assertEqual(shift, 0)

    def test_nearest_octave_and_key_safe_have_distinct_documented_semantics(self):
        abc = "K:C\nV:Vocal\nC c' c' c''|\n"
        node = self.mod.VocalRangeRetarget()
        _, safe_shift, _ = node.retarget(abc, "female_soprano", "nearest_key_safe", 0)
        _, octave_shift, _ = node.retarget(abc, "female_soprano", "nearest_octave", 0)
        self.assertEqual(safe_shift, 0)
        self.assertEqual(octave_shift, -12)

    def test_nearest_octave_refuses_a_centering_shift_that_reduces_coverage(self):
        abc = "K:C\nV:Vocal\nC C C c' c' c' c''|\n"
        _, shift, _ = self.mod.VocalRangeRetarget().retarget(
            abc, "female_soprano", "nearest_octave", 0,
        )
        self.assertEqual(shift, 0)

    def test_nearest_octave_chooses_second_safe_centering_candidate(self):
        abc = "K:C\nV:Vocal\nC,, C,, ^F, ^f ^f|\n"
        before = [pitch for voice, _, pitch in self.mod.score_notes(abc) if voice == "Vocal"]
        self.assertEqual(before, [36, 36, 54, 78, 78])
        output, shift, _ = self.mod.VocalRangeRetarget().retarget(
            abc, "female_soprano", "nearest_octave", 0,
        )
        self.assertEqual(shift, 24)
        after = [pitch for voice, _, pitch in self.mod.score_notes(output) if voice == "Vocal"]
        self.assertEqual(after, [pitch + 24 for pitch in before])

    def test_vocal_range_report_clarifies_pitch_range_is_not_gender_or_timbre(self):
        abc = "K:C\nV:Vocal\nC D E|\n"
        _, _, report = self.mod.VocalRangeRetarget().retarget(
            abc, "female_soprano", "nearest_key_safe", 0,
        )
        self.assertIn("target=female_soprano range=C4..C6", report)
        self.assertIn("source=C4..E4 median=D4", report)
        self.assertIn("result=C4..E4 median=D4", report)
        self.assertIn("applied=+0 semitones", report)
        self.assertIn("Pitch range only; vocal gender/timbre is controlled by the style prompt.", report)
        self.assertIn("female soprano vocal, light feminine timbre, clear head voice", report)

    def test_melody_cleanup_protects_decorations_comments_chords_and_inline_fields(self):
        source = (
            'K:C\nL:1/8\nQ:1/4=120\nV:Vocal\n'
            '+fermata+C !trill! D "quoted CDEF" [C E G]4 {C}D [M:4/4][L:1/8][Q:1/4=120] '
            '[K:C][V:Vocal] % comment CDEF\n'
        )
        output, _ = self.mod.MelodyCleanup().clean(
            source, "custom", 300, False, 24,
        )
        self.assertIn("+fermata+z !trill! z", output)
        self.assertIn('"quoted CDEF"', output)
        self.assertIn("[C E G]4 {C}z", output)
        self.assertIn("[M:4/4][L:1/8][Q:1/4=120] [K:C][V:Vocal]", output)
        self.assertIn("% comment CDEF", output)

    def test_melody_cleanup_preserves_effective_pitch_after_removed_accidental(self):
        source = "K:C\nL:1/32\nQ:1/4=120\nV:Vocal\n^F F4|\n"
        original = self.mod.score_notes(source)
        output, report = self.mod.MelodyCleanup().clean(
            source, "custom", 100, False, 24,
        )
        self.assertIn("z ^F4|", output)
        kept = self.mod.score_notes(output)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][2], original[1][2])
        self.assertIn("short_notes_removed=1", report)

    def test_melody_cleanup_uses_L_and_Q_beat_unit_for_note_timing(self):
        source = "K:C\nL:1/8\nQ:1/8=120\nV:Vocal\nC|\n"
        output, report = self.mod.MelodyCleanup().clean(
            source, "custom", 300, False, 24,
        )
        self.assertIn("C|", output)
        self.assertIn("short_notes_removed=0", report)

    def test_melody_cleanup_applies_tempo_changes_at_note_position(self):
        source = "K:C\nL:1/8\nQ:1/8=120\nV:Vocal\nC[Q:1/8=600]D|\n"
        output, report = self.mod.MelodyCleanup().clean(
            source, "custom", 300, False, 24,
        )
        self.assertIn("C[Q:1/8=600]z|", output)
        self.assertIn("short_notes_removed=1", report)


if __name__ == "__main__":
    unittest.main()

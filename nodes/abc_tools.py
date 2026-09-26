"""Dependency-free ABC utilities tailored to SheetSage2/YuE2 scores."""
from __future__ import annotations

import io
import math
import os
import re

from .utils import safe_stem, timestamp_dir

NOTE_RE = re.compile(r"(?P<acc>\^{1,2}|_{1,2}|=)?(?P<letter>[A-Ga-g])(?P<oct>[,']*)(?P<dur>\d+(?:/\d+)?|/\d+|/+)?(?P<tie>-)?")
# Inline fields and decorations are syntax, not pitch-bearing ABC tokens.
PROTECTED_MUSIC_RE = re.compile(r"\[[A-Za-z]:[^\]]*\]|![^!]*!|\+[^+]*\+|%.*$")
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_NAMES = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]
FLAT_NAMES = ["C", "_D", "D", "_E", "E", "F", "_G", "G", "_A", "A", "_B", "B"]
KEY_PC = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,
          "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
FIFTHS_MAJOR = {"C":0,"G":1,"D":2,"A":3,"E":4,"B":5,"F#":6,"C#":7,
                "F":-1,"Bb":-2,"Eb":-3,"Ab":-4,"Db":-5,"Gb":-6,"Cb":-7}
FIFTHS_MINOR = {"A":0,"E":1,"B":2,"F#":3,"C#":4,"G#":5,"D#":6,"A#":7,
                "D":-1,"G":-2,"C":-3,"F":-4,"Bb":-5,"Eb":-6,"Ab":-7}


def _key_parts(value):
    m = re.match(r"\s*([A-Ga-g])([#b]?)(.*)", value or "C")
    if not m: return "C", "", False
    root = m.group(1).upper() + m.group(2)
    suffix = m.group(3)
    minor = suffix.strip().lower().startswith("m") and not suffix.strip().lower().startswith("mix")
    return root, suffix, minor


def _key_accidentals(value):
    root, _, minor = _key_parts(value)
    fifths = (FIFTHS_MINOR if minor else FIFTHS_MAJOR).get(root, 0)
    out = {x: 0 for x in "ABCDEFG"}
    for x in "FCGDAEB"[:max(fifths, 0)]: out[x] = 1
    for x in "BEADGCF"[:max(-fifths, 0)]: out[x] = -1
    return out


def _pitch(match, key_acc, state):
    letter, octs, acc = match.group("letter"), match.group("oct"), match.group("acc")
    key = _note_state_key(match)
    octave = key[1]
    if acc:
        delta = {"=":0,"^":1,"^^":2,"_":-1,"__":-2}[acc]
        state[key] = delta
    else:
        delta = state.get(key, key_acc.get(letter.upper(), 0))
    return 12 * (octave + 1) + PC[letter.upper()] + delta


def _note_state_key(match):
    letter, octs = match.group("letter"), match.group("oct")
    octave = 5 if letter.islower() else 4
    octave += octs.count("'") - octs.count(",")
    return letter.upper(), octave


def _note_name(midi):
    return f"{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][midi % 12]}{midi // 12 - 1}"


def _abc_note(midi, duration="", tie="", prefer_flats=False):
    names = FLAT_NAMES if prefer_flats else SHARP_NAMES
    name = names[midi % 12]
    octave = midi // 12 - 1
    accidental, letter = name[:-1], name[-1]
    if octave >= 5:
        letter = letter.lower(); marks = "'" * (octave - 5)
    else:
        marks = "," * (4 - octave)
    return accidental + letter + marks + (duration or "") + (tie or "")


def _prefer_flats(value):
    root, _, minor = _key_parts(value)
    fifths = (FIFTHS_MINOR if minor else FIFTHS_MAJOR).get(root)
    return fifths < 0 if fifths is not None else "b" in root


def _abc_note_keyaware(midi, key_acc, state, duration="", tie="", prefer_flats=False):
    """Serialize a pitch relative to the destination key and measure state."""
    names = FLAT_NAMES if prefer_flats else SHARP_NAMES
    name = names[midi % 12]
    octave = midi // 12 - 1
    accidental, letter = name[:-1], name[-1]
    desired = {"": 0, "^": 1, "_": -1}.get(accidental)
    if desired is None:
        desired = accidental.count("^") - accidental.count("_")
    state_key = (letter.upper(), octave)
    effective = state.get(state_key, key_acc.get(letter.upper(), 0))
    out_acc = ""
    if desired != effective:
        out_acc = "=" if desired == 0 else ("^" * desired if desired > 0 else "_" * -desired)
        state[state_key] = desired
    out_letter = letter.lower() if octave >= 5 else letter.upper()
    marks = "'" * max(0, octave - 5) if octave >= 5 else "," * max(0, 4 - octave)
    return out_acc + out_letter + marks + (duration or "") + (tie or "")


def _transpose_key(value, semitones):
    root, suffix, _ = _key_parts(value)
    pc = (KEY_PC.get(root, 0) + semitones) % 12
    prefer_flats = "b" in root
    names = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"] if prefer_flats else \
            ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return names[pc] + suffix


def _transpose_chords(text, semitones):
    def pitch_class(root):
        m = re.fullmatch(r"([A-G])((?:#{1,2}|b{1,2})?)", root)
        if not m:
            return None
        return (PC[m.group(1)] + m.group(2).count("#") - m.group(2).count("b")) % 12

    sharp_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    flat_names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    def transpose_root(root, prefer_flats):
        pc = pitch_class(root)
        if pc is None:
            return root
        return (flat_names if prefer_flats else sharp_names)[(pc + semitones) % 12]

    def transpose_body(body):
        root_match = re.match(r"^([A-G](?:#{1,2}|b{1,2})?)", body)
        if not root_match:
            return body
        root = root_match.group(1)
        prefer_flats = "b" in root or ("#" not in root and semitones < 0)
        rest = body[root_match.end():]
        result = transpose_root(root, prefer_flats) + rest
        # Numeric slash extensions (e.g. C6/9) are not bass notes.
        return re.sub(r"/([A-G](?:#{1,2}|b{1,2})?)",
                      lambda m: "/" + transpose_root(m.group(1), prefer_flats), result)

    return re.sub(r'"([^"]*)"', lambda m: '"' + transpose_body(m.group(1)) + '"', text)


def _rewrite_tempo_value(value, mode, override_bpm, multiplier):
    """Rewrite a numeric ABC Q value while preserving its beat unit and spacing."""
    match=re.search(r"(?P<unit>\d+\s*/\s*\d+)(?P<separator>\s*=\s*)(?P<bpm>\d+(?:\.\d+)?)",value)
    if match:
        source_bpm=float(match.group("bpm"))
        next_bpm=float(override_bpm) if mode=="override" else source_bpm*float(multiplier)
        return (value[:match.start()]+match.group("unit")+match.group("separator")+
                f"{next_bpm:g}"+value[match.end():])
    match=re.search(r"(?P<prefix>.*?)(?P<bpm>\d+(?:\.\d+)?)(?P<suffix>\s*)$",value)
    if match:
        source_bpm=float(match.group("bpm"))
        next_bpm=float(override_bpm) if mode=="override" else source_bpm*float(multiplier)
        return match.group("prefix")+f"{next_bpm:g}"+match.group("suffix")
    return f"1/4={float(override_bpm):g}" if mode=="override" else value


def _music_segments(line):
    """Yield quoted/nonquoted segments so chord text is never parsed as notes."""
    # ABC comments may contain arbitrary note-like text. Split them before
    # looking for quoted chord symbols so even quotes in comments stay intact.
    quoted = False
    comment_at = len(line)
    for index, char in enumerate(line):
        if char == '"':
            quoted = not quoted
        elif char == "%" and not quoted:
            comment_at = index
            break
    music, comment = line[:comment_at], line[comment_at:]
    pos=0
    for m in re.finditer(r'"[^"]*"', music):
        yield False, music[pos:m.start()]; yield True, m.group(0); pos=m.end()
    yield False, music[pos:]
    if comment:
        yield False, comment


def _note_segments(text, preserve_unsupported_groups=False):
    pattern=PROTECTED_MUSIC_RE
    if preserve_unsupported_groups:
        pattern=re.compile(PROTECTED_MUSIC_RE.pattern+r"|\[[A-Ga-g,\^_=0-9/ '\-]+\](?:\d*(?:/\d+|/+))?|\{[^}]*\}")
    pos = 0
    for match in pattern.finditer(text):
        yield False, text[pos:match.start()]
        yield True, match.group(0)
        pos = match.end()
    yield False, text[pos:]


_BAR_RE = re.compile(r"\|\|?|\|\]|\[\||::|:|\|:")


def _iter_note_parts(text):
    """Yield measure-local music chunks and barlines in source order."""
    pos = 0
    for match in _BAR_RE.finditer(text):
        if pos < match.start():
            yield False, text[pos:match.start()]
        yield True, match.group(0)
        pos = match.end()
    if pos < len(text):
        yield False, text[pos:]


def score_notes(text):
    key="C"; voice="default"; states={}; result=[]; section="unsectioned"
    for raw in (text or "").splitlines():
        line=raw.strip()
        if line.startswith("K:"):
            key=line[2:].strip(); states.clear(); continue
        if line.startswith("V:"): voice=line[2:].strip().split()[0]; continue
        if line.startswith("%"): section=line[1:].strip().lower() or section; continue
        if re.match(r"^[A-Za-z]:", line): continue
        state=states.setdefault(voice,{})
        for quoted, seg in _music_segments(raw):
            if quoted: continue
            for protected, chunk in _note_segments(seg):
                if protected:
                    km = re.match(r"\[K:([^\]]+)\]", chunk)
                    vm = re.match(r"\[V:([^\]]+)\]", chunk)
                    if km:
                        key = km.group(1).strip(); states.clear(); state=states.setdefault(voice,{})
                    elif vm:
                        voice = vm.group(1).strip().split()[0]
                        state = states.setdefault(voice,{})
                    continue
                for is_bar, part in _iter_note_parts(chunk):
                    if is_bar:
                        state.clear(); continue
                    for m in NOTE_RE.finditer(part):
                        result.append((voice, section, _pitch(m,_key_accidentals(key),state)))
    return result


def analyze_abc(text):
    if not (text or "").strip(): raise ValueError("ABC input is empty")
    def header(name, default=""):
        m=re.search(rf"(?m)^{name}:\s*(.+)$",text); return m.group(1).strip() if m else default
    meter=header("M","4/4"); key=header("K","C")
    def fraction(value, fallback):
        match=re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*",value or "")
        try:
            numerator,denominator=map(int,match.groups()) if match else fallback
            return numerator/denominator if numerator > 0 and denominator > 0 else fallback[0]/fallback[1]
        except (TypeError,ValueError,ZeroDivisionError):
            return fallback[0]/fallback[1]

    meter_length=fraction(meter,(4,4))
    default_length=fraction(header("L","1/8"),(1,8))
    tempo_unit=1/4
    bpm=120.

    def parse_tempo(value, current_bpm, current_unit):
        match=re.search(r"(\d+\s*/\s*\d+)\s*=\s*(\d+(?:\.\d+)?)",value)
        if match:
            return float(match.group(2)),fraction(match.group(1),(1,4))
        match=re.search(r"=\s*(\d+(?:\.\d+)?)|(?<!\S)(\d+(?:\.\d+)?)\s*$",value)
        if match:
            return float(next(group for group in match.groups() if group)),current_unit
        return current_bpm,current_unit

    # Build one independent rhythmic timeline per voice. Barlines are separators,
    # not events; uppercase Z is a measure rest, unlike ordinary lowercase z.
    voice_totals={}
    voice_measure_totals={}
    voice="default"
    def add_duration(current_voice, whole_notes):
        voice_totals[current_voice]=voice_totals.get(current_voice,0.0)+whole_notes*(60/max(bpm,1e-6)/max(tempo_unit,1e-9))
        voice_measure_totals[current_voice]=voice_measure_totals.get(current_voice,0.0)+whole_notes/max(meter_length,1e-9)

    def add_music_duration(music, current_voice):
        # Quoted chord symbols, comments, decorations, and bracketed inline fields
        # do not contribute note events. Inline V fields are handled by the caller.
        music=re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"'," ",music)
        music=re.sub(r"%.*$","",music)
        music=re.sub(r"![^!]*!|\+[^+]*\+"," ",music)
        music=re.sub(r"\[[^\]]*\]"," ",music)
        i=0
        while i < len(music):
            char=music[i]
            if char == "Z":
                match=re.match(r"Z(\d*)",music[i:])
                count=int(match.group(1) or "1")
                add_duration(current_voice,count*meter_length)
                i += len(match.group(0)); continue
            if char == "z":
                match=re.match(r"z(\d*(?:/\d+|/+)?)",music[i:])
                suffix=match.group(1)
                add_duration(current_voice,default_length*length_multiplier(suffix))
                i += len(match.group(0)); continue
            # Skip accidental prefix, then parse a note and its ABC length suffix.
            start=i
            while i < len(music) and music[i] in "^_=": i+=1
            if i < len(music) and music[i] in "ABCDEFGabcdefg":
                i += 1
                while i < len(music) and music[i] in "',": i+=1
                suffix_match=re.match(r"(\d*(?:/\d+|/+)?)",music[i:])
                suffix=suffix_match.group(1) if suffix_match else ""
                add_duration(current_voice,default_length*length_multiplier(suffix))
                i += len(suffix)
            else:
                i=start+1

    def length_multiplier(suffix):
        if not suffix: return 1.0
        if "/" not in suffix: return float(suffix)
        numerator,slash_denominator=suffix.split("/",1)
        numerator=float(numerator or 1)
        denominator=float(slash_denominator) if slash_denominator.isdigit() else float(2**(1+len(slash_denominator)))
        return numerator/denominator

    for line in text.splitlines():
        header_match=re.match(r"^([A-Za-z]):\s*(.*)$",line)
        if header_match:
            field,value=header_match.groups()
            if field == "V": voice=value.strip().split()[0] if value.strip() else "default"
            elif field == "M": meter_length=fraction(value,(4,4))
            elif field == "L": default_length=fraction(value,(1,8))
            elif field == "Q":
                bpm,tempo_unit=parse_tempo(value,bpm,tempo_unit)
            continue
        # Inline timing and voice fields take effect at their position in the line.
        position=0
        for match in re.finditer(r"\[([A-Za-z]):([^\]]*)\]",line):
            add_music_duration(line[position:match.start()],voice)
            field,value=match.groups()
            if field == "V": voice=value.strip().split()[0] if value.strip() else "default"
            elif field == "M": meter_length=fraction(value,(4,4))
            elif field == "L": default_length=fraction(value,(1,8))
            elif field == "Q": bpm,tempo_unit=parse_tempo(value,bpm,tempo_unit)
            position=match.end()
        add_music_duration(line[position:],voice)

    duration=max(voice_totals.values(),default=0.0)
    bars=max(voice_measure_totals.values(),default=0.0)
    notes=score_notes(text); by={}
    for voice,section,pitch in notes: by.setdefault(voice,[]).append(pitch)
    lines=[f"BPM: {bpm:g}",f"Key: {key}",f"Meter: {meter}",f"Measures: ~{bars}",f"Duration: ~{duration:.1f} s"]
    for voice,pitches in sorted(by.items()):
        pitches.sort(); median=pitches[len(pitches)//2]
        lines.append(f"{voice}: notes={len(pitches)} range={_note_name(pitches[0])}..{_note_name(pitches[-1])} median={_note_name(median)}")
        if "vocal" in voice.lower() and median < 57: lines.append("WARNING: Vocal center is low; consider an octave-up test.")
        if "vocal" in voice.lower() and median > 76: lines.append("WARNING: Vocal center is high; consider an octave-down test.")
    return "\n".join(lines), bpm, key, duration


class ABCAnalyzer:
    DESCRIPTION="Reports tempo, key, meter, duration, voices, note counts, and vocal/instrument ranges."
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"abc":("STRING",{"multiline":True})}}
    RETURN_TYPES=("STRING","FLOAT","STRING","FLOAT"); RETURN_NAMES=("report","bpm","key","duration_seconds")
    FUNCTION="analyze"; CATEGORY="YuE2/ABC"
    def analyze(self,abc):
        if not (abc or "").strip():
            return "ABC input is empty (score planning is disabled).",0.0,"",0.0
        report,bpm,key,duration=analyze_abc(abc); return report,bpm,key,duration


NATIVE_DURATION_MIN_SECONDS = 0.04
NATIVE_DURATION_MAX_SECONDS = 900.0
LEGACY_SEMANTIC_MIN_TOKENS = 200
LEGACY_SEMANTIC_MAX_TOKENS = 16384
LEGACY_SEMANTIC_TOKENS_PER_SECOND = 25


def calculate_duration_budget(abc, headroom_percent=10.0, tail_seconds=2.0):
    """Calculate backend hard ceilings from the score's rhythmic duration."""
    score_seconds=analyze_abc(abc)[3]
    recommended=score_seconds*(1+float(headroom_percent)/100)+float(tail_seconds)
    native=min(NATIVE_DURATION_MAX_SECONDS,max(NATIVE_DURATION_MIN_SECONDS,recommended))
    legacy_raw=math.ceil(recommended*LEGACY_SEMANTIC_TOKENS_PER_SECOND)
    legacy=min(LEGACY_SEMANTIC_MAX_TOKENS,max(LEGACY_SEMANTIC_MIN_TOKENS,legacy_raw))
    warnings=[]
    if native != recommended:
        warnings.append(f"Native hard ceiling clamped from {recommended:.2f}s to {native:.2f}s.")
    if legacy != legacy_raw:
        warnings.append(f"Legacy semantic max_tokens clamped from {legacy_raw} to {legacy}.")
    report=(f"Score duration: {score_seconds:.2f}s. Recommended hard ceiling/headroom: "
            f"{recommended:.2f}s (headroom={float(headroom_percent):g}%, tail={float(tail_seconds):g}s). "
            f"Native max_duration={native:.2f}s; Legacy semantic_max_tokens={legacy} "
            f"at {LEGACY_SEMANTIC_TOKENS_PER_SECOND} tokens/s.")
    if warnings:
        report += " WARNING: " + " ".join(warnings)
    report += " These are hard ceilings; generation may finish earlier."
    return score_seconds,native,legacy,report


class YuE2DurationBudget:
    DESCRIPTION=("Calculates Native max_duration and Legacy semantic_max_tokens from ABC duration "
                 "with configurable headroom and tail. Outputs are hard ceilings, not target lengths.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "abc":("STRING",{"forceInput":True,"tooltip":"ABC score used to calculate the duration budget."}),
            "headroom_percent":("FLOAT",{"default":10.0,"min":0.0,"max":100.0,"step":0.5}),
            "tail_seconds":("FLOAT",{"default":2.0,"min":0.0,"max":60.0,"step":0.5}),
        }}

    RETURN_TYPES=("FLOAT","FLOAT","INT","STRING")
    RETURN_NAMES=("score_duration_seconds","native_max_duration","legacy_semantic_max_tokens","report")
    FUNCTION="calculate"
    CATEGORY="YuE2/ABC"

    def calculate(self,abc,headroom_percent=10.0,tail_seconds=2.0):
        return calculate_duration_budget(abc,headroom_percent,tail_seconds)


class ABCModifier:
    DESCRIPTION="Changes score tempo, transposes all or selected SheetSage2 voices, removes chords, and drops named sections."
    @classmethod
    def INPUT_TYPES(cls): return {"required":{
        "abc":("STRING",{"multiline":True}), "tempo_mode":(["keep","override","multiplier"],),
        "bpm":("FLOAT",{"default":120.,"min":20.,"max":400.,"step":.1}),
        "tempo_multiplier":("FLOAT",{"default":1.,"min":.25,"max":4.,"step":.01}),
        "transpose_scope":(["whole_score","vocal_only","instrumental_only","none"],),
        "semitones":("INT",{"default":0,"min":-36,"max":36}),
        "remove_chords":("BOOLEAN",{"default":False}),
        "drop_sections":("STRING",{"default":"","tooltip":"Comma-separated section comments to remove, for example intro,outro."})},
        "optional": {
        "vocal_transpose":("INT",{"default":0,"min":-36,"max":36,"tooltip":"Additional semitone shift applied only to a voice whose name contains Vocal."}),
        "instrumental_transpose":("INT",{"default":0,"min":-36,"max":36,"tooltip":"Additional shift applied only to Ins/Instrumental voices."}),
        "octave_shift":("INT",{"default":0,"min":-3,"max":3,"tooltip":"Global octave shift, added to the selected transpose."}),
        }}
    RETURN_TYPES=("STRING","STRING"); RETURN_NAMES=("abc","report"); FUNCTION="modify"; CATEGORY="YuE2/ABC"
    def modify(self,abc,tempo_mode,bpm,tempo_multiplier,transpose_scope,semitones,remove_chords,drop_sections,
               vocal_transpose=0,instrumental_transpose=0,octave_shift=0):
        if not abc.strip(): raise ValueError("ABC input is empty")
        old_report,old_bpm,key,_=analyze_abc(abc); target=old_bpm
        if tempo_mode=="override": target=bpm
        elif tempo_mode=="multiplier": target=old_bpm*tempo_multiplier
        drop={x.strip().lower() for x in drop_sections.split(",") if x.strip()}
        is_noop = (tempo_mode == "keep" and not drop and not remove_chords and not semitones
                   and not vocal_transpose and not instrumental_transpose and not octave_shift)
        if is_noop:
            result = abc
        else:
            lines=[]; voice="default"; section=""; skipping=False
            source_key=key
            output_key=_transpose_key(key,semitones) if transpose_scope=="whole_score" and semitones else key
            source_states={}; dest_states={}; has_tempo=False
            for physical in abc.splitlines(keepends=True):
                ending = "\r\n" if physical.endswith("\r\n") else (physical[-1:] if physical.endswith(("\n", "\r")) else "")
                raw = physical[:-len(ending)] if ending else physical
                if raw.startswith("%"):
                    section=raw[1:].strip().lower(); skipping=section in drop
                    if skipping: continue
                if skipping: continue
                if raw.startswith("Q:"):
                    has_tempo=True
                    if tempo_mode!="keep":
                        field=re.match(r"^(Q:\s*)(.*)$",raw)
                        raw=field.group(1)+_rewrite_tempo_value(
                            field.group(2),tempo_mode,bpm,tempo_multiplier)
                if raw.startswith("V:"):
                    voice=raw[2:].strip().split()[0]
                if raw.startswith("K:"):
                    field = re.match(r"^(K:\s*)(.*?)(\s*)$", raw)
                    source_key = field.group(2)
                    output_key = (_transpose_key(source_key.strip(), semitones)
                                  if transpose_scope=="whole_score" and semitones else source_key.strip())
                    source_states.clear(); dest_states.clear()
                    if output_key != source_key.strip():
                        raw = field.group(1) + output_key + field.group(3)
                    lines.append(raw + ending)
                    continue
                if re.match(r"^[A-Za-z]:",raw) or raw.startswith("%"):
                    lines.append(raw + ending); continue

                pieces=[]
                for quoted,seg in _music_segments(raw):
                    if quoted:
                        if remove_chords:
                            pieces.append("")
                        elif transpose_scope=="whole_score" and semitones:
                            pieces.append(_transpose_chords(seg,semitones))
                        else:
                            pieces.append(seg)
                        continue

                    transformed=[]
                    for protected,chunk in _note_segments(seg):
                        if protected:
                            km=re.match(r"\[K:(?P<before>\s*)(?P<key>[^\]]*?)(?P<after>\s*)\]",chunk)
                            vm=re.match(r"\[V:([^\]]+)\]",chunk)
                            qm=re.match(r"\[Q:(.*)\]",chunk)
                            if km:
                                source_key=km.group("key").strip()
                                output_key=(_transpose_key(source_key,semitones)
                                            if transpose_scope=="whole_score" and semitones else source_key)
                                source_states.clear(); dest_states.clear()
                                new_chunk=(f"[K:{km.group('before')}{output_key}{km.group('after')}]"
                                           if output_key != source_key else chunk)
                                transformed.append(new_chunk)
                            elif qm:
                                has_tempo=True
                                if tempo_mode!="keep":
                                    qvalue=_rewrite_tempo_value(qm.group(1),tempo_mode,bpm,tempo_multiplier)
                                    transformed.append(f"[Q:{qvalue}]")
                                else:
                                    transformed.append(chunk)
                            else:
                                transformed.append(chunk)
                                if vm:
                                    voice=vm.group(1).strip().split()[0]
                            continue

                        selected=(transpose_scope=="whole_score" or
                                  transpose_scope=="vocal_only" and "vocal" in voice.lower() or
                                  transpose_scope=="instrumental_only" and ("ins" in voice.lower() or "inst" in voice.lower()))
                        offset=(semitones if selected else 0) + 12*octave_shift
                        if "vocal" in voice.lower(): offset += vocal_transpose
                        elif "ins" in voice.lower() or "inst" in voice.lower(): offset += instrumental_transpose
                        source_state=source_states.setdefault(voice,{})
                        dest_state=dest_states.setdefault(voice,{})
                        for is_bar,part in _iter_note_parts(chunk):
                            if is_bar:
                                source_state.clear(); dest_state.clear(); transformed.append(part); continue
                            out=[]; pos=0
                            source_keyacc=_key_accidentals(source_key)
                            dest_keyacc=_key_accidentals(output_key)
                            for match in NOTE_RE.finditer(part):
                                out.append(part[pos:match.start()])
                                source_pitch=_pitch(match,source_keyacc,source_state)
                                target_pitch=source_pitch+offset
                                if offset:
                                    out.append(_abc_note_keyaware(target_pitch,dest_keyacc,dest_state,
                                        match.group("dur"),match.group("tie"),_prefer_flats(output_key)))
                                else:
                                    out.append(match.group(0))
                                    if match.group("acc"):
                                        state_key = _note_state_key(match)
                                        dest_state[state_key] = source_state[state_key]
                                pos=match.end()
                            out.append(part[pos:]); transformed.append("".join(out))
                    pieces.append("".join(transformed))
                lines.append("".join(pieces)+ending)
            if tempo_mode!="keep" and not has_tempo:
                insert=next((i+1 for i,line in enumerate(lines) if line.startswith("L:")),min(3,len(lines)))
                newline="\r\n" if "\r\n" in abc else "\n"
                lines.insert(insert,f"Q:1/4={target:g}{newline}")
            result="".join(lines)
        report=(f"bpm {old_bpm:g}->{target:g}; transpose={semitones} scope={transpose_scope}; "
                f"vocal={vocal_transpose:+d}; instrumental={instrumental_transpose:+d}; "
                f"octaves={octave_shift:+d}; removed_sections={sorted(drop)}")
        if (transpose_scope in {"vocal_only","instrumental_only"} and semitones%12) or vocal_transpose%12 or instrumental_transpose%12:
            report += " | warning: non-octave voice-only transposition changes its harmonic relationship"
        return result,report


class VocalRangeRetarget:
    DESCRIPTION=("Shifts only Vocal ABC pitches toward a selected range. These choices are pitch ranges only. "
                 "They do not select vocal gender or timbre. Set female/male/soprano/etc. character in the YuE2 style prompt.")
    RANGES={"female_contralto":(52,77),"female_alto":(53,77),"female_mezzo":(57,81),
            "female_soprano":(60,84),"male_bass":(40,64),"male_baritone":(45,69),"male_tenor":(48,72)}
    @classmethod
    def INPUT_TYPES(cls): return {"required":{
        "abc":("STRING",{"multiline":True}),
        "target_voice":(["original",*cls.RANGES.keys()],{"tooltip":
            "Pitch range only; this does not select vocal gender, timbre, or singer identity. Describe those in the YuE2 style prompt."}),
        "mode":(["nearest_key_safe","nearest_octave","manual"],{"tooltip":
            "nearest_key_safe maximizes notes inside the range; nearest_octave moves the median toward its center without reducing in-range coverage. Automatic modes use octave shifts only."}),
        "manual_semitones":("INT",{"default":0,"min":-36,"max":36})}}
    RETURN_TYPES=("STRING","INT","STRING"); RETURN_NAMES=("abc","applied_semitones","report")
    FUNCTION="retarget"; CATEGORY="YuE2/ABC"
    def retarget(self,abc,target_voice,mode,manual_semitones):
        pitches=[p for v,_s,p in score_notes(abc) if "vocal" in v.lower()]
        if not pitches: raise ValueError("No Vocal voice was found in the ABC score")
        pitches.sort(); median=pitches[len(pitches)//2]
        if target_voice=="original": shift=0
        elif mode=="manual": shift=int(manual_semitones)
        else:
            lo,hi=self.RANGES[target_voice]; center=(lo+hi)/2
            candidates=(-36,-24,-12,0,12,24,36)
            def coverage(candidate):
                shifted=[pitch+candidate for pitch in pitches]
                inside=sum(lo <= pitch <= hi for pitch in shifted)
                overflow=sum(max(lo-pitch,0)+max(pitch-hi,0) for pitch in shifted)
                shifted_median=median+candidate
                return inside,overflow,abs(candidate),abs(shifted_median-center)
            baseline=coverage(0)
            if baseline[0] == len(pitches):
                shift=0
            elif mode=="nearest_key_safe":
                shift=min(candidates,key=lambda candidate:(
                    -coverage(candidate)[0],coverage(candidate)[1],
                    coverage(candidate)[2],coverage(candidate)[3]))
            else:
                # nearest_octave prioritizes placing the median near range center,
                # but refuses any shift that reduces in-range note coverage.
                preferred=min(candidates,key=lambda candidate:(
                    coverage(candidate)[3],coverage(candidate)[1],
                    coverage(candidate)[2]))
                shift=preferred if coverage(preferred)[0] >= baseline[0] else 0
        out,mod_report=ABCModifier().modify(abc,"keep",120,1,"none",0,False,"",
                                             vocal_transpose=shift)
        result_pitches=[pitch+shift for pitch in pitches]
        result_median=result_pitches[len(result_pitches)//2]
        if target_voice=="original":
            target_range=f"{_note_name(pitches[0])}..{_note_name(pitches[-1])}"
        else:
            lo,hi=self.RANGES[target_voice]
            target_range=f"{_note_name(lo)}..{_note_name(hi)}"
        report=(f"target={target_voice} range={target_range}; "
                f"source={_note_name(pitches[0])}..{_note_name(pitches[-1])} median={_note_name(median)}; "
                f"result={_note_name(result_pitches[0])}..{_note_name(result_pitches[-1])} "
                f"median={_note_name(result_median)}; applied={shift:+d} semitones. "
                "Pitch range only; vocal gender/timbre is controlled by the style prompt. "
                "For a female soprano timbre, also describe the voice in the YuE2 style prompt, "
                "for example: female soprano vocal, light feminine timbre, clear head voice.")
        return out,shift,report


class MelodyCleanup:
    DESCRIPTION="Conservatively removes very short or extreme Vocal notes from ABC by replacing them with equal-duration rests."
    @classmethod
    def INPUT_TYPES(cls): return {"required":{
        "abc":("STRING",{"multiline":True}), "preset":(["off","light","medium","custom"],),
        "remove_notes_shorter_than_ms":("FLOAT",{"default":40.,"min":0.,"max":500.,"step":5.}),
        "remove_pitch_outliers":("BOOLEAN",{"default":True}),
        "outlier_semitones":("INT",{"default":24,"min":6,"max":48})}}
    RETURN_TYPES=("STRING","STRING"); RETURN_NAMES=("abc","report")
    FUNCTION="clean"; CATEGORY="YuE2/ABC"
    def clean(self,abc,preset,remove_notes_shorter_than_ms,remove_pitch_outliers,outlier_semitones):
        if preset=="off": return abc,"cleanup disabled"
        if preset=="light": threshold,outliers=30.,False
        elif preset=="medium": threshold,outliers=50.,True
        else: threshold,outliers=remove_notes_shorter_than_ms,remove_pitch_outliers
        pitches=sorted(p for v,_s,p in score_notes(abc) if "vocal" in v.lower())
        median=pitches[len(pitches)//2] if pitches else 60
        def fraction(value, fallback):
            match=re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*",value or "")
            try:
                numerator,denominator=map(int,match.groups()) if match else fallback
                return numerator/denominator if numerator>0 and denominator>0 else fallback[0]/fallback[1]
            except (TypeError,ValueError,ZeroDivisionError):
                return fallback[0]/fallback[1]

        def tempo(value,current_bpm,current_unit):
            match=re.search(r"(\d+\s*/\s*\d+)\s*=\s*(\d+(?:\.\d+)?)",value)
            if match:
                return float(match.group(2)),fraction(match.group(1),(1,4))
            match=re.search(r"=\s*(\d+(?:\.\d+)?)|(?<!\S)(\d+(?:\.\d+)?)\s*$",value)
            if match: return float(next(x for x in match.groups() if x)),current_unit
            return current_bpm,current_unit

        def length_multiplier(suffix):
            if not suffix: return 1.0
            if "/" not in suffix: return float(suffix)
            numerator,denominator_text=suffix.split("/",1)
            numerator=float(numerator or 1)
            if denominator_text.isdigit(): denominator=float(denominator_text)
            else: denominator=float(2**(1+len(denominator_text)))
            return numerator/denominator

        meter_length=1.0
        default_length=1/8
        bpm=120.; tempo_unit=1/4
        voice="default"; key="C"; source_states={}; dest_states={}
        removed_short=removed_outlier=0; lines=[]

        for physical in abc.splitlines(keepends=True):
            ending="\r\n" if physical.endswith("\r\n") else (physical[-1:] if physical.endswith(("\n","\r")) else "")
            raw=physical[:-len(ending)] if ending else physical
            field_match=re.match(r"^([A-Za-z]):\s*(.*)$",raw)
            if field_match:
                field,value=field_match.groups()
                if field=="V": voice=value.strip().split()[0] if value.strip() else "default"
                elif field=="K":
                    key=value.strip(); source_states.clear(); dest_states.clear()
                elif field=="M": meter_length=fraction(value,(4,4))
                elif field=="L": default_length=fraction(value,(1,8))
                elif field=="Q": bpm,tempo_unit=tempo(value,bpm,tempo_unit)
                lines.append(physical); continue

            pieces=[]
            for quoted,segment in _music_segments(raw):
                if quoted:
                    pieces.append(segment); continue
                transformed=[]
                for protected,chunk in _note_segments(segment,preserve_unsupported_groups=True):
                    if protected:
                        transformed.append(chunk)
                        field=re.match(r"\[([A-Za-z]):([^\]]*)\]",chunk)
                        if field:
                            name,value=field.groups()
                            if name=="V": voice=value.strip().split()[0] if value.strip() else "default"
                            elif name=="K":
                                key=value.strip(); source_states.clear(); dest_states.clear()
                            elif name=="M": meter_length=fraction(value,(4,4))
                            elif name=="L": default_length=fraction(value,(1,8))
                            elif name=="Q": bpm,tempo_unit=tempo(value,bpm,tempo_unit)
                        continue

                    source_state=source_states.setdefault(voice,{})
                    dest_state=dest_states.setdefault(voice,{})
                    for is_bar,part in _iter_note_parts(chunk):
                        if is_bar:
                            source_state.clear(); dest_state.clear(); transformed.append(part); continue
                        out=[]; position=0
                        key_acc=_key_accidentals(key)
                        for match in NOTE_RE.finditer(part):
                            out.append(part[position:match.start()])
                            if "vocal" not in voice.lower():
                                out.append(match.group(0)); position=match.end(); continue

                            source_pitch=_pitch(match,key_acc,source_state)
                            duration=match.group("dur") or ""
                            seconds=(default_length*length_multiplier(duration)*
                                     60/max(bpm,1e-9)/max(tempo_unit,1e-9))
                            short=seconds*1000 < threshold
                            extreme=outliers and abs(source_pitch-median)>outlier_semitones
                            if short or extreme:
                                removed_short+=int(short)
                                removed_outlier+=int(extreme and not short)
                                out.append("z"+duration)
                            else:
                                state_key=_note_state_key(match)
                                base_pitch=12*(state_key[1]+1)+PC[match.group("letter").upper()]
                                if match.group("acc"):
                                    # Retain explicit source spelling and carry its accidental
                                    # into the destination state only for notes that remain.
                                    out.append(match.group(0))
                                    dest_state[state_key]=source_pitch-base_pitch
                                else:
                                    dest_pitch=base_pitch+dest_state.get(
                                        state_key,key_acc.get(match.group("letter").upper(),0))
                                    if dest_pitch==source_pitch:
                                        out.append(match.group(0))
                                    else:
                                        out.append(_abc_note_keyaware(
                                            source_pitch,key_acc,dest_state,duration,
                                            match.group("tie") or "",_prefer_flats(key)))
                            position=match.end()
                        out.append(part[position:]); transformed.append("".join(out))
                pieces.append("".join(transformed))
            lines.append("".join(pieces)+ending)
        result="".join(lines)
        return result,f"preset={preset} short_notes_removed={removed_short} pitch_outliers_removed={removed_outlier}"


class ABCFileLoader:
    DESCRIPTION="Loads an ABC score from the ComfyUI input directory."
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        files=[]
        try: files=sorted(x for x in os.listdir(folder_paths.get_input_directory()) if x.lower().endswith(".abc"))
        except OSError: pass
        return {"required":{"abc_file":(files or ["Place an .abc file in the ComfyUI input folder"],)}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("abc",); FUNCTION="load"; CATEGORY="YuE2/ABC"
    def load(self,abc_file):
        import folder_paths
        path=abc_file if os.path.isabs(abc_file) else os.path.join(folder_paths.get_input_directory(),abc_file)
        if not os.path.isfile(path): raise ValueError(f"ABC file does not exist: {path}")
        return (io.open(path,encoding="utf-8",errors="replace").read(),)


class ABCFileSaver:
    DESCRIPTION="Saves ABC text under ComfyUI/output/YuE2_ABC."
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"abc":("STRING",{"multiline":True}),"filename":("STRING",{"default":"score"})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("path",); OUTPUT_NODE=True; FUNCTION="save"; CATEGORY="YuE2/ABC"
    def save(self,abc,filename):
        out=timestamp_dir("YuE2_ABC"); path=os.path.join(out,safe_stem(filename)+".abc")
        io.open(path,"w",encoding="utf-8").write(abc); return (path,)


class YuE2StyleBuilder:
    DESCRIPTION="Builds a compact YuE2 style prompt from structured musical descriptors."
    @classmethod
    def INPUT_TYPES(cls): return {"required":{
        "language":("STRING",{"default":"English"}), "genre":("STRING",{"default":"industrial breakbeat"}),
        "secondary_genres":("STRING",{"default":"big beat, rave punk"}), "era":("STRING",{"default":"1990s"}),
        "vocal":("STRING",{"default":"female, raspy, aggressive, shouted"}),
        "instruments":("STRING",{"default":"distorted bass, abrasive synths"}),
        "drums":("STRING",{"default":"hard syncopated breakbeats"}),
        "mood":("STRING",{"default":"raw, dark, hostile"}), "tempo":("STRING",{"default":"fast"}),
        "extra":("STRING",{"default":"","multiline":True})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("style",); FUNCTION="build"; CATEGORY="YuE2/Prompting"
    def build(self,**kw):
        parts=[]
        for value in kw.values():
            for x in str(value).split(","):
                x=x.strip()
                if x and x.lower() not in {p.lower() for p in parts}: parts.append(x)
        return (", ".join(parts),)


class LyricsMelodyFit:
    DESCRIPTION="Heuristically compares lyric syllable counts with Vocal ABC note counts per section."
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"abc":("STRING",{"multiline":True}),"lyrics":("STRING",{"multiline":True}),
                                                    "language":(["English","generic"],)}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("report",); FUNCTION="analyze"; CATEGORY="YuE2/ABC"
    def analyze(self,abc,lyrics,language):
        lyric_sections={}; sec="unsectioned"
        for line in lyrics.splitlines():
            m=re.match(r"\s*\[([^]]+)\]\s*$",line)
            if m: sec=m.group(1).lower(); lyric_sections.setdefault(sec,[])
            elif line.strip(): lyric_sections.setdefault(sec,[]).append(line.strip())
        note_sections={}
        for voice,section,p in score_notes(abc):
            if "vocal" in voice.lower(): note_sections[section]=note_sections.get(section,0)+1
        def syllables(s):
            if language!="English": return len(re.findall(r"\w+",s))
            total=0
            for word in re.findall(r"[A-Za-z]+",s.lower()):
                n=len(re.findall(r"[aeiouy]+",word)); n-=int(word.endswith("e") and n>1); total+=max(1,n)
            return total
        out=["Heuristic only: melisma and sustained notes can make a good fit differ from 1:1."]
        keys=list(dict.fromkeys([*lyric_sections,*note_sections]))
        for s in keys:
            sy=syllables(" ".join(lyric_sections.get(s,[]))); no=note_sections.get(s,0); ratio=sy/max(no,1)
            verdict="good" if .65<=ratio<=1.5 else "lyrics may be dense" if ratio>1.5 else "melody may require melisma/sustains"
            out.append(f"[{s}] vocal_notes={no} estimated_syllables={sy} ratio={ratio:.2f} | {verdict}")
        return ("\n".join(out),)


NODE_CLASS_MAPPINGS={"YuE2ABCAnalyzer":ABCAnalyzer,"YuE2ABCDurationBudget":YuE2DurationBudget,
 "YuE2ABCModifier":ABCModifier,
 "YuE2VocalRangeRetarget":VocalRangeRetarget,"YuE2MelodyCleanup":MelodyCleanup,
 "YuE2LoadABCFile":ABCFileLoader,"YuE2SaveABCFile":ABCFileSaver,
 "YuE2StyleBuilder":YuE2StyleBuilder,"YuE2LyricsMelodyFit":LyricsMelodyFit}
NODE_DISPLAY_NAME_MAPPINGS={"YuE2ABCAnalyzer":"YuE2 ABC Analyzer","YuE2ABCDurationBudget":"YuE2 ABC Duration Budget",
 "YuE2ABCModifier":"YuE2 ABC Modifier",
 "YuE2VocalRangeRetarget":"YuE2 Vocal Range Retarget","YuE2MelodyCleanup":"YuE2 Melody Cleanup",
 "YuE2LoadABCFile":"YuE2 ABC File Loader","YuE2SaveABCFile":"YuE2 ABC File Saver",
 "YuE2StyleBuilder":"YuE2 Style Prompt Builder","YuE2LyricsMelodyFit":"YuE2 Lyrics / Melody Fit Analyzer"}

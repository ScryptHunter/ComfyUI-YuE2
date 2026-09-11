"""Dependency-free ABC utilities tailored to SheetSage2/YuE2 scores."""
from __future__ import annotations

import io
import os
import re

from .utils import safe_stem, timestamp_dir

NOTE_RE = re.compile(r"(?<![A-Za-z])(?P<acc>\^{1,2}|_{1,2}|=)?(?P<letter>[A-Ga-g])(?P<oct>[,']*)(?P<dur>\d+(?:/\d+)?|/\d+|/)?(?P<tie>-)?")
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
    octave = 5 if letter.islower() else 4
    octave += octs.count("'") - octs.count(",")
    key = (letter.upper(), octave)
    if acc:
        delta = {"=":0,"^":1,"^^":2,"_":-1,"__":-2}[acc]
        state[key] = delta
    else:
        delta = state.get(key, key_acc.get(letter.upper(), 0))
    return 12 * (octave + 1) + PC[letter.upper()] + delta


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


def _transpose_key(value, semitones):
    root, suffix, _ = _key_parts(value)
    pc = (KEY_PC.get(root, 0) + semitones) % 12
    prefer_flats = "b" in root
    names = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"] if prefer_flats else \
            ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return names[pc] + suffix


def _transpose_chords(text, semitones):
    def chord(m):
        body=m.group(1)
        return '"' + re.sub(r"(?<![A-Za-z])([A-G])([#b]?)(?=[:A-Za-z0-9/(]|$)",
            lambda n: _transpose_key(n.group(1)+n.group(2), semitones), body) + '"'
    return re.sub(r'"([^"]*)"', chord, text)


def _music_segments(line):
    """Yield quoted/nonquoted segments so chord text is never parsed as notes."""
    pos=0
    for m in re.finditer(r'"[^"]*"', line):
        yield False, line[pos:m.start()]; yield True, m.group(0); pos=m.end()
    yield False, line[pos:]


def score_notes(text):
    key="C"; voice="default"; states={}; result=[]; section="unsectioned"
    for raw in (text or "").splitlines():
        line=raw.strip()
        if line.startswith("K:"): key=line[2:].strip(); continue
        if line.startswith("V:"): voice=line[2:].strip().split()[0]; continue
        if line.startswith("%"): section=line[1:].strip().lower() or section; continue
        if re.match(r"^[A-Za-z]:", line): continue
        state=states.setdefault(voice,{})
        for quoted, seg in _music_segments(raw):
            if quoted: continue
            parts=re.split(r"(\|+)",seg)
            for part in parts:
                if part.startswith("|"): state.clear(); continue
                for m in NOTE_RE.finditer(part):
                    result.append((voice, section, _pitch(m,_key_accidentals(key),state)))
    return result


def analyze_abc(text):
    if not (text or "").strip(): raise ValueError("ABC input is empty")
    def header(name, default=""):
        m=re.search(rf"(?m)^{name}:\s*(.+)$",text); return m.group(1).strip() if m else default
    meter=header("M","4/4"); key=header("K","C"); q=header("Q","")
    bpm_match=re.search(r"=\s*(\d+(?:\.\d+)?)",q)
    if not bpm_match: bpm_match=re.search(r"(\d+(?:\.\d+)?)\s*$",q)
    bpm=float(bpm_match.group(1)) if bpm_match else 120.
    try: num,den=map(int,meter.split("/",1))
    except Exception: num,den=4,4
    bars=sum(max(1,len(re.findall(r"\|",l))) for l in text.splitlines() if l and not re.match(r"^[A-Za-z%]:",l))
    # Voices repeat the same measure layout, so use the largest per-voice bar count.
    voice_bars={}; voice="default"
    for l in text.splitlines():
        if l.startswith("V:"): voice=l[2:].strip().split()[0]; continue
        if not re.match(r"^[A-Za-z%]:",l): voice_bars[voice]=voice_bars.get(voice,0)+l.count("|")
    bars=max(voice_bars.values(),default=bars)
    duration=bars*num*(4/den)*60/max(bpm,1e-6)
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
        report,bpm,key,duration=analyze_abc(abc); return report,bpm,key,duration


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
        lines=[]; voice="default"; section=""; skipping=False; original_key=key
        for raw in abc.splitlines():
            if raw.startswith("%"):
                section=raw[1:].strip().lower(); skipping=section in drop
                if skipping: continue
            if skipping: continue
            if raw.startswith("Q:") and tempo_mode!="keep": raw=f"Q:1/4={target:g}"
            if raw.startswith("V:"): voice=raw[2:].strip().split()[0]
            selected=(transpose_scope=="whole_score" or
                      transpose_scope=="vocal_only" and "vocal" in voice.lower() or
                      transpose_scope=="instrumental_only" and ("ins" in voice.lower() or "inst" in voice.lower()))
            offset=(semitones if selected else 0) + 12*octave_shift
            if "vocal" in voice.lower(): offset += vocal_transpose
            elif "ins" in voice.lower() or "inst" in voice.lower(): offset += instrumental_transpose
            if raw.startswith("K:") and transpose_scope=="whole_score" and semitones:
                raw="K:"+_transpose_key(raw[2:].strip(),semitones)
            elif not re.match(r"^[A-Za-z]:",raw) and not raw.startswith("%"):
                pieces=[]; state={}; keyacc=_key_accidentals(original_key)
                for quoted,seg in _music_segments(raw):
                    if quoted:
                        pieces.append("" if remove_chords else (_transpose_chords(seg,semitones) if transpose_scope=="whole_score" and semitones else seg)); continue
                    if offset:
                        out=[]; pos=0
                        for m in NOTE_RE.finditer(seg):
                            out.append(seg[pos:m.start()]); pitch=_pitch(m,keyacc,state)+offset
                            out.append(_abc_note(pitch,m.group("dur"),m.group("tie"),"b" in original_key)); pos=m.end()
                        out.append(seg[pos:]); seg="".join(out)
                    pieces.append(seg)
                raw="".join(pieces)
            lines.append(raw)
        if tempo_mode!="keep" and not any(l.startswith("Q:") for l in lines):
            insert=next((i+1 for i,l in enumerate(lines) if l.startswith("L:")),3); lines.insert(insert,f"Q:1/4={target:g}")
        result="\n".join(lines).rstrip()+"\n"
        report=(f"bpm {old_bpm:g}->{target:g}; transpose={semitones} scope={transpose_scope}; "
                f"vocal={vocal_transpose:+d}; instrumental={instrumental_transpose:+d}; "
                f"octaves={octave_shift:+d}; removed_sections={sorted(drop)}")
        if (transpose_scope in {"vocal_only","instrumental_only"} and semitones%12) or vocal_transpose%12 or instrumental_transpose%12:
            report += " | warning: non-octave voice-only transposition changes its harmonic relationship"
        return result,report


class VocalRangeRetarget:
    DESCRIPTION="Analyzes the Vocal ABC voice and shifts it toward a selected singing range. Key-safe mode uses octave-only shifts."
    RANGES={"female_contralto":(52,77),"female_alto":(53,77),"female_mezzo":(57,81),
            "female_soprano":(60,84),"male_bass":(40,64),"male_baritone":(45,69),"male_tenor":(48,72)}
    @classmethod
    def INPUT_TYPES(cls): return {"required":{
        "abc":("STRING",{"multiline":True}),
        "target_voice":(["original",*cls.RANGES.keys()],),
        "mode":(["nearest_key_safe","nearest_octave","manual"],),
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
            raw=center-median
            # Pitch-class-preserving octave shifts are the only genuinely
            # harmony-safe automatic operation on one voice.
            shift=int(round(raw/12))*12
        out,mod_report=ABCModifier().modify(abc,"keep",120,1,"none",0,False,"",
                                             vocal_transpose=shift)
        report=(f"target={target_voice} source={_note_name(pitches[0])}..{_note_name(pitches[-1])} "
                f"median={_note_name(median)} applied={shift:+d} semitones | {mod_report}")
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
        _report,bpm,_key,_dur=analyze_abc(abc)
        if preset=="light": threshold,outliers=30.,False
        elif preset=="medium": threshold,outliers=50.,True
        else: threshold,outliers=remove_notes_shorter_than_ms,remove_pitch_outliers
        pitches=sorted(p for v,_s,p in score_notes(abc) if "vocal" in v.lower())
        median=pitches[len(pitches)//2] if pitches else 60
        base_ms=60000/max(bpm,1)/8 # default L:1/32 used by SheetSage2
        voice="default"; removed_short=removed_outlier=0; lines=[]
        for raw in abc.splitlines():
            if raw.startswith("V:"): voice=raw[2:].strip().split()[0]
            if "vocal" not in voice.lower() or re.match(r"^[A-Za-z]:",raw): lines.append(raw); continue
            state={}; keyacc=_key_accidentals(re.search(r"(?m)^K:\s*(.+)$",abc).group(1) if re.search(r"(?m)^K:\s*(.+)$",abc) else "C")
            def repl(m):
                nonlocal removed_short,removed_outlier
                d=m.group("dur") or "1"
                if "/" in d:
                    a,b=(d.split("/",1)+["2"])[:2]; units=(float(a) if a else 1)/(float(b) if b else 2)
                else: units=float(d)
                pitch=_pitch(m,keyacc,state)
                short=units*base_ms < threshold
                extreme=outliers and abs(pitch-median)>outlier_semitones
                if short or extreme:
                    removed_short+=int(short); removed_outlier+=int(extreme and not short)
                    return "z"+(m.group("dur") or "")
                return m.group(0)
            pieces=[]
            for quoted,seg in _music_segments(raw): pieces.append(seg if quoted else NOTE_RE.sub(repl,seg))
            lines.append("".join(pieces))
        return "\n".join(lines).rstrip()+"\n",f"preset={preset} short_notes_removed={removed_short} pitch_outliers_removed={removed_outlier}"


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
        "extra":("STRING",{"default":""})}}
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


NODE_CLASS_MAPPINGS={"YuE2ABCAnalyzer":ABCAnalyzer,"YuE2ABCModifier":ABCModifier,
 "YuE2VocalRangeRetarget":VocalRangeRetarget,"YuE2MelodyCleanup":MelodyCleanup,
 "YuE2LoadABCFile":ABCFileLoader,"YuE2SaveABCFile":ABCFileSaver,
 "YuE2StyleBuilder":YuE2StyleBuilder,"YuE2LyricsMelodyFit":LyricsMelodyFit}
NODE_DISPLAY_NAME_MAPPINGS={"YuE2ABCAnalyzer":"YuE2 ABC Analyzer","YuE2ABCModifier":"YuE2 ABC Modifier",
 "YuE2VocalRangeRetarget":"YuE2 Vocal Range Retarget","YuE2MelodyCleanup":"YuE2 Melody Cleanup",
 "YuE2LoadABCFile":"YuE2 ABC File Loader","YuE2SaveABCFile":"YuE2 ABC File Saver",
 "YuE2StyleBuilder":"YuE2 Style Prompt Builder","YuE2LyricsMelodyFit":"YuE2 Lyrics / Melody Fit Analyzer"}

"""Apply ComfyUI's SheetSage2 key-aware chord spelling to legacy ABC output.

The native audio encoder already performs this correction while rebuilding
ABC. Legacy HF snapshots can contain older notation code, so their returned
score is normalized here using the implementation shipped by ComfyUI rather
than carrying a second copy of the spelling algorithm.
"""
from __future__ import annotations

import re
from collections.abc import Callable


_KEY_VALUE_RE = re.compile(
    r"^(?P<before>\s*)(?P<root>[A-G][#b]?)(?P<mode>major|maj|minor|min|m)?"
    r"(?P<after>(?:\s.*|%.*)?)$", re.IGNORECASE,
)
_KEY_LINE_RE = re.compile(r"^(?P<prefix>\s*K\s*:\s*)(?P<value>.*)$")
_CHORD_ROOT_RE = re.compile(r"^(?P<root>[A-G](?:#{1,2}|b{1,2})?)(?P<body>.*)$")

# Inverse of SheetSage2's ABC serialization vocabulary. Unknown qualities are
# intentionally left untouched instead of being treated as major chords.
_ABC_QUALITY_TO_LABEL = {
    "": "maj",
    "m": "min",
    "dim": "dim",
    "aug": "aug",
    "7": "7",
    "maj7": "maj7",
    "m7": "min7",
    "dim7": "dim7",
    "m7b5": "hdim7",
    "sus4": "sus4",
    "sus2": "sus2",
    "6": "maj6",
    "m6": "min6",
    "7sus4": "sus4(b7)",
    "m(maj7)": "minmaj7",
}


def _load_core_helpers() -> tuple[Callable[[str, str], str], Callable[[str], str]] | None:
    try:
        from comfy.audio_encoders.sheetsage2_abc import correct_chord_spelling, normalize_key_name
    except (ImportError, AttributeError):
        # Both helpers are required for consistent key and chord spelling.
        # Older cores leave the legacy ABC unchanged.
        return None
    return correct_chord_spelling, normalize_key_name


def _normalize_abc_key(value: str, normalizer: Callable[[str], str]) -> tuple[str, str | None]:
    match = _KEY_VALUE_RE.fullmatch(value)
    if match is None:
        return value, None
    trailing = match.group("after").lstrip()
    if trailing and not trailing.startswith("%") and re.match(r"[A-Za-z]+(?:\s|$)", trailing):
        # A separated mode name is not a key-field attribute such as clef=.
        return value, None
    alias = (match.group("mode") or "").lower()
    mode = "minor" if alias in {"m", "min", "minor"} else "major"
    source_root = match.group("root")
    source_root = source_root[0].upper() + source_root[1:].lower()
    try:
        normalized = normalizer(f"{source_root}:{mode}")
        root, normalized_mode = normalized.split(":", 1)
    except (ValueError, KeyError, TypeError):
        return value, None
    if normalized_mode != mode or not re.fullmatch(r"[A-G](?:#|b)?", root):
        return value, None
    rewritten = (match.group("before") + root + (match.group("mode") or "")
                 + match.group("after"))
    return rewritten, normalized


def _rewrite_chord(text: str, key: str | None,
                   corrector: Callable[[str, str], str]) -> str:
    if key is None or not text:
        return text

    # Keep any accidental whitespace inside the chord's quotes byte-for-byte.
    leading = text[:len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()):]
    chord = text.strip()
    match = _CHORD_ROOT_RE.fullmatch(chord)
    if match is None:
        return text

    root, body = match.group("root"), match.group("body")
    if "/" in body:
        quality_suffix, slash_bass = body.split("/", 1)
        bass_suffix = "/" + slash_bass
    else:
        quality_suffix, bass_suffix = body, ""
    quality = _ABC_QUALITY_TO_LABEL.get(quality_suffix)
    if quality is None:
        return text

    try:
        corrected = corrector(f"{root}:{quality}{bass_suffix}", key)
    except (ValueError, KeyError):
        # Malformed or unsupported labels are safer left as the model emitted
        # them; spelling correction must never change chord quality.
        return text
    corrected_root = corrected.partition(":")[0]
    if not corrected_root:
        return text
    return leading + corrected_root + quality_suffix + bass_suffix + trailing


def _rewrite_line(line: str, key: str | None,
                  corrector: Callable[[str, str], str],
                  normalizer: Callable[[str], str]) -> tuple[str, str | None]:
    out: list[str] = []
    i = 0
    while i < len(line):
        if line[i] == "%":
            out.append(line[i:])
            break
        if line.startswith("[K:", i):
            end = line.find("]", i + 3)
            if end < 0:
                out.append(line[i:])
                break
            value, key = _normalize_abc_key(line[i + 3:end], normalizer)
            out.append("[K:" + value + "]")
            i = end + 1
            continue
        if line[i] == '"':
            end = line.find('"', i + 1)
            if end < 0:
                out.append(line[i:])
                break
            out.append('"' + _rewrite_chord(line[i + 1:end], key, corrector) + '"')
            i = end + 1
            continue
        out.append(line[i])
        i += 1
    else:
        return "".join(out), key
    return "".join(out), key


def correct_abc_chord_spellings(
    abc_text: str,
    *,
    corrector: Callable[[str, str], str] | None = None,
    normalizer: Callable[[str], str] | None = None,
) -> str:
    """Canonicalize keys and correct quoted chord roots relative to them.

    Notes, durations, barlines, slash-bass text, quality suffixes, comments,
    and line endings are preserved. Inline ``[K:...]`` changes take effect
    from their position in the score onward.
    """
    if not abc_text:
        return abc_text
    if corrector is None or normalizer is None:
        helpers = _load_core_helpers()
        if helpers is None:
            return abc_text
        corrector = corrector or helpers[0]
        normalizer = normalizer or helpers[1]

    current_key = None
    output: list[str] = []
    for line in abc_text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        newline = line[len(content):]
        key_line = _KEY_LINE_RE.match(content)
        if key_line:
            value, current_key = _normalize_abc_key(key_line.group("value"), normalizer)
            content = key_line.group("prefix") + value
        rewritten, current_key = _rewrite_line(content, current_key, corrector, normalizer)
        output.append(rewritten + newline)
    return "".join(output)

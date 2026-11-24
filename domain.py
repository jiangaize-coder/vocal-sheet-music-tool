#!/usr/bin/env python3
"""
domain.py - Single source of truth domain model for Jianpu / multi-part vocal scores.

Design rules:
  - All transcription, synthesis, and PDF/HTML engraving read/write this single typed model.
  - Illegal rhythmic/pitch states are rejected at the boundary via `validate()`.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

KEY_TO_SEMITONE: Dict[str, int] = {
    "C": 0, "C#": 1, "DB": 1,
    "D": 2, "D#": 3, "EB": 3,
    "E": 4,
    "F": 5, "F#": 6, "GB": 6,
    "G": 7, "G#": 8, "AB": 8,
    "A": 9, "A#": 10, "BB": 10,
    "B": 11,
}

DEGREE_TO_SEMITONE: Dict[str, int] = {
    "1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11,
}

SEMITONE_TO_DEGREE: Dict[int, Tuple[str, str]] = {
    0: ("", "1"), 1: ("#", "1"), 2: ("", "2"), 3: ("#", "2"),
    4: ("", "3"), 5: ("", "4"), 6: ("#", "4"), 7: ("", "5"),
    8: ("#", "5"), 9: ("", "6"), 10: ("b", "7"), 11: ("", "7"),
}


def parse_key_tonic_midi(key_str: str) -> Tuple[str, int]:
    """Normalize a key string (e.g. '1=bB', 'Bb', 'G') into (display_key, tonic_midi_around_C4)."""
    k = key_str.strip().upper().replace("♭", "B").replace("♯", "#")
    if k.startswith("1="):
        k = k[2:].strip()
    if k.endswith("M") and len(k) >= 2:
        k = k[:-1]
    if k not in KEY_TO_SEMITONE:
        raise ValueError(f"Unsupported key signature: {key_str!r}")
    tonic = 60 + KEY_TO_SEMITONE[k]
    if tonic >= 69:
        tonic -= 12
    label = k[0] + (k[1:].lower() if len(k) > 1 else "")
    return label, tonic


def sanitize_ascii(text: str, fallback: str = "Track") -> str:
    """Enforce ASCII-safe strings for MIDI track names (prevents latin-1 encode errors)."""
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else "" for ch in text).strip()
    return cleaned or fallback


@dataclass
class NoteToken:
    degree: str          # "0"-"7" or "-"
    beats: float         # quarter-note units (e.g. 1.0, 0.5, 0.25, 0.75, 1/3)
    octave: int = 0      # +1 high dot, -1 low dot
    accidental: str = "" # "", "#", "b"
    dotted: bool = False
    triplet: bool = False
    lyrics: List[str] = field(default_factory=list)  # [verse1, verse2, ...]
    beam_lines: Optional[int] = None  # printed reduction lines from image recognition
    dots: Optional[int] = None        # explicit number of augmentation dots
    tuplet_actual: Optional[int] = None
    tuplet_start: bool = False
    tuplet_stop: bool = False
    articulation: Optional[str] = None
    display_accidental: Optional[str] = None  # empty means a tied note carries the previous sign

    @property
    def accidental_mark(self) -> str:
        return self.accidental if self.display_accidental is None else self.display_accidental

    @property
    def is_note(self) -> bool:
        return self.degree in DEGREE_TO_SEMITONE

    @property
    def is_dash(self) -> bool:
        return self.degree == "-"

    @property
    def underlines(self) -> int:
        if self.beam_lines is not None:
            return self.beam_lines
        if self.is_dash or self.beats >= 1.0 - 1e-4:
            return 0
        if self.triplet:
            return 1
        if self.beats <= 0.25 + 1e-4:
            return 2
        return 1

    def to_midi(self, tonic_midi: int) -> Optional[int]:
        if not self.is_note:
            return None
        acc = 1 if self.accidental == "#" else (-1 if self.accidental == "b" else 0)
        pitch = tonic_midi + DEGREE_TO_SEMITONE[self.degree] + acc + 12 * self.octave
        return max(36, min(96, pitch))


@dataclass
class BeatGroup:
    tokens: List[NoteToken] = field(default_factory=list)

    @property
    def total_beats(self) -> float:
        return sum(t.beats for t in self.tokens)


@dataclass
class Measure:
    number: int
    beats: List[BeatGroup]
    key: Optional[str] = None          # Optional modulation key at this bar (e.g. "B")
    repeat_start: bool = False         # ||:
    repeat_end: bool = False           # :||
    ending: Optional[str] = None       # "1." or "2."
    badge_note: str = ""               # e.g. "转1=B"
    expected_beats: Optional[float] = None  # confirmed pickup or local measure length
    show_shared_lyrics: bool = False       # print identical lyrics on both voices here
    chords: List[str] = field(default_factory=list)  # bar chords (e.g. ["Bb", "F"])
    section: str = ""                  # optional form label (e.g. "主歌", "副歌")

    @property
    def total_beats(self) -> float:
        return sum(b.total_beats for b in self.beats)

    def all_tokens(self) -> List[NoteToken]:
        out: List[NoteToken] = []
        for b in self.beats:
            out.extend(b.tokens)
        return out


@dataclass(frozen=True)
class Arc:
    kind: str
    start_measure: int
    start_token: int
    end_measure: int
    end_token: int

    @property
    def start(self) -> Tuple[int, int]:
        return self.start_measure, self.start_token

    @property
    def end(self) -> Tuple[int, int]:
        return self.end_measure, self.end_token


@dataclass
class Part:
    name: str
    ascii_name: str
    measures: List[Measure]
    arcs: List[Arc] = field(default_factory=list)


@dataclass
class Score:
    title: str
    composer: str = ""
    key: str = "C"
    time_sig: str = "4/4"
    bpm: float = 100.0
    parts: List[Part] = field(default_factory=list)
    play_order: Optional[List[int]] = None  # 1-based measure playback order if repeats exist
    system_breaks: Optional[List[List[int]]] = None
    rows_per_page: int = 5
    engraver: str = ""
    beat_times: Optional[List[float]] = None  # audio timestamps (sec) per quarter beat
    audio_path: str = ""                      # optional source audio path
    mood: str = ""                            # tempo-line mood text, e.g. "阳光、活泼地"

    @property
    def beats_per_measure(self) -> float:
        num, den = [int(x) for x in self.time_sig.split("/")]
        return num * (4.0 / den)

    def validate(self) -> List[str]:
        """Return list of structural violations; empty list means valid."""
        errors: List[str] = []
        if not self.parts:
            errors.append("Score has no parts.")
            return errors
        expected = self.beats_per_measure
        # For 2/2 cut time written as 4 quarter-beats per measure, expected is 4.0
        for part in self.parts:
            for m in part.measures:
                bar_expected = expected if m.expected_beats is None else m.expected_beats
                if abs(m.total_beats - bar_expected) > 1e-2:
                    errors.append(
                        f"Part '{part.name}' Measure {m.number}: beat sum {m.total_beats:.3f} != {bar_expected:.3f}"
                    )
                if m.key:
                    try:
                        parse_key_tonic_midi(m.key)
                    except ValueError as e:
                        errors.append(f"Part '{part.name}' Measure {m.number}: {e}")
            if part.arcs:
                _, tonic = parse_key_tonic_midi(self.key)
                pitches = {}
                positions = []
                for measure in part.measures:
                    if measure.key:
                        try:
                            _, tonic = parse_key_tonic_midi(measure.key)
                        except ValueError:
                            continue
                    for index, token in enumerate(measure.all_tokens()):
                        if not token.is_dash:
                            position = (measure.number, index)
                            positions.append(position)
                            pitches[position] = token.to_midi(tonic)
                adjacent = set(zip(positions, positions[1:]))
                for arc in part.arcs:
                    if (
                        arc.kind not in {"tie", "slur"} or arc.start >= arc.end
                        or pitches.get(arc.start) is None or pitches.get(arc.end) is None
                        or (arc.kind == "tie" and (
                            (arc.start, arc.end) not in adjacent
                            or pitches[arc.start] != pitches[arc.end]
                        ))
                    ):
                        errors.append(f"Part '{part.name}': invalid {arc.kind} {arc.start} → {arc.end}")
        return errors

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "Score":
        raw = json.loads(path.read_text(encoding="utf-8"))
        parts: List[Part] = []
        for p in raw.get("parts", []):
            measures: List[Measure] = []
            for m in p.get("measures", []):
                beats = [
                    BeatGroup(tokens=[NoteToken(**t) for t in b.get("tokens", [])])
                    for b in m.get("beats", [])
                ]
                measures.append(Measure(
                    number=m["number"],
                    beats=beats,
                    key=m.get("key"),
                    repeat_start=m.get("repeat_start", False),
                    repeat_end=m.get("repeat_end", False),
                    ending=m.get("ending"),
                    badge_note=m.get("badge_note", ""),
                    expected_beats=m.get("expected_beats"),
                    show_shared_lyrics=m.get("show_shared_lyrics", False),
                    chords=list(m.get("chords", [])),
                    section=m.get("section", ""),
                ))
            parts.append(Part(
                name=p["name"], ascii_name=p.get("ascii_name") or sanitize_ascii(p["name"]),
                measures=measures, arcs=[Arc(**arc) for arc in p.get("arcs", [])],
            ))
        return cls(
            title=raw["title"],
            composer=raw.get("composer", ""),
            key=raw.get("key", "C"),
            time_sig=raw.get("time_sig", "4/4"),
            bpm=float(raw.get("bpm", 100.0)),
            parts=parts,
            play_order=raw.get("play_order"),
            system_breaks=raw.get("system_breaks"),
            rows_per_page=int(raw.get("rows_per_page", 5)),
            engraver=raw.get("engraver", ""),
            beat_times=raw.get("beat_times"),
            audio_path=raw.get("audio_path", ""),
            mood=raw.get("mood", ""),
        )


def timed_tokens(tokens):
    """Yield exact quarter-note onset, duration, and the original note."""
    onset = Fraction()
    for token in tokens:
        duration = Fraction(token.beats).limit_denominator(960)
        yield onset, duration, token
        onset += duration


def beat_groups(tokens: List[NoteToken]) -> List[BeatGroup]:
    groups, group = [], []
    start = cursor = 0.0
    for token in tokens:
        if group and int(cursor + 1e-7) > int(start + 1e-7):
            groups.append(BeatGroup(group))
            group, start = [], cursor
        group.append(token)
        cursor += token.beats
    if group:
        groups.append(BeatGroup(group))
    return groups


def parse_jianpu_string_to_score(
    text: str,
    title: str = "简谱",
    default_key: str = "C",
    default_bpm: float = 100.0,
    default_time_sig: str = "4/4",
) -> Score:
    """Parse flexible Jianpu text syntax into a validated single-part Score."""
    key_str = default_key
    bpm = default_bpm
    time_sig = default_time_sig

    m_key = re.search(r"1\s*=\s*([A-Ga-g][#b♯♭]?)", text)
    if m_key:
        key_str = m_key.group(1)
    m_bpm = re.search(r"(?:BPM|bpm|♩)\s*=\s*(\d+(?:\.\d+)?)", text)
    if m_bpm:
        bpm = float(m_bpm.group(1))
    m_mood = re.search(r"mood=(\S+)", text)
    mood = m_mood.group(1) if m_mood else ""
    meter_pattern = r"(?<!\S)([1-9]\d*/(?:2|4|8|16|32))(?=\s|[|｜]|$)"
    m_ts = re.search(meter_pattern, text)
    if m_ts:
        time_sig = m_ts.group(1)

    key_label, _ = parse_key_tonic_midi(key_str)
    clean_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("《", "注：", "注:", "//", "#", "title=", "LP:")):
            continue
        s = re.sub(r"1\s*=\s*[A-Ga-g][#b♯♭]?", " ", s)
        s = re.sub(r"(?:BPM|bpm|♩)\s*=\s*\d+(?:\.\d+)?", " ", s)
        s = re.sub(r"mood=\S+", " ", s)
        s = re.sub(meter_pattern, " ", s)
        clean_lines.append(s)

    body = re.sub(r"[|｜]+", " | ", " ".join(clean_lines)).replace("—", " - ").replace("–", " - ")
    tokens: List[NoteToken] = []
    explicit_bars: List[List[NoteToken]] = []
    for raw_tok in body.split():
        tok = raw_tok.strip()
        if not tok:
            continue
        if tok == "|":
            if tokens:
                explicit_bars.append(tokens)
                tokens = []
            continue
        if tok == "-":
            tokens.append(NoteToken(degree="-", beats=1.0))
            continue

        beats = 1.0
        triplet = False
        dotted = False
        work = tok

        if work.endswith("/3"):
            beats, triplet, work = 1.0 / 3.0, True, work[:-2]
        elif work.startswith("((") and work.endswith("))") and len(work) > 4:
            beats, work = 0.25, work[2:-2]
        elif work.startswith("(") and work.endswith(")") and len(work) > 2:
            beats, work = 0.5, work[1:-1]

        if work.startswith("s") and any(c.isdigit() for c in work):
            beats, work = 0.25, work[1:]
        elif work.startswith("q") and any(c.isdigit() for c in work):
            beats, work = 0.5, work[1:]

        if work.endswith("_."):
            beats, dotted, work = 0.75, True, work[:-2]
        elif work.endswith("__") or work.endswith("="):
            beats, work = 0.25, work[:-2] if work.endswith("__") else work[:-1]
        elif work.endswith("_"):
            beats, work = 0.5, work[:-1]
        elif work.endswith("."):
            beats, dotted, work = 1.5, True, work[:-1]

        oct_up = work.count("'") + work.count("^") + work.count("+")
        oct_dn = work.count(",")
        acc = "#" if ("#" in work or "♯" in work) else ("b" if ("b" in work or "♭" in work) else "")
        m_deg = re.search(r"[0-7]", work)
        if not m_deg:
            continue
        tokens.append(NoteToken(
            degree=m_deg.group(0),
            beats=beats,
            octave=oct_up - oct_dn,
            accidental=acc,
            dotted=dotted,
            triplet=triplet,
        ))

    num, den = [int(x) for x in time_sig.split("/")]
    measure_q = num * (4.0 / den)

    if "|" in body:
        if tokens:
            explicit_bars.append(tokens)
        measures = [Measure(i, beat_groups(bar)) for i, bar in enumerate(explicit_bars, 1)]
        return Score(
            title=title, key=key_label, time_sig=time_sig, bpm=bpm, mood=mood,
            parts=[Part(name="主声", ascii_name="Lead", measures=measures)],
        )

    beats_list = beat_groups(tokens)
    measures: List[Measure] = []
    cur_m: List[BeatGroup] = []
    acc_m = 0.0
    m_num = 1
    for bg in beats_list:
        cur_m.append(bg)
        acc_m += bg.total_beats
        if acc_m >= measure_q - 1e-4:
            measures.append(Measure(number=m_num, beats=cur_m))
            cur_m, acc_m, m_num = [], 0.0, m_num + 1
    if cur_m:
        pad = round((measure_q - acc_m) * 4) / 4.0
        while pad >= 1.0 - 1e-4:
            cur_m.append(BeatGroup(tokens=[NoteToken(degree="0", beats=1.0)]))
            pad -= 1.0
        if pad > 1e-4:
            cur_m.append(BeatGroup(tokens=[NoteToken(degree="0", beats=pad)]))
        measures.append(Measure(number=m_num, beats=cur_m))

    return Score(
        title=title,
        key=key_label,
        time_sig=time_sig,
        bpm=bpm,
        mood=mood,
        parts=[Part(name="主声", ascii_name="Lead", measures=measures)],
    )

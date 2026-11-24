#!/usr/bin/env python3
"""
theory.py - 4-Pillar Music Theory & 音值组合法 (Note Grouping) Engine for P2 声乐乐谱工具.

Encodes executable music-theory constraints for AI transcription ("猜测"),
score generation ("生成"), and Jianpu engraving ("排版"):
  1. 节奏全量与音值组合法 (Rhythm, Meter & Note Grouping)
  2. 音高与声乐音域体系 (Pitch, Scale, Modulation & Vocal Tessitura)
  3. 曲式与结构体系 (Musical Form & Section Detection)
  4. 和声与分声部体系 (Functional Harmony, Beat-level Chord Recognition & Part Writing)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from domain import (
    BeatGroup,
    DEGREE_TO_SEMITONE,
    KEY_TO_SEMITONE,
    Measure,
    NoteToken,
    Part,
    SEMITONE_TO_DEGREE,
    Score,
    parse_key_tonic_midi,
)

# ---------------------------------------------------------------------------
# Pillar 1: 节奏全量与音值组合法 (Meter Taxonomy & Note Grouping)
# ---------------------------------------------------------------------------

METER_PROFILES: Dict[str, Dict] = {
    "2/4": {"kind": "单拍子", "unit_q": 1.0, "half_bar_q": None, "strength": [1.0, 0.4]},
    "3/4": {"kind": "单拍子", "unit_q": 1.0, "half_bar_q": None, "strength": [1.0, 0.4, 0.4]},
    "4/4": {"kind": "复拍子", "unit_q": 1.0, "half_bar_q": 2.0, "strength": [1.0, 0.35, 0.75, 0.35]},
    "2/2": {"kind": "单拍子(二二)", "unit_q": 1.0, "half_bar_q": 2.0, "strength": [1.0, 0.35, 0.75, 0.35]},
    "6/8": {"kind": "复拍子", "unit_q": 1.5, "half_bar_q": 1.5, "strength": [1.0, 0.3, 0.3, 0.75, 0.3, 0.3]},
    "3/8": {"kind": "单拍子", "unit_q": 1.5, "half_bar_q": None, "strength": [1.0, 0.35, 0.35]},
    "9/8": {"kind": "复拍子", "unit_q": 1.5, "half_bar_q": None, "strength": [1.0, 0.3, 0.3, 0.7, 0.3, 0.3, 0.7, 0.3, 0.3]},
    "12/8": {"kind": "复拍子", "unit_q": 1.5, "half_bar_q": 3.0, "strength": [1.0, 0.3, 0.3, 0.65, 0.3, 0.3, 0.8, 0.3, 0.3, 0.65, 0.3, 0.3]},
    "5/4": {"kind": "混合拍子", "unit_q": 1.0, "half_bar_q": 3.0, "strength": [1.0, 0.35, 0.35, 0.75, 0.35]},
    "7/8": {"kind": "混合拍子", "unit_q": 1.0, "half_bar_q": 2.0, "strength": [1.0, 0.3, 0.65, 0.3, 0.75, 0.3, 0.3]},
}

# Standard vocal 1-beat (1.0 quarter) rhythm templates for Jianpu readability
VOCAL_BEAT_TEMPLATES: List[Tuple[str, Tuple[float, ...]]] = [
    ("四分", (1.0,)),
    ("二八", (0.5, 0.5)),
    ("前八后十六", (0.5, 0.25, 0.25)),
    ("前十六后八", (0.25, 0.25, 0.5)),
    ("小切分", (0.25, 0.5, 0.25)),
    ("前附点", (0.75, 0.25)),
    ("后附点", (0.25, 0.75)),
    ("四十六", (0.25, 0.25, 0.25, 0.25)),
]


def _decompose_duration(q: float) -> List[float]:
    std = [2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
    out: List[float] = []
    rem = round(q * 4) / 4.0
    while rem >= 0.25 - 1e-6:
        for s in std:
            if rem >= s - 1e-6:
                out.append(s)
                rem = round((rem - s) * 4) / 4.0
                break
        else:
            break
    return out


def normalize_measure_rhythm(measure: Measure, time_sig: str = "4/4") -> Measure:
    """Apply 音值组合法 (Note Grouping) to a measure without altering melody pitches or lyrics.

    Rules enforced:
      - Merges consecutive rests inside the SAME unit beat (e.g. `0= 0=` -> `0_`, `0_ 0_` -> `0`).
      - Never merges rests across the 4/4 half-bar boundary (beat 2 -> beat 3) or across beats
        when lyrics/notes split them.
      - Re-groups tokens cleanly into unit-beat BeatGroups (1.0q for x/4 & 2/2, 1.5q for 6/8).
    """
    profile = METER_PROFILES.get(time_sig, METER_PROFILES["4/4"])
    unit_q = float(profile["unit_q"])
    half_bar_q = profile["half_bar_q"]

    # Step 1: Collect flat events with exact quarter-note offsets
    raw_tokens = measure.all_tokens()
    if any(t.triplet or t.tuplet_actual for t in raw_tokens):
        # Preserve explicit tuplets untouched
        return measure

    events: List[Tuple[float, NoteToken]] = []
    pos = 0.0
    for tok in raw_tokens:
        dur = round(tok.beats * 4) / 4.0
        if dur <= 1e-6:
            continue
        events.append((pos, copy.deepcopy(tok)))
        pos = round((pos + dur) * 4) / 4.0

    # Step 2: Merge adjacent rests ONLY within the same unit beat (respecting half-bar boundary)
    merged: List[Tuple[float, NoteToken]] = []
    for start, tok in events:
        if (
            merged
            and tok.degree == "0"
            and not tok.lyrics
            and merged[-1][1].degree == "0"
            and not merged[-1][1].lyrics
        ):
            prev_start, prev_tok = merged[-1]
            same_beat = int(prev_start / unit_q + 1e-6) == int(start / unit_q + 1e-6)
            crosses_half = (
                half_bar_q is not None
                and prev_start < half_bar_q - 1e-6
                and start + tok.beats > half_bar_q + 1e-6
            )
            combined = round((prev_tok.beats + tok.beats) * 4) / 4.0
            # In Jianpu 音值组合法, merge rests to 0.5 (0_) or 1.0 (0), never dotted 0.75 (0_.)
            if same_beat and not crosses_half and combined in (0.5, 1.0) and combined <= unit_q + 1e-6:
                prev_tok.beats = combined
                prev_tok.dotted = False
                prev_tok.beam_lines = None
                continue
        merged.append((start, tok))

    # Step 3: Split into clean unit-beat BeatGroups (never letting a rest span across unit_q)
    groups: List[BeatGroup] = []
    cur_toks: List[NoteToken] = []
    acc_b = 0.0
    for _, tok in merged:
        rem = round(tok.beats * 4) / 4.0
        first_piece = True
        while rem >= 0.25 - 1e-6:
            space = round((unit_q - acc_b) * 4) / 4.0
            if space <= 1e-6:
                groups.append(BeatGroup(tokens=cur_toks))
                cur_toks, acc_b = [], 0.0
                space = unit_q
            take = min(rem, space)
            for piece in _decompose_duration(take):
                if tok.degree == "0":
                    piece_tok = NoteToken(degree="0", beats=piece, dotted=(piece in (0.75, 1.5)))
                elif first_piece:
                    piece_tok = copy.deepcopy(tok)
                    piece_tok.beats = piece
                    piece_tok.dotted = piece in (0.75, 1.5)
                    piece_tok.beam_lines = None
                    first_piece = False
                else:
                    # Continuation across beat boundary -> Jianpu sustain dash for >=1.0 beat, else tied same pitch
                    if piece >= 1.0 - 1e-4 and acc_b <= 1e-6:
                        piece_tok = NoteToken(degree="-", beats=piece)
                    else:
                        piece_tok = NoteToken(
                            degree=tok.degree, beats=piece, octave=tok.octave,
                            accidental=tok.accidental, dotted=(piece in (0.75, 1.5)),
                        )
                cur_toks.append(piece_tok)
                acc_b = round((acc_b + piece) * 4) / 4.0
                if acc_b >= unit_q - 1e-4:
                    groups.append(BeatGroup(tokens=cur_toks))
                    cur_toks, acc_b = [], 0.0
            rem = round((rem - take) * 4) / 4.0

    if cur_toks:
        groups.append(BeatGroup(tokens=cur_toks))

    out = copy.deepcopy(measure)
    out.beats = groups
    return out


def normalize_score_rhythm(score: Score) -> Score:
    """Apply 音值组合法 across all parts and measures of a Score (preserves parts with explicit arcs)."""
    out = copy.deepcopy(score)
    for part in out.parts:
        if part.arcs:
            continue
        part.measures = [normalize_measure_rhythm(m, out.time_sig) for m in part.measures]
    return out


# ---------------------------------------------------------------------------
# Pillar 2: 音高与声乐音域体系 (Pitch, Scale, Vocal Range & Transposition)
# ---------------------------------------------------------------------------

VOCAL_RANGES: Dict[str, Tuple[int, int, str]] = {
    "children": (57, 77, "少儿舒适音域 (A3–F5)"),
    "soprano": (60, 81, "女高音 (C4–A5)"),
    "alto": (55, 74, "女中音 (G3–D5)"),
    "tenor": (48, 69, "男高音 (C3–A4)"),
    "baritone": (45, 65, "男中音 (A2–F4)"),
}


def check_vocal_range(score: Score, voice_type: str = "children") -> Dict:
    """Analyze pitch range of every part against standard vocal tessitura bounds."""
    lo_lim, hi_lim, label = VOCAL_RANGES.get(voice_type, VOCAL_RANGES["children"])
    report: Dict = {"voice_type": voice_type, "reference": label, "parts": []}
    _, default_tonic = parse_key_tonic_midi(score.key)

    for part in score.parts:
        tonic = default_tonic
        pitches: List[int] = []
        out_of_range: List[Dict] = []
        for m in part.measures:
            if m.key:
                try:
                    _, tonic = parse_key_tonic_midi(m.key)
                except ValueError:
                    pass
            for tok in m.all_tokens():
                mp = tok.to_midi(tonic)
                if mp is not None:
                    pitches.append(mp)
                    if mp < lo_lim or mp > hi_lim:
                        out_of_range.append({"measure": m.number, "midi": mp, "degree": tok.degree})
        report["parts"].append({
            "part": part.name,
            "min_midi": min(pitches) if pitches else None,
            "max_midi": max(pitches) if pitches else None,
            "span_semitones": (max(pitches) - min(pitches)) if pitches else 0,
            "out_of_range_count": len(out_of_range),
            "sample_warnings": out_of_range[:5],
        })
    return report


def transpose_score(score: Score, new_key: str) -> Score:
    """Losslessly transpose a Jianpu score to a new key (movable-do: degrees stay identical, key shifts)."""
    new_label, _ = parse_key_tonic_midi(new_key)
    old_label, _ = parse_key_tonic_midi(score.key)
    shift = (KEY_TO_SEMITONE[new_label.upper()] - KEY_TO_SEMITONE[old_label.upper()]) % 12
    SEMI_TO_KEY = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

    out = copy.deepcopy(score)
    out.key = new_label
    for part in out.parts:
        for m in part.measures:
            if m.key:
                ml, _ = parse_key_tonic_midi(m.key)
                m_new = SEMI_TO_KEY[(KEY_TO_SEMITONE[ml.upper()] + shift) % 12]
                m.key = m_new
                if m.badge_note and "1=" in m.badge_note:
                    m.badge_note = f"转1={m_new}"
            if m.chords:
                new_chords = []
                for ch in m.chords:
                    if not ch:
                        continue
                    root = ch[:2] if len(ch) >= 2 and ch[1] in ("b", "#") else ch[0]
                    suffix = ch[len(root):]
                    if root.upper() in KEY_TO_SEMITONE:
                        r_new = SEMI_TO_KEY[(KEY_TO_SEMITONE[root.upper()] + shift) % 12]
                        new_chords.append(r_new + suffix)
                    else:
                        new_chords.append(ch)
                m.chords = new_chords
    return out


# ---------------------------------------------------------------------------
# Pillar 3: 曲式与结构体系 (Phrase Similarity & Section Labeling)
# ---------------------------------------------------------------------------

def detect_sections(score: Score) -> List[Dict]:
    """Detect 4-bar phrase structure & repeated sections (引子 / 主歌 / 副歌 / 尾声)."""
    if not score.parts or not score.parts[0].measures:
        return []
    measures = score.parts[0].measures
    n = len(measures)
    if n < 4:
        return [{"start": 1, "end": n, "label": "乐段A"}]

    def bar_signature(m: Measure) -> Tuple[int, float, int]:
        notes = [t for t in m.all_tokens() if t.is_note]
        if not notes:
            return (0, 0.0, 0)
        avg_oct = sum(t.octave * 7 + int(t.degree) for t in notes) / len(notes)
        return (len(notes), round(avg_oct, 1), 1 if m.key else 0)

    # Group in 4-bar phrases and label based on register & density
    phrases: List[Dict] = []
    sigs = [bar_signature(m) for m in measures]
    for start in range(0, n, 4):
        end = min(n, start + 4)
        chunk = sigs[start:end]
        avg_notes = sum(c[0] for c in chunk) / len(chunk)
        avg_reg = sum(c[1] for c in chunk) / len(chunk)
        has_mod = any(c[2] for c in chunk)
        if start == 0 and avg_notes <= 4.5:
            lbl = "引子"
        elif has_mod:
            lbl = "副歌(转调)"
        elif avg_reg >= 8.5 and avg_notes >= 6:
            lbl = "副歌"
        elif end == n and avg_notes <= 4.0:
            lbl = "尾声"
        else:
            lbl = "主歌"
        phrases.append({"start": measures[start].number, "end": measures[end - 1].number, "label": lbl})

    # Merge adjacent identical labels
    merged: List[Dict] = []
    for ph in phrases:
        if merged and merged[-1]["label"] == ph["label"]:
            merged[-1]["end"] = ph["end"]
        else:
            merged.append(ph)
    return merged


# ---------------------------------------------------------------------------
# Pillar 4: 和声与和弦识别体系 (Beat-level Chroma + Viterbi Chord Recognition)
# ---------------------------------------------------------------------------

PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Diatonic chord templates (major / minor triads)
def _build_chord_templates() -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    vecs: List[np.ndarray] = []
    for root in range(12):
        # Major triad: root, +4, +7
        v_maj = np.zeros(12, dtype=np.float32)
        v_maj[root] = 1.0
        v_maj[(root + 4) % 12] = 0.85
        v_maj[(root + 7) % 12] = 0.90
        names.append(PITCH_NAMES[root])
        vecs.append(v_maj / np.linalg.norm(v_maj))
        # Minor triad: root, +3, +7
        v_min = np.zeros(12, dtype=np.float32)
        v_min[root] = 1.0
        v_min[(root + 3) % 12] = 0.85
        v_min[(root + 7) % 12] = 0.90
        names.append(PITCH_NAMES[root] + "m")
        vecs.append(v_min / np.linalg.norm(v_min))
    return names, np.stack(vecs, axis=0)


CHORD_NAMES, CHORD_TEMPLATES = _build_chord_templates()


def estimate_bar_chords(
    wav_path: Path,
    beat_times: np.ndarray,
    key_str: str,
    beats_per_bar: int = 4,
    num_measures: int = 0,
    start_beat_idx: int = 0,
    beat_scale: float = 1.0,
) -> List[List[str]]:
    """Estimate diatonic-biased chords per measure using Chroma CQT + Viterbi smoothing."""
    import librosa

    if num_measures <= 0 or len(beat_times) < 4:
        return [[] for _ in range(max(0, num_measures))]

    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=512)

    key_label, _ = parse_key_tonic_midi(key_str)
    tonic_pc = KEY_TO_SEMITONE[key_label.upper()]
    # Diatonic major key chords get a gentle prior boost (I, ii, iii, IV, V, vi)
    diatonic_set = {
        PITCH_NAMES[tonic_pc],
        PITCH_NAMES[(tonic_pc + 2) % 12] + "m",
        PITCH_NAMES[(tonic_pc + 4) % 12] + "m",
        PITCH_NAMES[(tonic_pc + 5) % 12],
        PITCH_NAMES[(tonic_pc + 7) % 12],
        PITCH_NAMES[(tonic_pc + 9) % 12] + "m",
    }
    prior = np.array([0.08 if name in diatonic_set else 0.0 for name in CHORD_NAMES], dtype=np.float32)

    # Compute per-half-bar (2-beat) chroma scores so mid-bar chord changes (e.g. Eb -> F) are captured
    slots_per_bar = 2 if beats_per_bar >= 4 else 1
    total_slots = num_measures * slots_per_bar
    slot_beats = beats_per_bar / slots_per_bar
    n_chords = len(CHORD_NAMES)
    scores = np.zeros((total_slots, n_chords), dtype=np.float32)
    for s_idx in range(total_slots):
        b0 = start_beat_idx + int(round(s_idx * slot_beats * beat_scale))
        b1 = start_beat_idx + int(round((s_idx + 1) * slot_beats * beat_scale))
        if b0 >= len(beat_times):
            scores[s_idx] = prior
            continue
        t0 = beat_times[max(0, b0)]
        t1 = beat_times[min(len(beat_times) - 1, b1)] if b1 < len(beat_times) else frame_times[-1]
        mask = (frame_times >= t0) & (frame_times <= max(t0 + 0.1, t1))
        if not np.any(mask):
            scores[s_idx] = prior
            continue
        vec = chroma[:, mask].mean(axis=1)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        scores[s_idx] = (CHORD_TEMPLATES @ vec) + prior

    # Light Viterbi smoothing (avoids jitter while preserving real half-bar chord changes)
    stay_bonus = 0.02
    dp = np.zeros_like(scores)
    back = np.zeros((total_slots, n_chords), dtype=np.int32)
    dp[0] = scores[0]
    for i in range(1, total_slots):
        for c in range(n_chords):
            trans = dp[i - 1].copy()
            trans[c] += stay_bonus
            best_prev = int(np.argmax(trans))
            dp[i, c] = trans[best_prev] + scores[i, c]
            back[i, c] = best_prev

    path = [0] * total_slots
    path[-1] = int(np.argmax(dp[-1]))
    for i in range(total_slots - 2, -1, -1):
        path[i] = int(back[i + 1, path[i + 1]])

    bar_chords: List[List[str]] = []
    for m_idx in range(num_measures):
        slot_names = [CHORD_NAMES[path[m_idx * slots_per_bar + k]] for k in range(slots_per_bar)]
        dedup: List[str] = []
        for nm in slot_names:
            if not dedup or dedup[-1] != nm:
                dedup.append(nm)
        bar_chords.append(dedup)
    return bar_chords


def audit_score_theory(score: Score, voice_type: str = "children") -> Dict:
    """Run a complete 4-pillar music theory health check on any Score."""
    return {
        "title": score.title,
        "key": f"1={score.key}",
        "time_sig": score.time_sig,
        "meter_kind": METER_PROFILES.get(score.time_sig, {}).get("kind", "自定义拍子"),
        "bpm": score.bpm,
        "validation_errors": score.validate(),
        "vocal_range": check_vocal_range(score, voice_type=voice_type),
        "sections": detect_sections(score),
        "chords_present": any(bool(m.chords) for p in score.parts for m in p.measures),
        "audio_sync_ready": bool(score.beat_times),
    }

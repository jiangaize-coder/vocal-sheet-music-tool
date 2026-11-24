#!/usr/bin/env python3
"""
synth.py - Pure offline audio synthesis & multi-part choral practice track generator.

Consumes `domain.Score` directly and renders:
  - Solo vocal pitch guide (唱音 ah / 哼鸣 oo) & warm piano guide
  - Multi-part choir mix + practice-focus mixes (100% target part + 32% companion part)
  - Standard multi-track MIDI with guaranteed ASCII track names
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import butter, filtfilt, iirpeak, lfilter

from domain import Part, Score, parse_key_tonic_midi, sanitize_ascii


def find_ffmpeg() -> str:
    for c in [shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError("未找到 ffmpeg，请先安装 ffmpeg（macOS 可用 Homebrew：brew install ffmpeg；Linux 可用系统包管理器）。")


def midi_to_hz(midi_pitch: int) -> float:
    return 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))


def synth_piano_note(freq: float, dur_sec: float, sr: int = 44100) -> np.ndarray:
    tail = 0.24
    total_dur = dur_sec + tail
    n = max(1, int(total_dur * sr))
    t = np.linspace(0.0, total_dur, n, endpoint=False)
    harmonics = [(1, 1.00, 1.8), (2, 0.50, 2.5), (3, 0.24, 3.4), (4, 0.14, 4.5), (5, 0.08, 6.0)]
    sig = np.zeros(n, dtype=np.float64)
    for k, amp, decay in harmonics:
        if freq * k < sr * 0.45:
            sig += amp * np.exp(-decay * t) * np.sin(2.0 * np.pi * freq * k * t)
    env = np.ones(n, dtype=np.float64)
    atk = min(int(0.008 * sr), n // 4)
    if atk > 0:
        env[:atk] = np.linspace(0.0, 1.0, atk)
    rel_start = min(int(dur_sec * sr), n - 1)
    if n - rel_start > 0:
        env[rel_start:] = np.exp(-np.linspace(0.0, 6.0, n - rel_start))
    return sig * env * 0.35


def synth_vocal_note(freq: float, dur_sec: float, sr: int = 44100, vowel: str = "ah") -> np.ndarray:
    n = max(1, int(dur_sec * sr))
    t = np.linspace(0.0, dur_sec, n, endpoint=False)
    vib_depth = 0.008 * np.clip((t - 0.16) / 0.25, 0.0, 1.0)
    inst_freq = freq * (1.0 + vib_depth * np.sin(2.0 * np.pi * 5.3 * t))
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / sr

    source = np.zeros(n, dtype=np.float64)
    max_k = min(20, int((sr * 0.42) / max(freq, 50.0)))
    for k in range(1, max_k + 1):
        source += (1.0 / (k ** 1.15)) * np.sin(k * phase)

    rng = np.random.default_rng(int(freq * 100) % 100000)
    noise = rng.normal(0.0, 0.014, n)
    b_lp, a_lp = butter(2, 3200.0 / (sr / 2.0), btype="low")
    if n > 12:
        source += filtfilt(b_lp, a_lp, noise)

    formants = {
        "ah": [(730, 8.0, 1.0), (1150, 10.0, 0.65), (2550, 14.0, 0.25)],
        "oo": [(340, 8.0, 1.0), (850, 10.0, 0.55), (2350, 14.0, 0.18)],
    }.get(vowel, [(730, 8.0, 1.0), (1150, 10.0, 0.65), (2550, 14.0, 0.25)])

    shaped = np.zeros(n, dtype=np.float64)
    for f0, q, gain in formants:
        b, a = iirpeak(f0 / (sr / 2.0), q)
        shaped += gain * lfilter(b, a, source)

    shaped = shaped / (np.max(np.abs(shaped)) + 1e-8)
    env = np.ones(n, dtype=np.float64)
    atk = min(int(0.042 * sr), n // 3)
    rel = min(int(0.060 * sr), n // 3)
    if atk > 0:
        env[:atk] = np.sin(np.linspace(0.0, np.pi / 2.0, atk))
    if rel > 0:
        env[-rel:] = np.cos(np.linspace(0.0, np.pi / 2.0, rel))
    return shaped * env * 0.45


def synth_click(sr: int = 44100, accented: bool = False) -> np.ndarray:
    dur = 0.035
    n = int(dur * sr)
    t = np.linspace(0.0, dur, n, endpoint=False)
    f = 1400.0 if accented else 950.0
    return np.sin(2.0 * np.pi * f * t) * np.exp(-90.0 * t) * (0.28 if accented else 0.16)


def add_room_reverb(mono: np.ndarray, sr: int = 44100) -> np.ndarray:
    delays_ms = [29, 43, 61, 83]
    gains = [0.22, 0.16, 0.11, 0.07]
    left, right = mono.copy(), mono.copy()
    for i, (d_ms, g) in enumerate(zip(delays_ms, gains)):
        d = int(sr * d_ms / 1000.0)
        if d < len(mono):
            if i % 2 == 0:
                left[d:] += g * mono[:-d]
                right[d:] += g * 0.7 * mono[:-d]
            else:
                right[d:] += g * mono[:-d]
                left[d:] += g * 0.7 * mono[:-d]
    stereo = np.stack([left, right], axis=1)
    peak = np.max(np.abs(stereo)) + 1e-8
    return stereo * (0.92 / peak) if peak > 0.92 else stereo


def flatten_part_events(score: Score, part: Part) -> List[Tuple[Optional[int], float]]:
    """Expand score playback order (with modulations and '-' dashes) into (midi_or_None, beats)."""
    by_num = {m.number: m for m in part.measures}
    order = score.play_order or [m.number for m in part.measures]
    _, cur_tonic = parse_key_tonic_midi(score.key)

    events: List[Tuple[Optional[int], float]] = []
    ties = {(arc.start, arc.end) for arc in part.arcs if arc.kind == "tie"}
    previous = None
    for m_num in order:
        m = by_num[m_num]
        if m.key:
            _, cur_tonic = parse_key_tonic_midi(m.key)
        for index, tok in enumerate(m.all_tokens()):
            if tok.is_dash:
                if events:
                    prev_p, prev_b = events[-1]
                    events[-1] = (prev_p, prev_b + tok.beats)
                else:
                    events.append((None, tok.beats))
            else:
                pitch = tok.to_midi(cur_tonic)
                position = (m_num, index)
                if pitch is not None and (previous, position) in ties and events[-1][0] == pitch:
                    events[-1] = (pitch, events[-1][1] + tok.beats)
                else:
                    events.append((pitch, tok.beats))
                previous = position
    return events


def render_part_mono(
    score: Score,
    part: Part,
    timbre: str = "vocal",
    count_in: bool = True,
    metronome: bool = False,
    sr: int = 44100,
    *,
    silent_count_in: bool = False,
) -> np.ndarray:
    events = flatten_part_events(score, part)
    sec_per_beat = 60.0 / score.bpm
    beats_per_bar = score.beats_per_measure
    lead = beats_per_bar if count_in else 0.0
    total_beats = lead + sum(b for _, b in events)
    buf = np.zeros(int((total_beats * sec_per_beat + 1.0) * sr), dtype=np.float64)

    if (count_in and not silent_count_in) or metronome:
        numerator, denominator = (int(n) for n in score.time_sig.split("/"))
        step = 4.0 / denominator
        n_clicks = numerator if not metronome else int(np.ceil(total_beats / step - 1e-9))
        for beat in range(n_clicks):
            pos = int(beat * step * sec_per_beat * sr)
            click = synth_click(sr=sr, accented=(beat % numerator == 0))
            end = min(len(buf), pos + len(click))
            buf[pos:end] += click[: end - pos]

    cursor = lead
    for midi_p, beats in events:
        if midi_p is not None:
            dur = beats * sec_per_beat
            freq = midi_to_hz(midi_p)
            pos = int(cursor * sec_per_beat * sr)
            if timbre == "piano":
                wave = synth_piano_note(freq, dur, sr=sr)
            elif timbre == "vocal_oo":
                wave = synth_vocal_note(freq, dur, sr=sr, vowel="oo")
            else:
                wave = synth_vocal_note(freq, dur, sr=sr, vowel="ah")
            end = min(len(buf), pos + len(wave))
            buf[pos:end] += wave[: end - pos]
        cursor += beats
    return buf


def save_mp3(stereo: np.ndarray, out_mp3: Path, sr: int = 44100) -> Path:
    ffmpeg = find_ffmpeg()
    peak = np.max(np.abs(stereo)) + 1e-8
    if peak > 0.92:
        stereo = stereo * (0.92 / peak)
    tmp_wav = out_mp3.with_suffix(".tmp.wav")
    sf.write(str(tmp_wav), stereo, sr, subtype="PCM_16")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(tmp_wav), "-b:a", "192k", str(out_mp3)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        tmp_wav.unlink(missing_ok=True)
    return out_mp3


def export_score_midi(score: Score, out_midi: Path) -> Path:
    """Export all parts of Score into a standard MIDI with guaranteed ASCII track names."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=score.bpm)
    numerator, denominator = (int(n) for n in score.time_sig.split("/"))
    pm.time_signature_changes.append(pretty_midi.TimeSignature(numerator, denominator, 0))
    sec_per_beat = 60.0 / score.bpm
    for idx, part in enumerate(score.parts):
        track_name = sanitize_ascii(part.ascii_name or part.name, fallback=f"Part_{idx + 1}")
        inst = pretty_midi.Instrument(program=52 if idx == 0 else 53, name=track_name)
        cursor = 0.0
        for midi_p, beats in flatten_part_events(score, part):
            dur = beats * sec_per_beat
            if midi_p is not None:
                inst.notes.append(pretty_midi.Note(
                    velocity=95,
                    pitch=int(midi_p),
                    start=cursor,
                    end=cursor + dur * 0.94,
                ))
            cursor += dur
        pm.instruments.append(inst)
    pm.write(str(out_midi))
    return out_midi


def render_score_audio_bundle(
    score: Score,
    out_dir: Path,
    count_in: bool = True,
    metronome: bool = False,
) -> Dict[str, str]:
    """Render solo, choir, focus-practice (for 2+ parts), piano, and MIDI tracks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = score.title.replace(" ", "_")
    outputs: Dict[str, str] = {}

    midi_path = export_score_midi(score, out_dir / f"{stem}.mid")
    outputs["midi"] = str(midi_path)

    if len(score.parts) == 1:
        p = score.parts[0]
        v = render_part_mono(score, p, "vocal", count_in=count_in, metronome=metronome)
        pi = render_part_mono(score, p, "piano", count_in=count_in, metronome=metronome)
        outputs["vocal_mp3"] = str(save_mp3(add_room_reverb(v), out_dir / f"{stem}_人声唱音.mp3"))
        outputs["piano_mp3"] = str(save_mp3(add_room_reverb(pi), out_dir / f"{stem}_钢琴示范.mp3"))
        return outputs

    # Multi-part choral bundle (e.g. 主声 + 和声)
    p0, p1 = score.parts[0], score.parts[1]
    v0 = render_part_mono(score, p0, "vocal", count_in=count_in, metronome=metronome)
    v0_companion = render_part_mono(score, p0, "vocal", count_in=count_in, silent_count_in=True)
    v1_solo = render_part_mono(score, p1, "vocal", count_in=count_in, metronome=metronome)
    v1_companion = render_part_mono(score, p1, "vocal_oo", count_in=count_in, silent_count_in=True)
    pi0 = render_part_mono(score, p0, "piano", count_in=count_in, metronome=metronome)
    pi1_solo = render_part_mono(score, p1, "piano", count_in=count_in, metronome=metronome)
    pi1_companion = render_part_mono(score, p1, "piano", count_in=count_in, silent_count_in=True)

    n = min(len(v0), len(v1_companion), len(pi0), len(pi1_companion))
    v0, v0_companion, v1_solo, v1_companion = v0[:n], v0_companion[:n], v1_solo[:n], v1_companion[:n]
    pi0, pi1_solo, pi1_companion = pi0[:n], pi1_solo[:n], pi1_companion[:n]

    mix_choir = add_room_reverb(v0 * 0.72 + v1_companion * 0.68)
    mix_choir[:, 0] += v0 * 0.14
    mix_choir[:, 1] += v1_companion * 0.14

    outputs["part1_vocal_mp3"] = str(save_mp3(add_room_reverb(v0), out_dir / f"01-{stem}-{p0.name}独唱-人声唱音.mp3"))
    outputs["part2_vocal_mp3"] = str(save_mp3(add_room_reverb(v1_solo), out_dir / f"02-{stem}-{p1.name}独唱-人声唱音.mp3"))
    outputs["choir_vocal_mp3"] = str(save_mp3(mix_choir, out_dir / f"03-{stem}-双声部合唱-人声唱音.mp3"))
    outputs["part1_focus_mp3"] = str(save_mp3(add_room_reverb(v0 * 1.0 + v1_companion * 0.32), out_dir / f"04-{stem}-练{p0.name}版.mp3"))
    outputs["part2_focus_mp3"] = str(save_mp3(add_room_reverb(v1_solo * 1.0 + v0_companion * 0.32), out_dir / f"05-{stem}-练{p1.name}版.mp3"))
    outputs["choir_piano_mp3"] = str(save_mp3(add_room_reverb(pi0 * 0.72 + pi1_companion * 0.68), out_dir / f"06-{stem}-双声部钢琴示范.mp3"))
    outputs["part1_piano_mp3"] = str(save_mp3(add_room_reverb(pi0), out_dir / f"07-{stem}-{p0.name}钢琴示范.mp3"))
    outputs["part2_piano_mp3"] = str(save_mp3(add_room_reverb(pi1_solo), out_dir / f"08-{stem}-{p1.name}钢琴示范.mp3"))
    return outputs

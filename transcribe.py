#!/usr/bin/env python3
"""
transcribe.py - Local offline audio/video to domain.Score + Staff MusicXML/SVG/PDF pipeline.

Uses:
  - ffmpeg / yt-dlp (input normalization; never bypasses DRM)
  - Meta Demucs htdemucs (optional vocal stem separation)
  - Spotify Basic Pitch ONNX (100% offline audio-to-MIDI)
  - librosa + music21 (BPM, key detection, quantization, MusicXML)
  - Verovio (Staff SVG rendering)
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import music21
import numpy as np
import pretty_midi
import verovio
from basic_pitch.inference import ICASSP_2022_MODEL_PATH, predict

from domain import (
    BeatGroup,
    Measure,
    NoteToken,
    Part,
    SEMITONE_TO_DEGREE,
    Score,
    parse_key_tonic_midi,
)


def find_binary(name: str) -> Optional[str]:
    for c in [
        shutil.which(name),
        f"/opt/homebrew/bin/{name}",
        f"/usr/local/bin/{name}",
        os.path.expanduser(f"~/.local/bin/{name}"),
    ]:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def prepare_wav(input_source: str, work_dir: Path) -> Tuple[Path, str]:
    ffmpeg = find_binary("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("未找到 ffmpeg，请先安装 Homebrew ffmpeg。")

    if input_source.startswith(("http://", "https://")):
        ytdlp = find_binary("yt-dlp")
        if not ytdlp:
            raise FileNotFoundError("未找到 yt-dlp，无法下载公开链接音频。")
        subprocess.run(
            [ytdlp, "-f", "bestaudio/best", "--no-playlist", "-o", str(work_dir / "dl.%(ext)s"), input_source],
            check=True,
        )
        found = sorted(work_dir.glob("dl.*"))
        if not found:
            raise RuntimeError("yt-dlp 未下载到有效音频文件。")
        src, stem = found[0], "online_audio"
    else:
        src = Path(input_source).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"找不到输入文件: {src}")
        stem = src.stem

    wav_path = work_dir / "input_44k.wav"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(src), "-ac", "2", "-ar", "44100", str(wav_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return wav_path, stem


def separate_vocals(wav_path: Path, work_dir: Path) -> Path:
    demucs_out = work_dir / "demucs_out"
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", str(demucs_out), str(wav_path)],
        check=True,
    )
    vocals = list(demucs_out.rglob("vocals.wav"))
    if not vocals:
        raise RuntimeError("Demucs 分离完成，但未找到 vocals.wav。")
    return vocals[0]


def audio_to_midi(
    wav_path: Path,
    midi_out: Path,
    vocal_mode: bool = True,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_ms: float = 120.0,
) -> pretty_midi.PrettyMIDI:
    onnx_model = Path(ICASSP_2022_MODEL_PATH).parent / "nmp.onnx"
    model_path = onnx_model if onnx_model.exists() else ICASSP_2022_MODEL_PATH
    _, midi_data, _ = predict(
        str(wav_path),
        model_or_model_path=model_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_ms,
        minimum_frequency=80.0 if vocal_mode else 32.7,
        maximum_frequency=1100.0 if vocal_mode else 2093.0,
        melodia_trick=True,
    )
    midi_data.write(str(midi_out))
    return midi_data


def estimate_bpm(wav_path: Path, override_bpm: Optional[float] = None) -> float:
    if override_bpm and override_bpm > 0:
        return round(float(override_bpm), 1)
    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    val = float(tempo.flat[0]) if isinstance(tempo, np.ndarray) and tempo.size > 0 else float(tempo)
    if val < 50:
        val *= 2
    elif val > 180:
        val /= 2
    return round(val if 50 <= val <= 180 else 100.0, 1)


def _decompose(q: float) -> List[float]:
    std = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
    out, rem = [], round(q * 4) / 4.0
    while rem >= 0.25 - 1e-6:
        for s in std:
            if rem >= s - 1e-6:
                out.append(s)
                rem = round((rem - s) * 4) / 4.0
                break
        else:
            break
    return out


def transcribe_audio_to_score(
    input_source: str,
    out_dir: Path,
    title: Optional[str] = None,
    separate: bool = False,
    mode: str = "melody",
    key_override: Optional[str] = None,
    bpm_override: Optional[float] = None,
    time_sig: str = "4/4",
) -> Tuple[Score, Dict[str, str]]:
    """Run full audio -> MIDI -> MusicXML + Verovio Staff SVG + domain.Score pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p2_score_") as tmp:
        work = Path(tmp)
        wav_path, auto_stem = prepare_wav(input_source, work)
        display_title = title or auto_stem
        safe_stem = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", display_title).strip("_") or "score"

        target_wav = separate_vocals(wav_path, work) if separate else wav_path
        midi_path = out_dir / f"{safe_stem}.mid"
        pm = audio_to_midi(target_wav, midi_path, vocal_mode=(mode == "melody"))
        bpm = estimate_bpm(wav_path, bpm_override)

        # Beat-synchronous time->quarter-beat mapping from full mix (prevents tempo drift & phase offset)
        y_mix, sr_mix = librosa.load(str(wav_path), sr=22050, mono=True)
        detected_tempo, beat_frames = librosa.beat.beat_track(y=y_mix, sr=sr_mix)
        det_bpm = float(detected_tempo.flat[0]) if isinstance(detected_tempo, np.ndarray) and detected_tempo.size > 0 else float(detected_tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr_mix)

        # Extract monophonic melody for Jianpu domain.Score + MusicXML
        raw = [
            (n.start, n.end, n.pitch, n.velocity)
            for inst in pm.instruments if not inst.is_drum
            for n in inst.notes if n.end - n.start >= 0.06
        ]
        if not raw:
            raise RuntimeError("未能从音频中识别出有效旋律音符。")
        raw.sort(key=lambda x: (x[0], -x[2], -x[3]))
        cleaned: List[List] = []
        for s, e, p, _ in raw:
            if not cleaned or s >= cleaned[-1][1]:
                cleaned.append([s, e, p])
            elif s - cleaned[-1][0] < 0.05:
                if p > cleaned[-1][2]:
                    cleaned[-1][2], cleaned[-1][1] = p, max(cleaned[-1][1], e)
            else:
                cleaned[-1][1] = s
                if cleaned[-1][1] - cleaned[-1][0] < 0.05:
                    cleaned.pop()
                cleaned.append([s, e, p])

        sec_per_q = 60.0 / bpm
        num, den = [int(x) for x in time_sig.split("/")]
        measure_q = num * (4.0 / den)

        use_beat_grid = len(beat_times) >= 8 and det_bpm > 30
        if use_beat_grid:
            ratio = det_bpm / bpm
            if 1.7 <= ratio <= 2.3:
                beat_scale = 2.0
            elif 0.4 <= ratio <= 0.6:
                beat_scale = 0.5
            else:
                beat_scale = 1.0
            idx0 = int(np.argmin(np.abs(beat_times - cleaned[0][0])))
            def to_q(t: float) -> float:
                idx = float(np.interp(t, beat_times, np.arange(len(beat_times))))
                return (idx - idx0) / beat_scale
        else:
            first_q = round((cleaned[0][0] / sec_per_q) * 4) / 4.0
            shift = first_q if first_q >= measure_q else 0.0
            def to_q(t: float) -> float:
                return t / sec_per_q - shift

        m21_score = music21.stream.Score()
        m21_part = music21.stream.Part()
        m21_part.insert(0, music21.meter.TimeSignature(time_sig))
        m21_part.insert(0, music21.tempo.MetronomeMark(number=round(bpm)))

        timeline: List[Tuple[Optional[int], float]] = []
        cursor = 0.0
        for s, e, p in cleaned:
            raw_sq = round(to_q(s) * 4) / 4.0
            raw_eq = round(to_q(e) * 4) / 4.0
            # Drop overlapping vibrato/double-trigger fragments instead of pushing rhythm late
            if raw_sq < cursor - 1e-6:
                continue
            sq = max(0.0, raw_sq)
            dur = max(0.25, raw_eq - sq)
            if sq - cursor >= 0.25 - 1e-6:
                timeline.append((None, round((sq - cursor) * 4) / 4.0))
                cursor = sq
            timeline.append((p, dur))
            cursor = round((cursor + dur) * 4) / 4.0

        for p, q in timeline:
            for piece in _decompose(q):
                m21_part.append(music21.note.Rest(quarterLength=piece) if p is None else music21.note.Note(p, quarterLength=piece))
        m21_part.makeMeasures(inPlace=True)
        m21_score.insert(0, m21_part)

        if key_override:
            key_label, tonic_midi = parse_key_tonic_midi(key_override)
        else:
            k = m21_score.analyze("key")
            rel = k.relative if k.mode == "minor" else k
            key_label, tonic_midi = parse_key_tonic_midi(rel.tonic.name.replace("-", "b"))

        # Export MusicXML + Verovio Staff SVG
        mxml_path = out_dir / f"{safe_stem}.musicxml"
        m21_score.write("musicxml", fp=str(mxml_path))

        tk = verovio.toolkit()
        tk.setOptions({"pageWidth": 1900, "pageHeight": 2600, "scale": 42, "adjustPageHeight": True, "footer": "none"})
        tk.loadFile(str(mxml_path))
        staff_svgs: List[str] = []
        for p_idx in range(1, tk.getPageCount() + 1):
            sp = out_dir / f"{safe_stem}_staff_p{p_idx}.svg"
            sp.write_text(tk.renderToSVG(p_idx), encoding="utf-8")
            staff_svgs.append(str(sp))

        # Convert timeline into validated domain.Score (beat-grouped!)
        measures: List[Measure] = []
        cur_beats: List[BeatGroup] = []
        cur_toks: List[NoteToken] = []
        acc_b, acc_m, m_num = 0.0, 0.0, 1

        def flush_token(tok: NoteToken) -> None:
            nonlocal cur_beats, cur_toks, acc_b, acc_m, m_num
            cur_toks.append(tok)
            acc_b = round((acc_b + tok.beats) * 4) / 4.0
            if acc_b >= 1.0 - 1e-4:
                cur_beats.append(BeatGroup(tokens=cur_toks))
                acc_m = round((acc_m + acc_b) * 4) / 4.0
                cur_toks, acc_b = [], 0.0
                if acc_m >= measure_q - 1e-4:
                    measures.append(Measure(number=m_num, beats=cur_beats))
                    cur_beats, acc_m, m_num = [], 0.0, m_num + 1

        for p, q in timeline:
            rem = round(q * 4) / 4.0
            while rem >= 0.25 - 1e-6:
                beat_space = round((1.0 - acc_b) * 4) / 4.0
                if beat_space <= 1e-6:
                    beat_space = 1.0
                take = min(rem, beat_space)
                for piece in _decompose(take):
                    if p is None:
                        flush_token(NoteToken(degree="0", beats=piece))
                    else:
                        diff = p - tonic_midi
                        acc, deg = SEMITONE_TO_DEGREE[diff % 12]
                        flush_token(NoteToken(degree=deg, beats=piece, octave=math.floor(diff / 12), accidental=acc, dotted=(piece in (0.75, 1.5))))
                rem = round((rem - take) * 4) / 4.0

        if cur_toks:
            pad_b = round((1.0 - acc_b) * 4) / 4.0
            if pad_b >= 0.25:
                cur_toks.append(NoteToken(degree="0", beats=pad_b))
            cur_beats.append(BeatGroup(tokens=cur_toks))
            acc_m = round((acc_m + 1.0) * 4) / 4.0
        if cur_beats:
            while acc_m < measure_q - 1e-4:
                cur_beats.append(BeatGroup(tokens=[NoteToken(degree="0", beats=1.0)]))
                acc_m = round((acc_m + 1.0) * 4) / 4.0
            measures.append(Measure(number=m_num, beats=cur_beats))

        from theory import detect_sections, estimate_bar_chords, normalize_score_rhythm

        beats_per_bar_int = max(1, int(round(measure_q)))
        if use_beat_grid:
            bar_chords = estimate_bar_chords(
                wav_path, beat_times, key_label,
                beats_per_bar=beats_per_bar_int,
                num_measures=len(measures),
                start_beat_idx=idx0,
                beat_scale=beat_scale,
            )
            for m, ch in zip(measures, bar_chords):
                m.chords = ch
            total_q_beats = len(measures) * beats_per_bar_int
            q_grid = idx0 + np.arange(total_q_beats, dtype=float) * beat_scale
            bt_list = [round(float(np.interp(g, np.arange(len(beat_times)), beat_times)), 3) for g in q_grid]
        else:
            bt_list = None

        src_audio_str = ""
        if not input_source.startswith(("http://", "https://")):
            src_audio_str = str(Path(input_source).expanduser().resolve())

        score = Score(
            title=display_title,
            composer="Audio-to-Score (P2 Local AI)",
            key=key_label,
            time_sig=time_sig,
            bpm=bpm,
            parts=[Part(name="主声", ascii_name="Lead", measures=measures)],
            beat_times=bt_list,
            audio_path=src_audio_str,
        )
        score = normalize_score_rhythm(score)
        for sec in detect_sections(score):
            for m in score.parts[0].measures:
                if m.number == sec["start"]:
                    m.section = sec["label"]

        json_path = out_dir / f"{safe_stem}.score.json"
        score.to_json(json_path)

        return score, {
            "midi": str(midi_path),
            "musicxml": str(mxml_path),
            "score_json": str(json_path),
            "staff_svgs": staff_svgs,
        }

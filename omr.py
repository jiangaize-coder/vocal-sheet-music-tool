"""Import Dolce's positioned Jianpu notes before any text format conversion."""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from copy import deepcopy
from fractions import Fraction
from difflib import SequenceMatcher
import json
import math
import os
from pathlib import Path
from shutil import which
from statistics import median
import subprocess

import numpy as np
from PIL import Image, ImageOps

from domain import Arc, Measure, NoteToken, Part, Score, beat_groups, parse_key_tonic_midi, timed_tokens
from omr_fragments import (
    build_page_digit_templates,
    recover_missed_bar_fragments,
    refine_row_fragments,
    save_fragment_diagnostic_sheet,
    solve_bar_rhythm_constraints,
)
from theory import normalize_measure_rhythm


def recognize(images: list[Path], executable: Path | None) -> list[dict]:
    for path in images:
        if not path.is_file():
            raise FileNotFoundError(path)
    if executable is None:
        env_path = os.environ.get("DOLCE_OMR_CLI", "").strip()
        if env_path:
            executable = Path(env_path)
        else:
            installed = which("omr-cli")
            executable = Path(installed) if installed else None
    if executable is None or not executable.is_file():
        raise FileNotFoundError(
            "找不到 Dolce omr-cli：请用 --recognizer 指定本机程序，"
            "或设置 DOLCE_OMR_CLI 环境变量，或把 omr-cli 放到 PATH 上。"
        )
    result = subprocess.run(
        [str(executable.resolve()), "run", *(str(p.resolve()) for p in images), "--kind=jianpu"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Dolce 图片识别失败。")
    pages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(pages) != len(images):
        raise ValueError("Dolce 返回的页数与输入不一致。")
    by_file = {Path(page["file"]).resolve(): page for page in pages}
    return [by_file[path.resolve()] for path in images]


def center(note: dict) -> float:
    box = note["bbox"]
    return box["x"] + box["w"] / 2


def note_tokens(note: dict) -> list[NoteToken]:
    levels = max(0, int(note.get("div", 0)))
    dots = max(0, int(note.get("dot", 0)))
    beats = (2 - 0.5 ** dots) / 2 ** levels
    tuplet = note.get("tuplet") or {}
    if tuplet:
        beats *= tuplet["normal"] / tuplet["actual"]
    accidental = {"flat": "b", "sharp": "#"}.get(note.get("accidental"), "")
    return [
        NoteToken(
            str(note["digit"]), beats, octave=note.get("octave", 0),
            accidental=accidental, dotted=bool(dots), triplet=tuplet.get("actual") == 3,
            lyrics=note.get("lyrics") or [], beam_lines=levels, dots=dots,
            tuplet_actual=tuplet.get("actual"), tuplet_start=bool(tuplet.get("start")),
            tuplet_stop=bool(tuplet.get("stop")),
            articulation=note.get("articulation"),
        ),
        *(NoteToken("-", 1) for _ in range(max(0, int(note.get("augment", 0))))),
    ]


def beats_of(notes: list[dict]) -> float:
    return sum(token.beats for note in notes for token in note_tokens(note))


def visible_barline(image: Image.Image, row: dict, x: float) -> float:
    boxes = [note["bbox"] for note in row["nums"]]
    top = median(b["y"] for b in boxes)
    glyph_h = max(1, median(b["h"] for b in boxes))
    y_range = range(max(0, round(top - glyph_h * 0.6)), min(image.height, round(top + glyph_h * 1.9)))
    x_range = range(max(0, round(x) - 2), min(image.width, round(x) + 3))
    pixels = image.load()
    return max((sum(pixels[col, y] < 220 for y in y_range) for col in x_range), default=0) / glyph_h


def split_system(
    rows: list[dict], margin: float, expected: float, guide: list[list[Measure]] | None,
    image: Image.Image | None,
):
    voices = [
        sorted(
            (n for n in row["nums"] if center(n) >= margin and n.get("digit") in range(8) and not n.get("grace")),
            key=center,
        )
        for row in rows
    ]
    centers = [[center(n) for n in voice] for voice in voices]
    extent = [x for voice in centers for x in voice]
    if not extent:
        raise ValueError("谱行没有可读音符，请检查裁剪边界。")

    glyph_width = median(n["bbox"]["w"] for voice in voices for n in voice)
    clusters: list[list[tuple[float, int]]] = []
    for x, voice in sorted(
        (x, v) for v, row in enumerate(rows) for x in row["barlineXs"] if min(extent) < x < max(extent)
    ):
        if clusters and x - clusters[-1][0][0] <= max(5.0, glyph_width):
            clusters[-1].append((x, voice))
        else:
            clusters.append([(x, voice)])
    candidates = {min(extent) - 1: 0.0, max(extent) + 1: 0.0}
    strong_boundaries = set()
    for cluster in clusters:
        x = median(x for x, _ in cluster)
        support = len(set(v for _, v in cluster))
        pixels_support = (
            image is not None and min(visible_barline(image, row, x) for row in rows) >= 1.95
        )
        candidates[x] = 0.1 if support == len(rows) else 1.5
        if pixels_support:
            strong_boundaries.add(x)
    target = len(guide[0]) if guide else None
    if target is not None and len(candidates) - 1 < target:
        xs = sorted(set(extent))
        for a, b in sorted(zip(xs, xs[1:]), key=lambda pair: pair[1] - pair[0], reverse=True)[:target]:
            candidates.setdefault((a + b) / 2, 4.0)

    xs = sorted(candidates)
    offsets = [[bisect_left(voice, x) for x in xs] for voice in centers]
    strong_indices = sorted(xs.index(x) for x in strong_boundaries)
    best: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {(0, 0): (0, [])}
    for end in range(1, len(xs)):
        for start in range(end):
            if guide is None and any(start < index < end for index in strong_indices):
                continue
            segments = [voice[o[start]:o[end]] for voice, o in zip(voices, offsets)]
            if not any(segments):
                continue
            rhythm_cost = sum(abs(beats_of(notes) - expected) for notes in segments)
            if guide is None:
                rhythm_cost += sum(not notes for notes in segments) * 2
            for count in range(1, min(end, target if target else end) + 1):
                previous = best.get((start, count - 1))
                if previous is None:
                    continue
                cost = previous[0] + rhythm_cost + candidates[xs[end]]
                if guide is not None:
                    cost += sum(2 for index in strong_indices if start < index < end)
                if guide is not None:
                    for voice, notes in enumerate(segments):
                        reference = guide[voice][count - 1]
                        a = [str(n["digit"]) for n in notes]
                        b = [t.degree for t in reference.all_tokens() if not t.is_dash]
                        cost += 2 * max(len(a), len(b)) * (1 - SequenceMatcher(None, a, b, autojunk=False).ratio())
                if (end, count) not in best or cost < best[end, count][0]:
                    best[end, count] = (cost, previous[1] + [(start, end)])
    paths = [
        value for (end, count), value in best.items()
        if end == len(xs) - 1 and (target is None or count == target)
    ]
    if not paths:
        raise ValueError("无法对齐本行小节线，请核对原图裁边、声部数和参考谱。")
    _, cuts = min(paths, key=lambda value: value[0])
    bars = [[voice[o[a]:o[b]] for a, b in cuts] for voice, o in zip(voices, offsets)]
    bounds = [(xs[a], xs[b]) for a, b in cuts]
    return bars, [xs[b] for _, b in cuts[:-1] if candidates[xs[b]] >= 4.0], bounds


def match_positions(notes: list[dict], measure: Measure) -> dict[int, tuple[int, int]]:
    tokens = [(i, token.degree) for i, token in enumerate(measure.all_tokens()) if not token.is_dash]
    match = SequenceMatcher(None, [str(n["digit"]) for n in notes], [d for _, d in tokens], autojunk=False)
    return {
        a + offset: (measure.number, tokens[b + offset][0])
        for a, b, length in match.get_matching_blocks() for offset in range(length)
    }


def note_durations(measure: Measure) -> dict[int, float]:
    values = {}
    previous = None
    for index, token in enumerate(measure.all_tokens()):
        if token.is_dash and previous is not None:
            values[previous] += token.beats
        elif not token.is_dash:
            values[index] = token.beats
            previous = index
    return values


def attach_arcs(part: Part, found: list, key: str) -> list[dict]:
    _, tonic = parse_key_tonic_midi(key)
    pitches = {}
    for measure in part.measures:
        if measure.key:
            _, tonic = parse_key_tonic_midi(measure.key)
        for idx, token in enumerate(measure.all_tokens()):
            if not token.is_dash:
                pitches[measure.number, idx] = token.to_midi(tonic)
    adjacent = set(zip(pitches, list(pitches)[1:]))
    slurs: dict[int, deque] = {}
    previous_tie = None
    pending = []
    arcs = list(part.arcs)
    for note, end, page, bar in found:
        location = {"page": page, "measure": bar, "bbox": note["bbox"]}
        if note.get("tieStop"):
            start = previous_tie[0] if previous_tie else None
            if (start, end) in adjacent and pitches[end] is not None and pitches[end] == pitches[start]:
                arcs.append(Arc("tie", *start, *end))
            else:
                pending.append({**location, "mark": "延音线终点"})
        elif previous_tie is not None:
            pending.append({**previous_tie[1], "mark": "延音线起点"})
        previous_tie = (end, location) if note.get("tieStart") else None
        if tag := note.get("slurStop"):
            queue = slurs.setdefault(tag, deque())
            start = queue.popleft()[0] if queue else None
            if start and end and start < end and pitches.get(start) is not None and pitches.get(end) is not None:
                arcs.append(Arc("slur", *start, *end))
            else:
                pending.append({**location, "mark": "圆滑线终点"})
        if tag := note.get("slurStart"):
            slurs.setdefault(tag, deque()).append((end, location))
    part.arcs = list(dict.fromkeys(arcs))
    pending.extend({**loc, "mark": "圆滑线起点"} for queue in slurs.values() for _, loc in queue)
    if previous_tie:
        pending.append({**previous_tie[1], "mark": "延音线起点"})
    return pending


def beat_events(tokens, boxes=None):
    events = []
    for index, (onset, duration, token) in enumerate(timed_tokens(tokens)):
        if token.is_dash and events:
            old_start, old_duration, old_token, bbox = events[-1]
            events[-1] = old_start, old_duration + duration, old_token, bbox
        else:
            events.append((onset, duration, token, boxes[index] if boxes is not None else None))
    return events


def beat_slice(events, beat):
    segments = []
    for start, duration, token, bbox in events:
        left, right = max(start, beat), min(start + duration, beat + 1)
        if left >= right:
            continue
        segment = {
            "at": str(left - beat),
            "length": str(right - left),
            "degree": token.degree,
            "octave": token.octave,
            "accidental": token.accidental,
            "continued": start < beat,
        }
        if bbox is not None:
            segment["bbox"] = bbox
        segments.append(segment)
    return segments


def compare_quarter_beats(notes, expected, reference=None):
    tokens, boxes = [], []
    for note in notes:
        expanded = note_tokens(note)
        tokens.extend(expanded)
        boxes.extend([note["bbox"]] * len(expanded))
    observed_events = beat_events(tokens, boxes)
    reference_events = beat_events(reference.all_tokens()) if reference is not None else None
    actual = math.fsum(token.beats for token in tokens)
    for beat in range(math.ceil(max(expected, actual) - 1e-7)):
        observed = beat_slice(observed_events, beat)
        wanted = beat_slice(reference_events, beat) if reference_events is not None else None
        heard = [{k: v for k, v in segment.items() if k != "bbox"} for segment in observed]
        covered = math.fsum(float(Fraction(segment["length"])) for segment in observed)
        target = max(0, min(1, expected - beat))
        if abs(covered - target) > 0.01 or (wanted is not None and heard != wanted):
            yield {
                "quarter": beat + 1, "overflow": target == 0,
                "recognized": observed, "reference": wanted,
            }


def import_pages(
    pages: list[dict], *, voices: int, margin: float, title: str | None, time_sig: str | None,
    key: str | None, bpm: float | None, reference: Score | None = None,
    fragment_refine: bool = True, fragment_dir: Path | None = None,
) -> tuple[Score, dict]:
    if not pages or any(page.get("kind") != "jianpu" or "score" not in page for page in pages):
        raise ValueError("至少一页识别失败或不是简谱。")
    first = pages[0]["score"]
    score = deepcopy(reference) if reference else Score(
        title=title or first.get("title") or "图片识谱",
        key=key or first.get("key") or "C",
        time_sig=time_sig or f'{first["beats"]}/{first["beatType"]}',
        bpm=bpm or first.get("tempo") or 100,
        parts=[Part(*name, []) for name in [("主声", "Lead"), ("和声", "Harmony")][:voices]],
        system_breaks=[],
    )
    if title:
        score.title = title
    if time_sig:
        score.time_sig = time_sig
    if key:
        score.key = key
    if bpm is not None:
        score.bpm = bpm
    if not math.isfinite(score.bpm) or score.bpm <= 0:
        raise ValueError("速度必须大于零。")
    score.key = parse_key_tonic_midi(score.key)[0]
    if reference and (len(score.parts) != voices or not score.system_breaks):
        raise ValueError("参考谱必须有相同的声部数，以及原谱逐行的小节编号。")
    ref_parts = [{m.number: m for m in part.measures} for part in score.parts]
    arc_inputs: list[list] = [[] for _ in range(voices)]
    frag_summary = {
        "enabled": bool(fragment_refine and not reference),
        "div_refined": 0,
        "digit_refined": 0,
        "missed_notes_recovered": 0,
        "bars_rhythm_repaired": 0,
        "bars_theory_normalized": 0,
    }
    review: dict = {
        "mode": "reference" if reference else "fragment_recognition" if fragment_refine else "recognition",
        "reference_supplied": bool(reference),
        "score_meter": score.time_sig, "score_key": score.key,
        "fragment_refinement": frag_summary,
        "pages": [], "measures": [], "quarter_discrepancies": [], "unmatched_arc_markers": {},
    }
    system, next_bar = 0, 1
    for page_idx, page in enumerate(pages, 1):
        data = page["score"]
        picture = None
        gray = None
        try:
            with Image.open(page["file"]) as original:
                upright = ImageOps.exif_transpose(original)
                if upright.width <= 4500:
                    picture = upright.convert("L")
                    gray = np.asarray(picture)
        except (OSError, TypeError):
            pass
        rows = sorted((r for r in data["rows"] if r.get("nums")), key=lambda r: r["topY"])
        if len(rows) % voices:
            raise ValueError(f"第 {page_idx} 页谱行无法按 {voices} 个声部分组，请检查并裁掉页眉杂行。")
        templates = {}
        if fragment_refine and not reference and gray is not None:
            templates = build_page_digit_templates(gray, rows, margin)
            refined_rows = []
            for r in rows:
                rr, st = refine_row_fragments(gray, r, templates, margin)
                frag_summary["div_refined"] += st["div_changed"]
                frag_summary["digit_refined"] += st["digit_changed"]
                refined_rows.append(rr)
            rows = refined_rows
        review["pages"].append({
            "page": page_idx, "file": page["file"],
            "detected_meter": f'{data["beats"]}/{data["beatType"]}',
            "detected_key": data.get("key"), "barlines_checked_against_image": picture is not None,
            "digit_templates_built": sorted(templates.keys()),
        })
        for offset in range(0, len(rows), voices):
            source_rows = rows[offset:offset + voices]
            if reference:
                if system >= len(score.system_breaks):
                    raise ValueError("原图谱行多于参考谱，请检查页序以及声部数。")
                bar_numbers = score.system_breaks[system]
                guide = [[p[n] for n in bar_numbers] for p in ref_parts]
            else:
                guide = None
            bars, inferred, bar_bounds = split_system(source_rows, margin, score.beats_per_measure, guide, picture)
            if not reference:
                bar_numbers = list(range(next_bar, next_bar + len(bars[0])))
                score.system_breaks.append(bar_numbers)
            for v, voice in enumerate(bars):
                row_valid = [
                    n for n in source_rows[v]["nums"]
                    if center(n) >= margin and n.get("digit") in range(8) and not n.get("grace")
                ]
                row_top = float(median(n["bbox"]["y"] for n in row_valid)) if row_valid else float(source_rows[v]["topY"])
                glyph_h = float(median(n["bbox"]["h"] for n in row_valid)) if row_valid else 18.0
                glyph_w = float(median(n["bbox"]["w"] for n in row_valid)) if row_valid else 11.0
                for bar_i, (number, notes) in enumerate(zip(bar_numbers, voice)):
                    expected = score.beats_per_measure
                    if reference:
                        measure = ref_parts[v][number]
                        expected = measure.expected_beats or score.beats_per_measure
                        positions = match_positions(notes, measure)
                    else:
                        if fragment_refine and gray is not None:
                            cur_b = beats_of(notes)
                            if expected - cur_b >= 0.20:
                                b_left, b_right = bar_bounds[bar_i]
                                notes, n_added = recover_missed_bar_fragments(
                                    gray, notes, b_left, b_right, row_top, glyph_h, glyph_w,
                                    templates, expected, cur_b,
                                )
                                frag_summary["missed_notes_recovered"] += n_added
                            notes, repaired = solve_bar_rhythm_constraints(notes, expected)
                            if repaired:
                                frag_summary["bars_rhythm_repaired"] += 1
                        tokens, positions = [], {}
                        for idx, note in enumerate(notes):
                            positions[idx] = (number, len(tokens))
                            tokens.extend(note_tokens(note))
                        measure = Measure(number, beat_groups(tokens))
                        score.parts[v].measures.append(measure)
                    arc_inputs[v].extend(
                        (note, positions.get(idx), page_idx, number) for idx, note in enumerate(notes)
                    )
                    durations = note_durations(measure)
                    rhythm_regions = [
                        notes[idx]["bbox"] for idx, (_, token) in positions.items()
                        if abs(beats_of([notes[idx]]) - durations[token]) > 0.01
                    ] if reference else []
                    observed = beats_of(notes)
                    check_context = {"page": page_idx, "measure": number, "voice": score.parts[v].name}
                    review["quarter_discrepancies"].extend({
                        **check_context, **beat,
                    } for beat in compare_quarter_beats(notes, expected, measure if reference else None))
                    expected_degrees = [t.degree for t in measure.all_tokens() if not t.is_dash]
                    differences = (
                        abs(observed - expected) > 0.01 or len(positions) != len(expected_degrees)
                        or len(positions) != len(notes) or inferred or rhythm_regions
                        or any(note.get("doubt") for note in notes)
                    )
                    if differences:
                        if fragment_dir is not None and gray is not None:
                            save_fragment_diagnostic_sheet(
                                gray, notes,
                                fragment_dir / f"p{page_idx}_m{number:02d}_{score.parts[v].name}.png",
                            )
                        review["measures"].append({
                            "page": page_idx, "system": system + 1, "measure": number,
                            "voice": score.parts[v].name, "recognized_quarters": round(observed, 4),
                            "expected_quarters": expected, "inferred_barlines": inferred,
                            "regions": [n["bbox"] for n in notes if n.get("doubt") or abs(observed - expected) > 0.01],
                            "rhythm_regions": rhythm_regions,
                        })
            system += 1
            next_bar = bar_numbers[-1] + 1
    if reference and system != len(score.system_breaks):
        raise ValueError("原图谱行少于参考谱，请检查是否漏页。")
    for part, found in zip(score.parts, arc_inputs):
        pending = attach_arcs(part, found, score.key)
        if pending:
            review["unmatched_arc_markers"][part.name] = pending
    if fragment_refine and not reference:
        for part in score.parts:
            arc_measures = {a.start_measure for a in part.arcs} | {a.end_measure for a in part.arcs}
            new_measures = []
            for m in part.measures:
                if m.number not in arc_measures and abs(m.total_beats - score.beats_per_measure) <= 0.01:
                    nm = normalize_measure_rhythm(m, score.time_sig)
                    if len(nm.all_tokens()) != len(m.all_tokens()) or len(nm.beats) != len(m.beats):
                        frag_summary["bars_theory_normalized"] += 1
                    new_measures.append(nm)
                else:
                    new_measures.append(m)
            part.measures = new_measures
    review["score_violations"] = score.validate()
    return score, review

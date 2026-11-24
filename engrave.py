#!/usr/bin/env python3
"""
engrave.py - Publisher-Style Portrait A4 Jianpu PDF Engraver (JP-Word / Classic Choral Style).

Visual conventions matching published Chinese Jianpu sheet music:
  - Page 1 clean title header (title centered, key + stacked time signature left, composer right)
  - Boxed measure number ([1], [5], [8]...) only at the start of each system + part name on left
  - Exact system line breaks when `score.system_breaks` is provided (defaults to 4 bars/row)
  - Authentic L-shaped volta brackets (`1.` / `2.`), modulation headers (`转1=B`), repeat dots
  - Locked uniform font sizes across the entire score (big rounded Jianpu digits + large CJK lyrics)
  - True continuous beat beaming (小切分、前八后十六、前十六后八、附点八分+十六分)
  - Clean centered page numbers (`1/3`, `2/3`, `3/3`)
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from domain import BeatGroup, Measure, NoteToken, Part, Score, parse_key_tonic_midi, timed_tokens

# Locked typography constants (single source of truth; uniform across every bar & rest '0')
PDF_NUM_FONT_SIZE = 13.2
PDF_LYRIC_SIZE = 10.2
AUGMENT_DASH_WIDTH = 11.0
BAR_PADDING = 23.0
BEAT_GAP = 4.0
USABLE_WIDTH = A4[0] - 68.0 - 44.0


def register_fonts() -> Tuple[str, str]:
    """Resolve CJK + rounded-digit fonts.

    Env overrides first (SCORE_TOOL_CJK_FONT / SCORE_TOOL_ROUND_FONT, file path),
    then macOS system fonts, then common Linux fallbacks. Falls back to
    Helvetica / Helvetica-Bold when nothing is found.
    """
    cjk_name = "Helvetica"
    cjk_candidates = []
    cjk_env = os.environ.get("SCORE_TOOL_CJK_FONT", "").strip()
    if cjk_env:
        cjk_candidates.append((cjk_env, None))
    cjk_candidates += [
        ("/System/Library/Fonts/Supplemental/Songti.ttc", 0),
        ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
        ("/System/Library/Fonts/PingFang.ttc", 0),
        ("/Library/Fonts/Arial Unicode.ttf", None),
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", None),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", None),
    ]
    for path, sub in cjk_candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("CJKFont", path, subfontIndex=sub) if sub is not None else TTFont("CJKFont", path))
                cjk_name = "CJKFont"
                break
            except Exception:
                continue

    num_name = "Helvetica-Bold"
    round_candidates = []
    round_env = os.environ.get("SCORE_TOOL_ROUND_FONT", "").strip()
    if round_env:
        round_candidates.append(round_env)
    round_candidates += [
        "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for round_path in round_candidates:
        if round_path and os.path.exists(round_path):
            try:
                pdfmetrics.registerFont(TTFont("RoundNum", round_path))
                num_name = "RoundNum"
                break
            except Exception:
                continue
    return cjk_name, num_name


def modulation_banner_text(badge_note: str, old_key: str = "") -> str:
    """Build a modulation banner like '转 1=B（前1=后7）' from a badge + old key.

    The badge (e.g. '转1=B') gives the new key; the old tonic's scale degree
    inside the new key is computed from the semitone difference. Falls back to
    the badge text unchanged when it cannot be parsed.
    """
    m = re.search(r"1\s*=\s*([A-Ga-g][#b♯♭]?)", badge_note or "")
    if not m or not old_key:
        return badge_note
    try:
        new_label, _ = parse_key_tonic_midi(m.group(1))
        _, old_tonic = parse_key_tonic_midi(old_key)
        _, new_tonic = parse_key_tonic_midi(new_label)
    except ValueError:
        return badge_note
    diff = (old_tonic - new_tonic) % 12
    degree = {0: "1", 2: "2", 4: "3", 5: "4", 7: "5", 9: "6", 11: "7"}.get(diff)
    if degree is None:
        return badge_note
    return f"转 1={new_label}（前1=后{degree}）"


def token_width(token: NoteToken, lyric_font: str) -> float:
    lyric_width = max(
        (pdfmetrics.stringWidth(text, lyric_font, PDF_LYRIC_SIZE) for text in token.lyrics[:2]),
        default=0.0,
    )
    glyph = AUGMENT_DASH_WIDTH + 4 if token.is_dash else (17.0 if token.accidental_mark else 12.8)
    return max(glyph, lyric_width + 4.0)


def group_width(group: BeatGroup, lyric_font: str) -> float:
    return sum(token_width(token, lyric_font) for token in group.tokens)


def measure_width(measure: Measure, lyric_font: str) -> float:
    return BAR_PADDING + BEAT_GAP * max(0, len(measure.beats) - 1) + sum(
        group_width(group, lyric_font) for group in measure.beats
    )


def compute_beat_xs(bx: float, bw: float, group: BeatGroup, lyric_font: str) -> Tuple[List[float], List[float]]:
    minimums = [token_width(token, lyric_font) for token in group.tokens]
    if not minimums:
        return [], []
    extra_per_note = max(0.0, bw - sum(minimums)) / len(minimums)
    widths = [m + extra_per_note for m in minimums]
    xs = []
    for width in widths:
        xs.append(bx + width / 2)
        bx += width
    return xs, widths


def beam_runs(measure: Measure) -> List[List[int]]:
    runs: List[List[int]] = []
    previous_beat = None
    for index, (onset, duration, token) in enumerate(timed_tokens(measure.all_tokens())):
        beat = int(onset)
        if not token.underlines:
            previous_beat = None
            continue
        crosses_beat = onset + duration > beat + 1
        if previous_beat != beat or crosses_beat:
            runs.append([])
        runs[-1].append(index)
        previous_beat = None if crosses_beat else beat
    return runs


def draw_measure_beams(
    c: canvas.Canvas, measure: Measure, xs: List[float], widths: List[float], y_base: float,
) -> None:
    levels = [token.underlines for token in measure.all_tokens()]
    for run in beam_runs(measure):
        caps = []
        for index in run:
            gaps = [
                abs(xs[index] - xs[neighbor]) for neighbor in (index - 1, index + 1)
                if 0 <= neighbor < len(xs)
            ]
            clearance = max(0.0, (min(gaps, default=widths[index]) - 2.4) / 0.88)
            caps.append(min(widths[index], clearance))
        draw_beams_pdf(c, [xs[i] for i in run], caps, [levels[i] for i in run], y_base)


def draw_beams_pdf(c: canvas.Canvas, xs: List[float], widths: List[float], levels: List[int], y_base: float) -> None:
    c.saveState()
    c.setStrokeColorRGB(0.03, 0.03, 0.03)
    c.setLineWidth(1.0)
    c.setLineCap(1)
    for lvl in range(1, max(levels, default=0) + 1):
        y_off = -4.2 - (lvl - 1) * 4.6
        i, n = 0, len(xs)
        while i < n:
            if levels[i] >= lvl:
                j = i
                while j + 1 < n and levels[j + 1] >= lvl:
                    j += 1
                pl = min(4.6, widths[i] * 0.44)
                pr = min(4.6, widths[j] * 0.44)
                c.line(xs[i] - pl, y_base + y_off, xs[j] + pr, y_base + y_off)
                i = j + 1
            else:
                i += 1
    c.restoreState()


def build_system_rows(score: Score, part: Part, widths: Dict[int, float]) -> List[List[Measure]]:
    by_number = {m.number: m for m in part.measures}
    if score.system_breaks:
        preferred = list(score.system_breaks)
        scheduled = {n for row in preferred for n in row}
        preferred = preferred + [[m.number] for m in part.measures if m.number not in scheduled]
        rows = []
        for source in preferred:
            row, occupied = [], 0.0
            for number in source:
                if number not in by_number:
                    continue
                width = widths[number]
                if width > USABLE_WIDTH:
                    raise ValueError(f"第 {number} 小节歌词超出 A4 行宽，请核对音节切分。")
                if row and occupied + width > USABLE_WIDTH:
                    rows.append(row)
                    row, occupied = [], 0.0
                row.append(by_number[number])
                occupied += width
            if row:
                rows.append(row)
        return rows

    # Automatic greedy packing (max 4 bars/row) to prevent ugly 1-bar orphan rows
    rows, row, occupied = [], [], 0.0
    for m in part.measures:
        width = widths[m.number]
        if width > USABLE_WIDTH:
            raise ValueError(f"第 {m.number} 小节歌词超出 A4 行宽，请核对音节切分。")
        if row and (len(row) >= 4 or occupied + width > USABLE_WIDTH):
            rows.append(row)
            row, occupied = [], 0.0
        row.append(m)
        occupied += width
    if row:
        rows.append(row)
    return rows


def draw_tuplet(c: canvas.Canvas, start: float, end: float, y: float, number: int, font: str) -> None:
    mid = (start + end) / 2
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    c.setLineWidth(0.8)
    c.line(start, y - 2.5, start, y)
    c.line(start, y, mid - 5, y)
    c.line(mid + 5, y, end, y)
    c.line(end, y, end, y - 2.5)
    c.setFont(font, 8.2)
    c.drawCentredString(mid, y - 2.6, str(number))


def draw_boxed_bar_number(c: canvas.Canvas, x: float, y: float, num: int, font_name: str) -> None:
    txt = str(num)
    w = 11.0 if len(txt) == 1 else 14.5
    h = 10.5
    c.saveState()
    c.setStrokeColorRGB(0.15, 0.15, 0.15)
    c.setLineWidth(0.65)
    c.rect(x - 1.5, y - 2.0, w, h, stroke=1, fill=0)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.setFont(font_name, 7.8)
    c.drawCentredString(x - 1.5 + w / 2.0, y + 0.6, txt)
    c.restoreState()


def draw_jianpu_token(
    c: canvas.Canvas, cx: float, y_base: float, t: NoteToken, num_font: str,
    num_font_size: float = PDF_NUM_FONT_SIZE,
) -> None:
    c.setFillColorRGB(0, 0, 0)
    if t.is_dash:
        c.saveState()
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(1.2)
        half = AUGMENT_DASH_WIDTH / 2
        c.line(cx - half, y_base + 4, cx + half, y_base + 4)
        c.restoreState()
        return
    sign = t.accidental_mark
    if sign == "♮":
        c.saveState()
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(0.68)
        lx, rx, base = cx - 8.2, cx - 5.2, y_base + 2
        c.line(lx, base + 2, lx, base + 11)
        c.line(rx, base, rx, base + 9)
        c.line(lx, base + 6, rx, base + 7)
        c.line(lx, base + 3, rx, base + 4)
        c.restoreState()
    elif sign:
        c.setFont(num_font, 8.2)
        c.drawString(cx - 8.2, y_base + 6.2, sign)
    c.setFont(num_font, num_font_size)
    c.drawCentredString(cx, y_base, t.degree)
    dots = int(t.dotted) if t.dots is None else t.dots
    for dot in range(dots):
        c.circle(cx + num_font_size * 0.44 + 4 * dot, y_base + 3.4, 1.45, fill=1, stroke=0)
    if t.octave > 0:
        for d in range(min(2, t.octave)):
            c.circle(cx, y_base + num_font_size + 2.8 + d * 4.2, 1.55, fill=1, stroke=0)
    if t.octave < 0:
        base_y = y_base - 9.0 - t.underlines * 3.0
        for d in range(min(2, -t.octave)):
            c.circle(cx, base_y - d * 4.2, 1.55, fill=1, stroke=0)
    if t.articulation == "staccato":
        c.circle(cx, y_base + 22 + 4.2 * min(2, max(0, t.octave)), 1.4, fill=1, stroke=0)
    elif t.articulation == "staccatissimo":
        tip = y_base + 33
        mark = c.beginPath()
        mark.moveTo(cx, tip)
        mark.lineTo(cx - 2.6, tip + 5)
        mark.lineTo(cx + 2.6, tip + 5)
        mark.close()
        c.drawPath(mark, fill=1, stroke=0)


def draw_part_pdf_pages(
    c: canvas.Canvas, score: Score, part: Part, cjk_font: str, num_font: str, draft: bool = False,
) -> None:
    page_w, page_h = A4
    widths: Dict[int, float] = {}
    for voice in score.parts:
        for measure in voice.measures:
            widths[measure.number] = max(widths.get(measure.number, 0.0), measure_width(measure, cjk_font))
    margin_left = 68.0   # Room for part name ('主声'/'和声') on left like original score!
    margin_right = 44.0
    usable_w = page_w - margin_left - margin_right  # 483.3 pt

    rows = build_system_rows(score, part, widths)
    row_for_measure = {measure.number: i for i, row in enumerate(rows) for measure in row}
    active_tuplet = None
    rows_per_page = max(1, score.rows_per_page)
    total_pages = max(1, math.ceil(len(rows) / rows_per_page))
    key_ascii = score.key.replace("♭", "b").replace("♯", "#")
    ts_num, ts_den = score.time_sig.split("/")
    last_bar_num = part.measures[-1].number if part.measures else 0

    for p_idx in range(total_pages):
        # Page 1 gets the classic clean title & metadata header (matching original 3-page score)
        if p_idx == 0:
            c.setFillColorRGB(0, 0, 0)
            c.setFont(cjk_font, 23)
            title_str = f"{score.title}（{part.name}）" if len(score.parts) > 1 else score.title
            c.drawCentredString(page_w / 2.0, page_h - 62, title_str)
            if score.engraver:
                c.setFont(cjk_font, 10.5)
                c.drawCentredString(page_w / 2.0, page_h - 79, score.engraver)

            # Key + stacked time signature on left
            c.setFont(num_font, 11.0)
            c.drawString(margin_left, page_h - 94, f"1={key_ascii}")
            tx = margin_left + 36.0
            c.setFont(num_font, 9.5)
            c.drawCentredString(tx, page_h - 89, ts_num)
            c.setLineWidth(0.75)
            c.line(tx - 5.0, page_h - 92, tx + 5.0, page_h - 92)
            c.drawCentredString(tx, page_h - 101, ts_den)

            c.setFont(cjk_font, 9.5)
            tempo_txt = f"速度={round(score.bpm)}" + (f"  {score.mood}" if score.mood else "")
            c.drawString(margin_left, page_h - 114, tempo_txt)
            if score.composer:
                c.drawRightString(page_w - margin_right, page_h - 94, score.composer)

            y_start = page_h - 168.0
            row_h = 128.0
        else:
            y_start = page_h - 96.0
            row_h = 142.0

        # Centered classic page number footer (1/3, 2/3, 3/3)
        c.setFont(num_font, 9.0)
        c.setFillColorRGB(0.25, 0.25, 0.25)
        c.drawCentredString(page_w / 2.0, 34, f"{p_idx + 1}/{total_pages}")
        if draft:
            c.setFont(cjk_font, 9.0)
            c.drawString(margin_left, 34, "识谱复核样张")

        page_rows = rows[p_idx * rows_per_page : (p_idx + 1) * rows_per_page]

        for r_i, row_bars in enumerate(page_rows):
            y_base = y_start - r_i * row_h

            # Check if first bar of this system has a modulation banner (e.g. 转1=B)
            first_m = row_bars[0]
            if first_m.badge_note:
                c.setFont(cjk_font, 11.5)
                c.setFillColorRGB(0, 0, 0)
                mod_txt = modulation_banner_text(first_m.badge_note, score.key)
                c.drawString(margin_left - 2.0, y_base + 42.0, mod_txt)

            # Boxed measure number at system start + part name on left margin
            draw_boxed_bar_number(c, margin_left - 26.0, y_base + 24.0, first_m.number, num_font)
            c.setFont(cjk_font, 10.5)
            c.setFillColorRGB(0.12, 0.12, 0.12)
            c.drawRightString(margin_left - 12.0, y_base + 1.0, part.name)

            minimums = [widths[m.number] for m in row_bars]
            extra = (usable_w - sum(minimums)) / len(minimums)
            bar_widths = [minimum + extra for minimum in minimums]
            anchors = {}
            row_index = p_idx * rows_per_page + r_i

            # Thick opening system barline
            c.setStrokeColorRGB(0.05, 0.05, 0.05)
            c.setLineWidth(1.6)
            c.line(margin_left, y_base - 11, margin_left, y_base + 18)

            cur_x = margin_left
            for m, bar_w in zip(row_bars, bar_widths):
                token_index = 0
                x0, x1 = cur_x, cur_x + bar_w
                cur_x = x1

                # Repeat start ||: on left of bar if applicable
                if m.repeat_start and m.number != first_m.number:
                    c.saveState()
                    c.setStrokeColorRGB(0.05, 0.05, 0.05)
                    c.setLineWidth(2.4)
                    c.line(x0, y_base - 11, x0, y_base + 18)
                    c.setLineWidth(0.9)
                    c.line(x0 + 3.4, y_base - 11, x0 + 3.4, y_base + 18)
                    c.setFillColorRGB(0.05, 0.05, 0.05)
                    c.circle(x0 + 6.8, y_base + 6.5, 1.4, fill=1, stroke=0)
                    c.circle(x0 + 6.8, y_base - 0.5, 1.4, fill=1, stroke=0)
                    c.restoreState()

                # Volta bracket (1. or 2.) above measure
                if m.ending:
                    vy = y_base + 57.0
                    c.saveState()
                    c.setStrokeColorRGB(0.1, 0.1, 0.1)
                    c.setLineWidth(0.75)
                    c.line(x0 + 1.0, vy - 7.0, x0 + 1.0, vy)
                    c.line(x0 + 1.0, vy, x1 - 1.0, vy)
                    if m.ending == "1.":
                        c.line(x1 - 1.0, vy, x1 - 1.0, vy - 7.0)
                    c.setFont(num_font, 8.2)
                    c.setFillColorRGB(0.1, 0.1, 0.1)
                    label_y = vy + 2.5 if m.badge_note else vy - 7.5
                    c.drawString(x0 + 4.5, label_y, m.ending)
                    c.restoreState()

                # Bar chord symbols & optional section badge above melody
                if m.chords:
                    c.saveState()
                    c.setFont(num_font, 9.0)
                    c.setFillColorRGB(0.14, 0.34, 0.68)
                    chord_step = (bar_w - 16.0) / max(1, len(m.chords))
                    for ch_i, ch_txt in enumerate(m.chords[:4]):
                        if ch_txt:
                            c.drawString(x0 + 8.0 + ch_i * chord_step, y_base + 27.5, ch_txt)
                    c.restoreState()
                if m.section:
                    c.saveState()
                    c.setFont(cjk_font, 8.2)
                    c.setFillColorRGB(0.35, 0.35, 0.35)
                    c.drawString(x0 + 6.0, y_base + 40.0, f"[{m.section}]")
                    c.restoreState()

                n_beats = len(m.beats)
                pad_l = 12.0 if (m.repeat_start and m.number != first_m.number) else 7.5
                pad_r = 10.0 if m.repeat_end else 7.5
                inner_w = bar_w - pad_l - pad_r - BEAT_GAP * max(0, n_beats - 1)
                beat_minimums = [group_width(bg, cjk_font) for bg in m.beats]
                extra_per_beat = (inner_w - sum(beat_minimums)) / max(1, n_beats)
                bx = x0 + pad_l
                measure_xs, measure_widths = [], []
                for bg, minimum in zip(m.beats, beat_minimums):
                    bw = minimum + extra_per_beat
                    xs, slot_ws = compute_beat_xs(bx, bw, bg, cjk_font)
                    bx += bw + BEAT_GAP
                    measure_xs.extend(xs)
                    measure_widths.extend(slot_ws)

                    for cx, t in zip(xs, bg.tokens):
                        anchors[m.number, token_index] = cx
                        token_index += 1
                        if t.tuplet_start:
                            active_tuplet = (row_index, cx, t.tuplet_actual)
                        if t.tuplet_stop and active_tuplet and active_tuplet[0] == row_index:
                            draw_tuplet(c, active_tuplet[1] - 4, cx + 4, y_base + PDF_NUM_FONT_SIZE + 8.5,
                                        active_tuplet[2] or 3, num_font)
                            active_tuplet = None
                        draw_jianpu_token(c, cx, y_base, t, num_font)

                        # Spacious, large, uniform lyrics directly under each note!
                        for v_idx, syl in enumerate(t.lyrics[:2]):
                            if syl:
                                c.setFont(cjk_font, PDF_LYRIC_SIZE)
                                c.setFillColorRGB(0, 0, 0) if v_idx == 0 else c.setFillColorRGB(0.18, 0.18, 0.18)
                                c.drawCentredString(cx, y_base - (27.0 if v_idx == 0 else 43.0), syl)

                    if xs and any(t.triplet for t in bg.tokens) and not any(t.tuplet_actual for t in bg.tokens):
                        draw_tuplet(c, xs[0] - 4, xs[-1] + 4, y_base + PDF_NUM_FONT_SIZE + 8.5, 3, num_font)

                draw_measure_beams(c, m, measure_xs, measure_widths, y_base)

                # Right barline
                c.setStrokeColorRGB(0.05, 0.05, 0.05)
                if m.number == last_bar_num:
                    c.setLineWidth(0.95)
                    c.line(x1 - 3.6, y_base - 11, x1 - 3.6, y_base + 18)
                    c.setLineWidth(2.8)
                    c.line(x1, y_base - 11, x1, y_base + 18)
                elif m.repeat_end:
                    c.setLineWidth(0.95)
                    c.line(x1 - 3.6, y_base - 11, x1 - 3.6, y_base + 18)
                    c.setLineWidth(2.6)
                    c.line(x1, y_base - 11, x1, y_base + 18)
                    c.setFillColorRGB(0.05, 0.05, 0.05)
                    c.circle(x1 - 7.0, y_base + 6.5, 1.4, fill=1, stroke=0)
                    c.circle(x1 - 7.0, y_base - 0.5, 1.4, fill=1, stroke=0)
                else:
                    c.setLineWidth(0.95)
                    c.line(x1, y_base - 11, x1, y_base + 18)

            c.saveState()
            c.setStrokeColorRGB(0.08, 0.08, 0.08)
            c.setLineWidth(0.7)
            for arc in part.arcs:
                start_row = row_for_measure.get(arc.start_measure, -1)
                end_row = row_for_measure.get(arc.end_measure, -1)
                if not start_row <= row_index <= end_row:
                    continue
                left = anchors.get(arc.start, margin_left)
                right = anchors.get(arc.end, page_w - margin_right)
                if right - left < 4:
                    continue
                y = y_base + (23 if arc.kind == "tie" else 34)
                rise = min(13 if arc.kind == "slur" else 6, (right - left) / 5)
                c.bezier(left + 1, y, left + (right - left) / 4, y + rise,
                         right - (right - left) / 4, y + rise, right - 1, y)
            c.restoreState()

        c.showPage()


def engrave_score_pdfs(score: Score, out_dir: Path, *, draft: bool = False) -> Dict[str, str]:
    """Generate individual print-ready Portrait A4 PDFs + combined PDF (if multi-part)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cjk_font, num_font = register_fonts()
    stem = score.title.replace(" ", "_")
    outputs: Dict[str, str] = {}

    for part in score.parts:
        fname = f"{stem}-{part.name}分谱-打印版.pdf" if len(score.parts) > 1 else f"{stem}-简谱打印版.pdf"
        out_pdf = out_dir / fname
        c = canvas.Canvas(str(out_pdf), pagesize=A4)
        draw_part_pdf_pages(c, score, part, cjk_font, num_font, draft)
        c.save()
        outputs[f"{part.name}_pdf"] = str(out_pdf)

    if len(score.parts) > 1:
        combined = out_dir / f"{stem}-打印版-全声部合订.pdf"
        c_all = canvas.Canvas(str(combined), pagesize=A4)
        for part in score.parts:
            draw_part_pdf_pages(c_all, score, part, cjk_font, num_font, draft)
        c_all.save()
        outputs["combined_pdf"] = str(combined)

    return outputs

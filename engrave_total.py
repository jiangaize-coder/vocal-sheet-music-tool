"""A4 two-voice Jianpu with the source's explicit systems and pagination."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from domain import Measure, Score, timed_tokens
from engrave import (
    AUGMENT_DASH_WIDTH, PDF_LYRIC_SIZE, PDF_NUM_FONT_SIZE, draw_boxed_bar_number, draw_jianpu_token,
    draw_measure_beams, draw_tuplet, register_fonts,
)

LEFT = 52.0
RIGHT = 32.0
INNER = A4[0] - LEFT - RIGHT
BAR_INSET = 6.0
BEAT_GAP = 2.3
VOICE_GAP = 68.0


def system_plan(rows, lyric_font: str, lyric_size: float = PDF_LYRIC_SIZE, num_size: float = PDF_NUM_FONT_SIZE):
    """Reserve horizontal space once per onset, shared by both voices."""
    scale = num_size / PDF_NUM_FONT_SIZE
    columns = []
    for upper, lower in zip(*rows):
        widths = {}
        for measure in (upper, lower):
            for beat, _, token in timed_tokens(measure.all_tokens()):
                lyric = max((stringWidth(t, lyric_font, lyric_size) for t in token.lyrics[:2]), default=0)
                dots = int(token.dotted) if token.dots is None else token.dots
                glyph = AUGMENT_DASH_WIDTH + 4 if token.is_dash else (15.0 * scale if token.accidental else 10.5 * scale)
                minimum = max(glyph + 3.5 * max(dots - 1, 0), lyric + 2.2)
                widths[beat] = max(widths.get(beat, 0), minimum)
        columns.append(sorted(widths.items()))
    beat_gaps = [
        sum(int(previous) != int(current) for (previous, _), (current, _) in zip(bar, bar[1:]))
        for bar in columns
    ]
    required = sum(
        2 * BAR_INSET + sum(width for _, width in bar) + gaps * BEAT_GAP
        for bar, gaps in zip(columns, beat_gaps)
    )
    if required > INNER:
        raise ValueError(f"第 {rows[0][0].number} 小节起的原谱行超出 A4 可排宽度。")
    extra = (INNER - required) / len(columns)
    x = LEFT
    plan = []
    for column in columns:
        left, filled = x, {}
        x += BAR_INSET
        extra_per_slot = extra / max(1, len(column))
        previous = None
        for time, minimum in column:
            if previous is not None and int(previous) != int(time):
                x += BEAT_GAP
            width = minimum + extra_per_slot
            filled[time] = (x + width / 2, width)
            x += width
            previous = time
        x += BAR_INSET + (extra if not column else 0)
        plan.append((left, x, filled))
    return plan


def draw_bar_edges(c: Canvas, measure: Measure, left: float, right: float, y: float, final: bool):
    c.setStrokeColorRGB(0.05, 0.05, 0.05)
    c.setLineWidth(0.7)
    c.line(right, y - 9, right, y + 16)
    if final or measure.repeat_end:
        c.setLineWidth(1.8)
        c.line(right + (0 if final else -1), y - 9, right + (0 if final else -1), y + 16)
        c.setLineWidth(0.7)
        c.line(right - 4, y - 9, right - 4, y + 16)
    c.setFillColorRGB(0, 0, 0)
    if measure.repeat_start:
        c.setLineWidth(1.8)
        c.line(left, y - 9, left, y + 16)
        c.setLineWidth(0.65)
        c.line(left + 3.4, y - 9, left + 3.4, y + 16)
        c.circle(left + 6.6, y + 3, 0.95, fill=1, stroke=0)
        c.circle(left + 6.6, y - 2, 0.95, fill=1, stroke=0)
    if measure.repeat_end:
        c.circle(right - 6.5, y + 3, 0.95, fill=1, stroke=0)
        c.circle(right - 6.5, y - 2, 0.95, fill=1, stroke=0)


def draw_voice(
    c, part, bars, plan, row_index, row_for, y, lyric_font, number_font, upper_lyrics,
    lyric_size: float = PDF_LYRIC_SIZE, num_size: float = PDF_NUM_FONT_SIZE,
):
    anchors = {}
    for measure, (left, right, columns) in zip(bars, plan):
        positions = list(timed_tokens(measure.all_tokens()))
        index = 0
        measure_xs, measure_widths = [], []
        tuplet_start = None
        for group in measure.beats:
            xs = []
            for _ in group.tokens:
                token_index = index
                time, _, token = positions[index]
                index += 1
                x, width = columns[time]
                anchors[measure.number, token_index] = x
                xs.append(x)
                measure_xs.append(x)
                measure_widths.append(width)
                draw_jianpu_token(c, x, y, token, number_font, num_font_size=num_size)
                for verse, lyric in enumerate(token.lyrics[:2]):
                    lyric_key = (measure.number, time, verse, lyric)
                    if lyric and (measure.show_shared_lyrics or upper_lyrics is None or lyric_key not in upper_lyrics):
                        c.setFont(lyric_font, lyric_size)
                        c.drawCentredString(x, y - 22 - 12 * verse, lyric)
                if token.tuplet_start:
                    tuplet_start = x
                if token.tuplet_stop and tuplet_start is not None:
                    draw_tuplet(c, tuplet_start - 3, x + 3, y + 22.5, token.tuplet_actual or 3, number_font)
                    tuplet_start = None
            if xs and any(t.triplet for t in group.tokens) and not any(t.tuplet_actual for t in group.tokens):
                draw_tuplet(c, xs[0] - 3, xs[-1] + 3, y + 22.5, 3, number_font)
        draw_measure_beams(c, measure, measure_xs, measure_widths, y)
        draw_bar_edges(c, measure, left, right, y, measure.number == part.measures[-1].number)

    c.setStrokeColorRGB(0.18, 0.18, 0.18)
    c.setLineWidth(0.55)
    for arc in part.arcs:
        first, last = row_for[arc.start_measure], row_for[arc.end_measure]
        if not first <= row_index <= last:
            continue
        a = anchors.get(arc.start, LEFT + 1)
        b = anchors.get(arc.end, LEFT + INNER - 1)
        if b - a < 4:
            continue
        height = y + (18.0 if arc.kind == "tie" else 26.5)
        rise = min(7 if arc.kind == "slur" else 4, (b - a) / 6)
        c.bezier(a, height, a + (b - a) / 4, height + rise,
                 b - (b - a) / 4, height + rise, b, height)


def _select_typography(systems, lyric_font: str, num_size: float | None, lyric_size: float | None):
    if num_size is not None and lyric_size is not None:
        return num_size, lyric_size, [system_plan(s, lyric_font, lyric_size=lyric_size, num_size=num_size) for s in systems]
    candidates = [
        (PDF_NUM_FONT_SIZE, PDF_LYRIC_SIZE),
        (11.4, 8.6),
        (10.8, 8.1),
        (10.2, 7.7),
        (9.6, 7.2),
        (9.0, 6.8),
        (8.4, 6.4),
    ]
    last_err = None
    for cand_num, cand_lyric in candidates:
        n_sz = num_size if num_size is not None else cand_num
        l_sz = lyric_size if lyric_size is not None else cand_lyric
        try:
            layouts = [system_plan(s, lyric_font, lyric_size=l_sz, num_size=n_sz) for s in systems]
            return n_sz, l_sz, layouts
        except ValueError as exc:
            last_err = exc
    raise last_err or ValueError("A4 无法容纳指定小节数的双声部排版。")


def engrave_source_layout(
    score: Score,
    output: Path,
    *,
    bars_per_row: int | None = None,
    rows_per_page: int | None = None,
    num_font_size: float | None = None,
    lyric_font_size: float | None = None,
    voice_gap: float | None = None,
    draft: bool = True,
    footer_credit: str = "",
) -> Path:
    if len(score.parts) != 2:
        raise ValueError("双声部总谱排版需要恰好两个声部。")
    if errors := score.validate():
        raise ValueError("。".join(errors))

    parts = score.parts
    if bars_per_row is not None:
        if bars_per_row < 1:
            raise ValueError("每行小节数必须大于等于 1。")
        nums = [m.number for m in parts[0].measures]
        system_breaks = [nums[i:i + bars_per_row] for i in range(0, len(nums), bars_per_row)]
        n_rows = rows_per_page or (4 if bars_per_row <= 4 else max(1, score.rows_per_page))
    else:
        system_breaks = score.system_breaks
        n_rows = rows_per_page or score.rows_per_page
    if not system_breaks or n_rows < 1:
        raise ValueError("原谱排版需要原谱逐行小节编号（或 --bars-per-row）和每页谱行数。")
    if any(not row for row in system_breaks):
        raise ValueError("原谱分行中有空行。")
    expected = [n for row in system_breaks for n in row]
    if len(set(expected)) != len(expected) or any([m.number for m in part.measures] != expected for part in parts):
        raise ValueError("原谱谱行编号必须与两个声部的完整小节顺序一致。")

    lyrics_font, number_font = register_fonts()
    indexes = [{m.number: m for m in part.measures} for part in parts]
    systems = [[[index[n] for n in row] for index in indexes] for row in system_breaks]
    if bars_per_row is not None or num_font_size is not None or lyric_font_size is not None:
        eff_num, eff_lyric, layouts = _select_typography(systems, lyrics_font, num_font_size, lyric_font_size)
    else:
        eff_num, eff_lyric = PDF_NUM_FONT_SIZE, PDF_LYRIC_SIZE
        layouts = [system_plan(s, lyrics_font, lyric_size=eff_lyric, num_size=eff_num) for s in systems]

    eff_voice_gap = voice_gap if voice_gap is not None else (78.0 if bars_per_row is not None and eff_lyric >= 8.0 else VOICE_GAP)
    row_for = {number: row_index for row_index, row in enumerate(system_breaks) for number in row}
    pages = (len(systems) + n_rows - 1) // n_rows
    w, h = A4
    first_page_rows = min(n_rows, len(systems))
    if first_page_rows > 1 and (h - 156 - eff_voice_gap - 57) / (first_page_rows - 1) < 104:
        raise ValueError("A4 无法按指定行数容纳双声部原谱。")
    output.parent.mkdir(parents=True, exist_ok=True)
    c = Canvas(str(output), pagesize=A4)
    c.setTitle(score.title)
    for page in range(pages):
        if page == 0:
            c.setFont(lyrics_font, 22)
            c.drawCentredString(w / 2, h - 46, score.title)
            c.setFont(lyrics_font, 10.5)
            if score.engraver:
                c.drawCentredString(w / 2, h - 68, score.engraver)
            c.drawRightString(w - RIGHT, h - 72, score.composer)
            c.setFont(number_font, 10.5)
            c.drawString(LEFT, h - 78, f"1={score.key}")
            numerator, denominator = score.time_sig.split("/")
            c.drawCentredString(LEFT + 33, h - 72, numerator)
            c.drawCentredString(LEFT + 33, h - 85, denominator)
            c.line(LEFT + 29, h - 76, LEFT + 37, h - 76)
            c.setFont(lyrics_font, 10.5)
            if score.time_sig == "2/2":
                c.drawString(LEFT, h - 102, f"二分音符 = {score.bpm / 2:g}")
            else:
                c.drawString(LEFT, h - 102, f"四分音符 = {score.bpm:g}")

        first_row = page * n_rows
        top = h - (156 if page == 0 else 95)
        page_rows = min(n_rows, len(systems) - first_row)
        step = min(158.0 if n_rows <= 4 else 144.0, (top - eff_voice_gap - 57) / max(page_rows - 1, 1))
        for local_row, row_index in enumerate(range(first_row, min(first_row + n_rows, len(systems)))):
            system = systems[row_index]
            plan = layouts[row_index]
            y = top - local_row * step
            first = system[0][0]
            draw_boxed_bar_number(c, LEFT - 1.5, y + 31, first.number, number_font)
            c.setStrokeColorRGB(0, 0, 0)
            c.setLineWidth(1.35)
            c.line(LEFT, y - eff_voice_gap - 10, LEFT, y + 17)
            top_lyrics = {
                (m.number, t, verse, lyric) for m in system[0] for t, _, token in timed_tokens(m.all_tokens())
                for verse, lyric in enumerate(token.lyrics[:2]) if lyric
            }

            for v, part in enumerate(parts):
                y_voice = y - v * eff_voice_gap
                c.setFillColorRGB(0, 0, 0)
                c.setFont(lyrics_font, 10)
                c.drawRightString(LEFT - 9, y_voice, part.name)
                draw_voice(
                    c, part, system[v], plan, row_index, row_for, y_voice, lyrics_font,
                    number_font, top_lyrics if v else None,
                    lyric_size=eff_lyric, num_size=eff_num,
                )

            for bar, (left, right, _) in zip(system[0], plan):
                badge = bar.badge_note or (f"转1={bar.key}" if bar.key else "")
                if badge:
                    c.setFont(lyrics_font, 10.5)
                    c.drawString(left + 11, y + 49, badge)
                if bar.ending:
                    line_y = y + 39
                    c.setLineWidth(0.55)
                    c.line(left + 1, line_y - 5, left + 1, line_y)
                    c.line(left + 1, line_y, right - 1, line_y)
                    c.setFont(number_font, 8)
                    c.drawString(left + 5, line_y + 2.5, bar.ending)

        c.setFillColorRGB(0.25, 0.25, 0.25)
        if draft:
            c.setFont(lyrics_font, 8.5)
            c.drawString(LEFT, 28, "识谱复核样张")
        c.setFont(number_font, 9)
        c.drawCentredString(w / 2, 28, f"{page + 1}/{pages}")
        if footer_credit:
            c.setFont(lyrics_font, 8.5)
            c.drawCentredString(w / 2, 14, footer_credit)
        c.showPage()
    c.save()
    return output

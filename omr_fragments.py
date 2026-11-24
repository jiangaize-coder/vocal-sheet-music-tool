"""Per-note fragment cropping, pixel-level feature refinement, and 音值组合法 rhythm repair.

Three-stage Jianpu OMR refinement pipeline:
  Stage 1 (Dolce coarse pass): rows, barlines, initial note bboxes, lyrics, and cross-note arcs.
  Stage 2 (this module):
    - Build per-page self-calibrated 0..7 digit templates from the page's own clean fragments.
    - Recover missed digit fragments inside under-filled bars (horizontal ink gap scan).
    - Crop each note's contextual fragment (digit + beams below + dot right + octave dots)
      to refine `div` (underline beams), `dot`, and high-margin `digit` confusions (e.g. 6 vs 0).
  Stage 3 (constrained rhythm solver + theory.py 音值组合法):
    - When a bar's beat sum still differs from expected beats, solve minimum-cost legal `div`
      adjustments guided by fragment pixel confidence, then normalize beat grouping.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

TMPL_W, TMPL_H = 16, 24


def _normalize_patch(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Optional[np.ndarray]:
    h_img, w_img = gray.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w_img, x1), min(h_img, y1)
    if x1 - x0 < 3 or y1 - y0 < 5:
        return None
    patch = 255.0 - gray[y0:y1, x0:x1].astype(np.float32)
    # Trim empty margins inside bbox to align digit ink tightly
    ys, xs = np.where(patch > 40.0)
    if len(xs) < 4:
        return None
    tight = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    pil = Image.fromarray(np.clip(tight, 0, 255).astype(np.uint8))
    resized = np.asarray(pil.resize((TMPL_W, TMPL_H), Image.Resampling.BILINEAR), dtype=np.float32)
    centered = resized - float(resized.mean())
    norm = float(np.linalg.norm(centered))
    if norm < 1e-3:
        return None
    return centered / norm


def build_page_digit_templates(gray: np.ndarray, rows: List[dict], margin: float) -> Dict[int, np.ndarray]:
    """Build self-calibrated normalized templates for digits 0..7 from the page's own glyphs."""
    buckets: Dict[int, List[np.ndarray]] = {d: [] for d in range(8)}
    for row in rows:
        for note in row.get("nums", []):
            if note.get("grace") or note.get("doubt"):
                continue
            d = note.get("digit")
            if d not in buckets:
                continue
            b = note["bbox"]
            if b["x"] + b["w"] / 2 < margin:
                continue
            vec = _normalize_patch(gray, int(b["x"]), int(b["y"]), int(b["x"] + b["w"]), int(b["y"] + b["h"]))
            if vec is not None:
                buckets[d].append(vec)
    templates: Dict[int, np.ndarray] = {}
    for d, vecs in buckets.items():
        if len(vecs) >= 2:
            avg = np.mean(np.stack(vecs, axis=0), axis=0)
            norm = float(np.linalg.norm(avg))
            if norm > 1e-3:
                templates[d] = avg / norm
    return templates


def _classify_digit_fragment(
    gray: np.ndarray, bbox: dict, templates: Dict[int, np.ndarray],
) -> Tuple[Optional[int], float, Dict[int, float]]:
    if not templates:
        return None, 0.0, {}
    x0, y0 = int(bbox["x"]), int(bbox["y"])
    x1, y1 = int(bbox["x"] + bbox["w"]), int(bbox["y"] + bbox["h"])
    vec = _normalize_patch(gray, x0, y0, x1, y1)
    if vec is None:
        return None, 0.0, {}
    scores = {d: float(np.sum(vec * tmpl)) for d, tmpl in templates.items()}
    best_d = max(scores, key=scores.get)
    return best_d, scores[best_d], scores


def _measure_beam_profile(gray: np.ndarray, bbox: dict, row_bottom: float, glyph_h: float) -> Tuple[int, float, List[float]]:
    """Measure underline beam lines (0, 1, or 2) directly beneath a single note fragment.

    Returns (detected_div, confidence, slot_scores) where slot_scores = [line1_ink, line2_ink].
    Uses a band anchored to the row's median digit bottom so slight per-box y jitter doesn't miss line 1.
    """
    h_img, w_img = gray.shape
    cx = bbox["x"] + bbox["w"] / 2.0
    # Use a fixed horizontal window around the digit center (wide enough to distinguish horizontal
    # beam strokes from narrow circular low-octave dots, narrow enough not to bleed into neighbors).
    half_w = max(5.0, min(bbox["w"] * 0.65, glyph_h * 0.45))
    x0 = max(0, int(round(cx - half_w)))
    x1 = min(w_img, int(round(cx + half_w)) + 1)
    if x1 - x0 < 4:
        return 0, 0.0, [0.0, 0.0]

    base_y = max(bbox["y"] + bbox["h"], row_bottom)
    # Scan from +1px to +0.85*glyph_h below the digit baseline
    y0 = max(0, int(round(base_y + 1)))
    y1 = min(h_img, int(round(base_y + max(10.0, glyph_h * 0.78))))
    if y1 <= y0 + 2:
        return 0, 0.0, [0.0, 0.0]

    # Soft darkness profile per row: a continuous beam spans most columns in [x0, x1]
    band = gray[y0:y1, x0:x1].astype(np.float32)
    # Fraction of columns darker than 205 (handles anti-aliased 1px gray beam lines cleanly)
    row_cov = (band < 205.0).mean(axis=1)
    # Require at least one outer third to have ink (handles beam start/end stubs while rejecting centered dots)
    third = max(1, (x1 - x0) // 3)
    left_cov = (band[:, :third] < 210.0).mean(axis=1)
    right_cov = (band[:, -third:] < 210.0).mean(axis=1)
    is_beam_row = (row_cov >= 0.52) & ((left_cov >= 0.33) | (right_cov >= 0.33))

    # Group contiguous True rows into distinct horizontal strokes
    runs: List[Tuple[int, int, float]] = []
    for idx, flag in enumerate(is_beam_row):
        if flag:
            if not runs or idx > runs[-1][1] + 1:
                runs.append((idx, idx, float(row_cov[idx])))
            else:
                s, _, best = runs[-1]
                runs[-1] = (s, idx, max(best, float(row_cov[idx])))

    # Filter out thick lyric strokes near the very bottom (beams are thin: <= 3px thick and close together)
    valid_runs = [r for r in runs if (r[1] - r[0] + 1) <= max(3, int(round(glyph_h * 0.22)))]
    # Enforce realistic spacing: beam 1 within first 45% of glyph_h; beam 2 within 2..7px below beam 1
    filtered: List[Tuple[int, int, float]] = []
    for r in valid_runs:
        if not filtered:
            if r[0] <= max(6, int(round(glyph_h * 0.48))):
                filtered.append(r)
        else:
            gap = r[0] - filtered[-1][1]
            if 2 <= gap <= max(6, int(round(glyph_h * 0.40))):
                filtered.append(r)
    div = min(2, len(filtered))
    s1 = filtered[0][2] if len(filtered) >= 1 else float(row_cov[:max(1, len(row_cov) // 2)].max(initial=0.0))
    s2 = filtered[1][2] if len(filtered) >= 2 else float(row_cov[max(1, len(row_cov) // 2):].max(initial=0.0))
    conf = min(r[2] for r in filtered) if filtered else float(1.0 - max(s1, s2))
    return div, conf, [round(s1, 3), round(s2, 3)]


def _measure_dot_right(gray: np.ndarray, bbox: dict, next_x: Optional[float], glyph_h: float) -> Tuple[int, float]:
    """Detect an augmentation dot immediately to the right of a note fragment."""
    h_img, w_img = gray.shape
    x0 = int(round(bbox["x"] + bbox["w"] + 2))
    max_x = int(round(bbox["x"] + bbox["w"] + max(8.0, glyph_h * 0.65)))
    if next_x is not None:
        max_x = min(max_x, int(round(next_x - 2)))
    x1 = min(w_img, max_x)
    y0 = max(0, int(round(bbox["y"] + bbox["h"] * 0.25)))
    y1 = min(h_img, int(round(bbox["y"] + bbox["h"] * 0.85)))
    if x1 - x0 < 3 or y1 - y0 < 4:
        return 0, 0.0
    sub = gray[y0:y1, x0:x1] < 175
    ink = int(sub.sum())
    if ink < 3:
        return 0, 0.0
    ys, xs = np.where(sub)
    bh = int(ys.max() - ys.min() + 1)
    bw = int(xs.max() - xs.min() + 1)
    # A dot is small and roughly square (not a tall barline or wide dash)
    if 2 <= bh <= max(5, int(glyph_h * 0.30)) and 2 <= bw <= max(5, int(glyph_h * 0.30)):
        density = ink / max(1, bh * bw)
        if density >= 0.45:
            return 1, float(density)
    return 0, 0.0


def _measure_octave_above(gray: np.ndarray, bbox: dict, glyph_h: float) -> int:
    """Detect 1 or 2 compact high-octave dots directly above the digit center."""
    h_img, w_img = gray.shape
    cx = bbox["x"] + bbox["w"] / 2.0
    x0 = max(0, int(round(cx - max(3.5, glyph_h * 0.22))))
    x1 = min(w_img, int(round(cx + max(3.5, glyph_h * 0.22))) + 1)
    y1 = max(0, int(round(bbox["y"] - 1)))
    y0 = max(0, int(round(bbox["y"] - max(7.0, glyph_h * 0.50))))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0
    sub = gray[y0:y1, x0:x1] < 175
    if int(sub.sum()) < 3:
        return 0
    ys, xs = np.where(sub)
    bh = int(ys.max() - ys.min() + 1)
    bw = int(xs.max() - xs.min() + 1)
    # Reject wide horizontal strokes (slurs/ties) and very tall marks
    if 2 <= bh <= max(5, int(glyph_h * 0.30)) and 2 <= bw <= max(5, int(glyph_h * 0.30)):
        return 1
    return 0


def refine_row_fragments(
    gray: np.ndarray,
    row: dict,
    templates: Dict[int, np.ndarray],
    margin: float,
) -> Tuple[dict, dict]:
    """Refine every note fragment in a Dolce row using cropped pixel inspection.

    Returns (refined_row, stats_dict).
    """
    refined = deepcopy(row)
    nums = [n for n in refined.get("nums", []) if not n.get("grace") and n.get("digit") in range(8)]
    valid = [n for n in nums if n["bbox"]["x"] + n["bbox"]["w"] / 2 >= margin]
    if not valid:
        return refined, {"notes": 0, "div_changed": 0, "digit_changed": 0, "dot_changed": 0}

    valid.sort(key=lambda n: n["bbox"]["x"])
    glyph_h = float(median(n["bbox"]["h"] for n in valid))
    row_bottom = float(median(n["bbox"]["y"] + n["bbox"]["h"] for n in valid))

    stats = {"notes": len(valid), "div_changed": 0, "digit_changed": 0, "dot_changed": 0}

    for idx, note in enumerate(valid):
        bbox = note["bbox"]
        next_x = valid[idx + 1]["bbox"]["x"] if idx + 1 < len(valid) else None
        old_div = max(0, int(note.get("div", 0)))
        old_dot = max(0, int(note.get("dot", 0)))
        old_digit = int(note["digit"])

        # 1. Underline beam refinement (do not overwrite tuplet notes unless clearly beamed)
        px_div, div_conf, slot_scores = _measure_beam_profile(gray, bbox, row_bottom, glyph_h)
        note["_frag"] = {
            "dolce_div": old_div,
            "px_div": px_div,
            "div_conf": round(div_conf, 3),
            "slot_scores": slot_scores,
            "dolce_digit": old_digit,
        }
        # Trust pixel beam count when it clearly sees beams Dolce missed, or clearly sees NO ink at Dolce's beam level
        # (never alter notes that have augment dashes '-' since dashed notes are >= 2 beats and never beamed)
        if not note.get("augment") and not note.get("tuplet"):
            if px_div > old_div and div_conf >= 0.62:
                note["div"] = px_div
                stats["div_changed"] += 1
            elif px_div < old_div and div_conf >= 0.62 and slot_scores[min(old_div - 1, 1)] < 0.35:
                note["div"] = px_div
                stats["div_changed"] += 1

        # 2. High-margin digit confusion check (specifically targets 6 <-> 0 and similar font confusions)
        best_d, best_s, all_s = _classify_digit_fragment(gray, bbox, templates)
        if best_d is not None and old_digit in all_s:
            old_s = all_s[old_digit]
            note["_frag"]["px_digit"] = best_d
            note["_frag"]["px_digit_score"] = round(best_s, 3)
            note["_frag"]["dolce_digit_score"] = round(old_s, 3)
            # Override Dolce's digit when zero-mean page template strongly prefers another digit (e.g. 6 vs 0)
            if best_d != old_digit and best_s >= 0.84 and (best_s - old_s) >= 0.06:
                note["digit"] = best_d
                stats["digit_changed"] += 1

        # 3. Record dot pixel observation for downstream rhythm solver
        px_dot, dot_conf = _measure_dot_right(gray, bbox, next_x, glyph_h)
        note["_frag"]["px_dot"] = px_dot
        note["_frag"]["dot_conf"] = round(dot_conf, 3)
        if old_dot == 0 and px_dot == 1 and dot_conf >= 0.70 and not note.get("augment"):
            # Keep Dolce's dot unless rhythm solver needs it, or attach as candidate
            note["_frag"]["dot_candidate"] = True

    return refined, stats


def recover_missed_bar_fragments(
    gray: np.ndarray,
    bar_notes: List[dict],
    bar_left: float,
    bar_right: float,
    row_top: float,
    glyph_h: float,
    glyph_w: float,
    templates: Dict[int, np.ndarray],
    expected_beats: float,
    current_beats: float,
) -> Tuple[List[dict], int]:
    """Scan horizontal gaps inside an under-filled bar for digit fragments Dolce dropped."""
    if not templates or expected_beats - current_beats < 0.20:
        return bar_notes, 0
    h_img, w_img = gray.shape
    y0 = max(0, int(round(row_top - 2)))
    y1 = min(h_img, int(round(row_top + glyph_h + 2)))
    x_min = max(0, int(round(bar_left + 2)))
    x_max = min(w_img, int(round(bar_right + glyph_w * 0.6)))
    if x_max - x_min < int(glyph_w) or y1 - y0 < 6:
        return bar_notes, 0

    # Build mask of x columns already covered by existing notes
    covered = np.zeros(x_max - x_min, dtype=bool)
    for n in bar_notes:
        b = n["bbox"]
        a = max(0, int(round(b["x"] - 2)) - x_min)
        c = min(x_max - x_min, int(round(b["x"] + b["w"] + 2)) - x_min)
        covered[a:c] = True

    band = gray[y0:y1, x_min:x_max] < 185
    col_ink = band.sum(axis=0)
    # Find contiguous dark column runs in uncovered regions
    active = (col_ink >= max(2, int(glyph_h * 0.25))) & (~covered)
    spans: List[Tuple[int, int]] = []
    for i, flag in enumerate(active):
        if flag:
            if not spans or i > spans[-1][1] + 2:
                spans.append((i, i))
            else:
                spans[-1] = (spans[-1][0], i)

    row_bottom = row_top + glyph_h
    added: List[dict] = []
    for s0, s1 in spans:
        w = s1 - s0 + 1
        if not (max(4, glyph_w * 0.45) <= w <= glyph_w * 1.65):
            continue
        sub = band[:, s0:s1 + 1]
        ys = np.where(sub.any(axis=1))[0]
        if len(ys) == 0:
            continue
        bh = int(ys.max() - ys.min() + 1)
        if not (glyph_h * 0.65 <= bh <= glyph_h * 1.25):
            continue
        bbox = {
            "x": float(x_min + s0),
            "y": float(y0 + int(ys.min())),
            "w": float(w),
            "h": float(bh),
        }
        best_d, score, _ = _classify_digit_fragment(gray, bbox, templates)
        if best_d is None or score < 0.78:
            continue
        px_div, div_conf, slot_scores = _measure_beam_profile(gray, bbox, row_bottom, glyph_h)
        px_oct = _measure_octave_above(gray, bbox, glyph_h)
        px_dot, dot_conf = _measure_dot_right(gray, bbox, None, glyph_h)
        added.append({
            "bbox": bbox,
            "digit": best_d,
            "div": px_div,
            "dot": 1 if (px_dot == 1 and dot_conf >= 0.65) else 0,
            "octave": px_oct if best_d != 0 else 0,
            "lyrics": [],
            "_recovered": True,
            "_frag": {
                "dolce_div": px_div,
                "px_div": px_div,
                "div_conf": round(div_conf, 3),
                "slot_scores": slot_scores,
                "px_digit": best_d,
                "px_digit_score": round(score, 3),
                "recovered": True,
            },
        })
    if not added:
        return bar_notes, 0
    merged = sorted([*bar_notes, *added], key=lambda n: n["bbox"]["x"])
    return merged, len(added)


def solve_bar_rhythm_constraints(bar_notes: List[dict], expected_beats: float) -> Tuple[List[dict], bool]:
    """Minimum-cost constrained rhythm repair guided by fragment pixel confidence.

    If a bar's total beats != expected_beats, search small legal `div` flips (0/1/2)
    on notes with lower pixel confidence (or where Dolce and pixel profile disagreed)
    to reach exact `expected_beats` with clean unit-beat boundaries.
    """
    if not bar_notes:
        return bar_notes, False

    def _note_dur(n: dict, div_override: Optional[int] = None) -> float:
        lv = max(0, int(n.get("div", 0))) if div_override is None else div_override
        dots = max(0, int(n.get("dot", 0)))
        beats = (2.0 - 0.5 ** dots) / (2.0 ** lv)
        tuplet = n.get("tuplet") or {}
        if tuplet:
            beats *= tuplet["normal"] / tuplet["actual"]
        return beats + max(0, int(n.get("augment", 0)))

    total = sum(_note_dur(n) for n in bar_notes)
    if abs(total - expected_beats) <= 0.01:
        return bar_notes, False

    # Build candidate (div, cost) choices per note (only touch non-tuplet, non-augment notes)
    choices: List[List[Tuple[int, float]]] = []
    for n in bar_notes:
        cur = max(0, int(n.get("div", 0)))
        if n.get("tuplet") or n.get("augment"):
            choices.append([(cur, 0.0)])
            continue
        frag = n.get("_frag") or {}
        dolce_div = frag.get("dolce_div", cur)
        px_div = frag.get("px_div", cur)
        conf = float(frag.get("div_conf", 0.7))
        opts = [(cur, 0.0)]
        for cand in (0, 1, 2):
            if cand == cur:
                continue
            # Lower cost if the candidate matches either Dolce or pixel measurement
            if cand == dolce_div or cand == px_div:
                cost = 0.4 + 0.5 * conf
            elif abs(cand - cur) == 1:
                cost = 1.2 + 0.8 * conf
            else:
                cost = 2.4 + conf
            opts.append((cand, cost))
        choices.append(opts)

    # DP over (note_index, quantized_16th_beats) -> (min_cost, beat_boundary_bonus, div_path)
    target_q = int(round(expected_beats * 4))
    dp: Dict[int, Tuple[float, List[int]]] = {0: (0.0, [])}
    for n, opts in zip(bar_notes, choices):
        nxt: Dict[int, Tuple[float, List[int]]] = {}
        for q_sum, (cost, path) in dp.items():
            for cand_div, step_cost in opts:
                dur = _note_dur(n, cand_div)
                dq = int(round(dur * 4))
                nq = q_sum + dq
                if nq > target_q + 1:
                    continue
                # Penalize crossing a quarter-beat boundary with an eighth/sixteenth note (音值组合法)
                boundary_pen = 0.0
                if dq in (1, 2) and (q_sum % 4) + dq > 4:
                    boundary_pen = 0.6
                nc = cost + step_cost + boundary_pen
                if nq not in nxt or nc < nxt[nq][0]:
                    nxt[nq] = (nc, path + [cand_div])
        dp = nxt

    if target_q not in dp:
        return bar_notes, False
    best_cost, best_path = dp[target_q]
    # Guard against wild rewrites: only accept low-cost repairs (<= 3.5 total cost)
    if best_cost > 3.5:
        return bar_notes, False
    repaired = deepcopy(bar_notes)
    for n, new_div in zip(repaired, best_path):
        n["div"] = new_div
    return repaired, True


def save_fragment_diagnostic_sheet(
    gray: np.ndarray, notes: List[dict], out_path: Path, title_prefix: str = "",
) -> Optional[Path]:
    """Save a horizontal strip of cropped note fragments for visual inspection."""
    if not notes:
        return None
    glyph_h = max(12, int(round(median(n["bbox"]["h"] for n in notes))))
    crops = []
    h_img, w_img = gray.shape
    for n in notes:
        b = n["bbox"]
        x0 = max(0, int(b["x"] - 4))
        x1 = min(w_img, int(b["x"] + b["w"] + 8))
        y0 = max(0, int(b["y"] - glyph_h * 0.55))
        y1 = min(h_img, int(b["y"] + b["h"] + glyph_h * 0.90))
        if x1 > x0 and y1 > y0:
            crops.append(Image.fromarray(gray[y0:y1, x0:x1]))
    if not crops:
        return None
    max_h = max(im.height for im in crops)
    total_w = sum(im.width + 6 for im in crops) + 6
    strip = Image.new("L", (total_w, max_h + 8), 255)
    x = 6
    for im in crops:
        strip.paste(im, (x, 4))
        x += im.width + 6
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_path)
    return out_path

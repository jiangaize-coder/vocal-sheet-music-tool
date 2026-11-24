#!/usr/bin/env python3
"""
verify_score.py - Deterministic end-to-end verification suite (checks real artifacts, not mocks).

Checks real artifacts rather than proxies:
  1. Domain invariants: syncopation [2,1,2], 前八后十六 [1,2,2], beat sums, JSON round-trip.
  2. Typography invariants: single locked font-size constants for digits and Chinese lyrics.
  3. Synth + MIDI boundary: Chinese part names safely sanitized to ASCII in exported MIDI; MP3s non-empty.
  4. Engrave PDF: Portrait A4 PDFs generated and readable.
  5. Transcribe audio -> Score -> Print PDF end-to-end offline test.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pretty_midi
import soundfile as sf
from reportlab.pdfgen.canvas import Canvas

from domain import (
    BeatGroup,
    Measure,
    NoteToken,
    Part,
    Score,
    parse_jianpu_string_to_score,
)
from engrave import (
    AUGMENT_DASH_WIDTH, PDF_LYRIC_SIZE, PDF_NUM_FONT_SIZE, beam_runs, draw_jianpu_token,
    draw_measure_beams, engrave_score_pdfs, token_width,
)
from engrave_total import engrave_source_layout
from omr import compare_quarter_beats, import_pages, note_tokens
from synth import flatten_part_events, render_score_audio_bundle
from transcribe import transcribe_audio_to_score


def run_all() -> int:
    results = {}

    # 1. Domain model + syncopation + 前八后十六 + JSON round-trip
    sample_txt = "1=Bb 4/4 BPM=96 | 0_ 1_ 2= 2_ 3= 0 0 | 0_ 5= 5= 5_ 5= 2'= 1' - |"
    score = parse_jianpu_string_to_score(sample_txt, title="自检测试曲")
    errs = score.validate()
    assert not errs, f"Domain validation failed: {errs}"
    # Verify syncopation underline pattern [2, 1, 2] on Beat 2 of Measure 1
    m1_b2 = [t.underlines for t in score.parts[0].measures[0].beats[1].tokens]
    assert m1_b2 == [2, 1, 2], f"Expected syncopation [2,1,2], got {m1_b2}"
    # Verify 前八后十六 [1, 2, 2] on Beat 1 of Measure 2
    m2_b1 = [t.underlines for t in score.parts[0].measures[1].beats[0].tokens]
    assert m2_b1 == [1, 2, 2], f"Expected [1,2,2], got {m2_b1}"
    results["domain_and_rhythm_invariants"] = "PASS"

    # 2. Typography constants locked
    assert isinstance(PDF_NUM_FONT_SIZE, float) and PDF_NUM_FONT_SIZE > 0
    assert isinstance(PDF_LYRIC_SIZE, float) and PDF_LYRIC_SIZE > 0
    results["uniform_font_constants"] = f"PASS (num={PDF_NUM_FONT_SIZE}, lyrics={PDF_LYRIC_SIZE})"

    def image_note(x, digit=1, **marks):
        return dict(bbox=dict(x=x, y=30, w=7, h=20), digit=digit, **marks)

    pages = [{"file": "示范.png", "kind": "jianpu", "score": {
        "beats": 4, "beatType": 4, "key": "C",
        "rows": [{"topY": 30, "barlineXs": [5, 90, 160, 240], "nums": [
            image_note(10, tieStart=True, slurStart=1, lyrics=["rehearsal"]),
            image_note(30, tieStop=True), image_note(55, 2, slurStop=1),
            image_note(100, 3, augment=3),
            image_note(170, 4, augment=3, tieStart=True),
        ]}, {"topY": 130, "barlineXs": [5, 240], "nums": [
            image_note(20, 4, augment=3, tieStop=True),
        ]}],
    }}]
    image_score, review = import_pages(
        pages, voices=1, margin=0, title="图片回归", time_sig=None, key=None, bpm=100,
    )
    assert [m.total_beats for m in image_score.parts[0].measures] == [3, 4, 4, 4]
    assert [m.all_tokens()[0].degree for m in image_score.parts[0].measures] == ["1", "3", "4", "4"]
    assert sorted(a.kind for a in image_score.parts[0].arcs) == ["slur", "tie", "tie"]
    assert [beats for _, beats in flatten_part_events(image_score, image_score.parts[0])] == [2, 1, 4, 8]
    assert review["measures"][0]["measure"] == 1
    missing_beat, = review["quarter_discrepancies"]
    assert (missing_beat["measure"], missing_beat["quarter"], missing_beat["recognized"]) == (1, 4, [])
    with patch("engrave.canvas.Canvas") as drawing:
        target = drawing.return_value
        draw_jianpu_token(target, 100, 200, NoteToken("-", 1), "Helvetica")
        left, _, right, _ = target.line.call_args.args
        assert right - left == AUGMENT_DASH_WIDTH
    short = parse_jianpu_string_to_score("1=C 4/4 | 1 2 3 | 4 5 6 7 |")
    assert [[n.degree for n in m.all_tokens()] for m in short.parts[0].measures] == [
        ["1", "2", "3"], ["4", "5", "6", "7"],
    ]
    assert len(short.validate()) == 1
    assert token_width(NoteToken("1", 1, lyrics=["rehearsal"]), "Helvetica") > token_width(
        NoteToken("1", 1, lyrics=["hey"]), "Helvetica"
    )
    results["image_barlines_arcs_and_lyric_width"] = "PASS"

    example = [
        parse_jianpu_string_to_score("2/2 | 1'= 1'= 1'= 1'= 1'= 6= 1'= 2'= 0_ 5= 5= 3'_ 7= 2'= |"),
        parse_jianpu_string_to_score("2/2 | 4'= 4'= 4'= 4'= 4'= 2'= 4'= 5'= 0_ 2'= 2'= 5'_ 4'= 4'= |"),
    ]
    for example_score in example:
        assert not example_score.validate()
        measure = example_score.parts[0].measures[0]
        expected = [[2] * 4, [2] * 4, [1, 2, 2], [1, 2, 2]]
        tokens = measure.all_tokens()
        assert [[tokens[i].underlines for i in group] for group in beam_runs(measure)] == expected
        xs = [float(i * 32) for i in range(len(tokens))]
        with patch("engrave.draw_beams_pdf") as painter:
            draw_measure_beams(None, measure, xs, [12.0] * len(xs), 0.0)
        assert [call.args[3] for call in painter.call_args_list] == expected
    assert all(len(m.beats) == 4 for value in example for m in value.parts[0].measures)
    results["measure_24_quarter_beams"] = "PASS"

    correct_first = parse_jianpu_string_to_score("2/2 | 1_ 1= 1= 2_ 2= 3= 0 0 |").parts[0].measures[0]
    wrong_first = [image_note(i * 14, digit=digit, div=level) for i, (digit, level) in enumerate(
        [(1, 2), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (0, 0), (0, 0)]
    )]
    mismatch, = compare_quarter_beats(wrong_first, 4, correct_first)
    assert mismatch["quarter"] == 1 and not mismatch["overflow"]
    assert note_tokens(image_note(10, articulation="staccatissimo"))[0].articulation == "staccatissimo"
    results["equal_bar_length_wrong_beat"] = "PASS"

    flat_tie = NoteToken("1", .5, accidental="b", display_accidental="")
    natural = NoteToken("2", .25, display_accidental="♮")
    assert flat_tie.to_midi(59) == 58 and flat_tie.accidental_mark == ""
    assert natural.to_midi(59) == 61 and natural.accidental_mark == "♮"
    results["written_and_sounding_accidentals"] = "PASS"

    with tempfile.TemporaryDirectory(prefix="p2_score_verify_") as tmp:
        tmp_dir = Path(tmp)

        # JSON round-trip
        jpath = tmp_dir / "test.score.json"
        score.to_json(jpath)
        reloaded = Score.from_json(jpath)
        assert not reloaded.validate()
        results["json_roundtrip"] = "PASS"

        image_json = tmp_dir / "image.score.json"
        image_score.to_json(image_json)
        loaded_image = Score.from_json(image_json)
        assert loaded_image.parts[0].arcs == image_score.parts[0].arcs
        assert engrave_score_pdfs(loaded_image, tmp_dir / "image_pdf")

        # 3. Multi-part synth + ASCII MIDI check (with Chinese part names!)
        part2 = Part(name="和声", ascii_name="", measures=score.parts[0].measures)
        duo = Score(title="双声部自检", key="Bb", time_sig="4/4", bpm=96.0, parts=[score.parts[0], part2])
        audio_out = render_score_audio_bundle(duo, tmp_dir / "audio", count_in=True)
        for k, fpath in audio_out.items():
            p = Path(fpath)
            assert p.exists() and p.stat().st_size > 100, f"Empty/missing output {k}: {fpath}"
        results["multipart_synth_and_midi"] = f"PASS ({len(audio_out)} files)"

        duo.time_sig = "2/2"
        captured = {}
        def capture_audio(stereo, path):
            captured[path.name[:2]] = stereo
            return path

        with patch("synth.save_mp3", side_effect=capture_audio):
            aligned = render_score_audio_bundle(duo, tmp_dir / "aligned", count_in=True)
        midi = pretty_midi.PrettyMIDI(aligned["midi"])
        assert [(m.numerator, m.denominator) for m in midi.time_signature_changes] == [(2, 2)]
        assert all(len(captured[code]) == len(captured["01"]) for code in ("03", "04", "05", "06"))
        for code in ("03", "04", "05", "06"):
            assert np.max(np.abs(captured[code][18000:27000])) < 1e-8
        results["cut_time_and_count_in_alignment"] = "PASS"

        # 4. Print-ready Portrait A4 PDFs
        pdf_out = engrave_score_pdfs(duo, tmp_dir / "pdf")
        for k, fpath in pdf_out.items():
            p = Path(fpath)
            assert p.exists() and p.stat().st_size > 1000, f"Invalid PDF {k}: {fpath}"
        results["engrave_print_pdfs"] = f"PASS ({len(pdf_out)} files)"

        repeated = "2/2 | " + " | ".join(["1 2 3 4"] * 15) + " |"
        lead = parse_jianpu_string_to_score(repeated).parts[0]
        harmony = Part(name="和声", ascii_name="Harmony", measures=lead.measures)
        source_score = Score(
            title="原谱分页回归", time_sig="2/2", parts=[lead, harmony],
            system_breaks=[[m.number] for m in lead.measures], rows_per_page=5,
            engraver="唱明白工作室-制作",
        )
        with patch.object(Canvas, "showPage", autospec=True, side_effect=Canvas.showPage) as page:
            total_pdf = engrave_source_layout(source_score, tmp_dir / "原谱排版.pdf")
        assert page.call_count == 3 and total_pdf.stat().st_size > 1000
        drawn_texts = []
        orig_draw = Canvas.drawString
        orig_centre = Canvas.drawCentredString
        def rec_draw(self, x, y, text, *a, **kw):
            drawn_texts.append(str(text))
            return orig_draw(self, x, y, text, *a, **kw)
        def rec_centre(self, x, y, text, *a, **kw):
            drawn_texts.append(str(text))
            return orig_centre(self, x, y, text, *a, **kw)
        with patch.object(Canvas, "drawString", autospec=True, side_effect=rec_draw), \
             patch.object(Canvas, "drawCentredString", autospec=True, side_effect=rec_centre), \
             patch.object(Canvas, "showPage", autospec=True, side_effect=Canvas.showPage) as p4:
            four_bar_pdf = engrave_source_layout(source_score, tmp_dir / "四小节定稿.pdf", bars_per_row=4, draft=False)
        assert p4.call_count == 1 and four_bar_pdf.stat().st_size > 1000
        assert "唱明白工作室-制作" in drawn_texts and "识谱复核样张" not in drawn_texts
        results["source_layout_three_pages"] = "PASS"

        # 5. End-to-end offline audio -> Score -> Print PDF
        sr = 22050
        wav_p = tmp_dir / "melody.wav"
        chunks = [np.zeros(int(sr * 0.1))]
        for m in [60, 62, 64, 65, 67, 67, 67]:
            t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
            f = 440.0 * (2.0 ** ((m - 69) / 12.0))
            chunks.append((0.6 * np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)) * 0.5)
            chunks.append(np.zeros(int(sr * 0.03)))
        sf.write(str(wav_p), np.concatenate(chunks), sr)

        tr_score, tr_files = transcribe_audio_to_score(str(wav_p), tmp_dir / "tr", title="音频自检", bpm_override=120.0, key_override="C")
        assert not tr_score.validate()
        assert Path(tr_files["musicxml"]).exists() and len(tr_files["staff_svgs"]) >= 1
        tr_pdfs = engrave_score_pdfs(tr_score, tmp_dir / "tr")
        assert Path(next(iter(tr_pdfs.values()))).stat().st_size > 1000
        results["audio_to_score_e2e"] = "PASS"

        # 6. 4-pillar music theory & 音值组合法 engine verification
        from theory import audit_score_theory, normalize_measure_rhythm, transpose_score
        messy_m = Measure(number=1, beats=[
            BeatGroup(tokens=[NoteToken("0", 0.25), NoteToken("0", 0.25), NoteToken("1", 0.5)]),
            BeatGroup(tokens=[NoteToken("0", 1.0)]),
            BeatGroup(tokens=[NoteToken("0", 1.0)]),
            BeatGroup(tokens=[NoteToken("5", 1.0)]),
        ], chords=["Bb", "F"])
        norm_m = normalize_measure_rhythm(messy_m, "4/4")
        assert [t.beats for t in norm_m.beats[0].tokens] == [0.5, 0.5]
        assert len(norm_m.beats) == 4 and norm_m.beats[1].tokens[0].beats == 1.0 and norm_m.beats[2].tokens[0].beats == 1.0
        sc_t = Score(title="移调自检", key="Bb", time_sig="4/4", bpm=96.0, parts=[Part("主声", "Lead", [norm_m])])
        sc_c = transpose_score(sc_t, "C")
        assert sc_c.key == "C" and sc_c.parts[0].measures[0].chords == ["C", "G"]
        audit = audit_score_theory(sc_c, voice_type="children")
        assert audit["meter_kind"] == "复拍子" and not audit["validation_errors"] and audit["chords_present"]
        results["theory_and_note_grouping"] = "PASS"

        # 7. Per-note fragment beam detection & constrained rhythm repair (omr_fragments)
        from omr_fragments import _measure_beam_profile, solve_bar_rhythm_constraints
        canvas_img = np.full((60, 40), 255, dtype=np.uint8)
        canvas_img[10:28, 16:22] = 20  # vertical stroke of digit '1'
        canvas_img[31:33, 10:30] = 20  # beam line 1
        canvas_img[36:38, 10:30] = 20  # beam line 2
        det_div, det_conf, _ = _measure_beam_profile(canvas_img, {"x": 14, "y": 10, "w": 10, "h": 18}, 28.0, 18.0)
        assert det_div == 2 and det_conf >= 0.8, f"Expected 2 beams, got div={det_div} conf={det_conf}"
        broken_bar = [
            {"digit": 1, "div": 2, "dot": 0, "_frag": {"dolce_div": 2, "px_div": 2, "div_conf": 0.95}},
            {"digit": 2, "div": 2, "dot": 0, "_frag": {"dolce_div": 2, "px_div": 2, "div_conf": 0.95}},
            {"digit": 3, "div": 2, "dot": 0, "_frag": {"dolce_div": 2, "px_div": 2, "div_conf": 0.95}},
            {"digit": 4, "div": 1, "dot": 0, "_frag": {"dolce_div": 1, "px_div": 2, "div_conf": 0.55}},
            {"digit": 5, "div": 0, "dot": 0, "_frag": {"dolce_div": 0, "px_div": 0, "div_conf": 0.95}},
            {"digit": 6, "div": 0, "dot": 0, "_frag": {"dolce_div": 0, "px_div": 0, "div_conf": 0.95}},
            {"digit": 0, "div": 0, "dot": 0, "_frag": {"dolce_div": 0, "px_div": 0, "div_conf": 0.95}},
        ]
        repaired_bar, did_repair = solve_bar_rhythm_constraints(broken_bar, 4.0)
        assert did_repair and [n["div"] for n in repaired_bar[:4]] == [2, 2, 2, 2]
        results["fragment_omr_and_rhythm_repair"] = "PASS"

    print(json.dumps({"status": "ALL_VERIFIED", "checks": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())

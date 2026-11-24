#!/usr/bin/env python3
"""
cli.py - 声乐乐谱工具的命令行入口。

子命令：
  transcribe  音频/视频（或公开链接）→ 简谱打印 PDF、五线谱、MIDI、Score JSON
  omr         简谱图片 → 待复核的 Score JSON、复核报告、打印稿
  synth       简谱文本或 Score JSON → 人声唱音、钢琴 MP3、MIDI
  engrave     简谱文本或 Score JSON → 竖版 A4 少儿大字打印 PDF
  verify      14 项端到端自检

退出码：0 正常；2 缺参数、缺文件或谱子无效；3 运行出错。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from domain import Score, parse_jianpu_string_to_score
from engrave import engrave_score_pdfs
from synth import render_score_audio_bundle
from transcribe import transcribe_audio_to_score


def load_score_input(source: str, title: str, key: str, bpm: float, time_sig: str) -> Score:
    p = Path(source).expanduser()
    if p.exists() and p.is_file():
        if p.suffix.lower() == ".json":
            return Score.from_json(p)
        return parse_jianpu_string_to_score(p.read_text(encoding="utf-8"), title=title or p.stem, default_key=key, default_bpm=bpm, default_time_sig=time_sig)
    return parse_jianpu_string_to_score(source, title=title or "简谱", default_key=key, default_bpm=bpm, default_time_sig=time_sig)


def main() -> int:
    parser = argparse.ArgumentParser(description="声乐乐谱工具：音频转谱、简谱转唱音、少儿打印排版。")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("transcribe", help="音频/视频转简谱，输出打印版、五线谱和 MIDI")
    p_tr.add_argument("input", help="本地音视频文件，或公开链接（非加密）")
    p_tr.add_argument("-o", "--output-dir", default="./score_out")
    p_tr.add_argument("--title", default=None)
    p_tr.add_argument("--separate", action="store_true", help="先用 Demucs 分离人声")
    p_tr.add_argument("--mode", choices=["melody", "poly"], default="melody")
    p_tr.add_argument("--key", default=None)
    p_tr.add_argument("--bpm", type=float, default=None)
    p_tr.add_argument("--time-sig", default="4/4")

    p_omr = sub.add_parser("omr", help="简谱图片转 Score JSON 和 A4 打印稿，带复核报告")
    p_omr.add_argument("images", nargs="*", type=Path, help="按阅读顺序提供原图")
    p_omr.add_argument("--source-jsonl", type=Path, help="复用之前保存的 Dolce 识别 JSONL")
    p_omr.add_argument("--recognizer", type=Path, help="本机 Dolce omr-cli 路径")
    p_omr.add_argument("--reference", type=Path, help="用已校订的 Score JSON 对齐，保留已知的音符、时值和反复")
    p_omr.add_argument("--voices", type=int, choices=[1, 2])
    p_omr.add_argument("--left-edge", type=float, default=0, help="原图上谱左边缘的横坐标，用来排除声部名称误识")
    p_omr.add_argument("--time-sig", help="原谱拍号。老师确认的值会覆盖 OCR")
    p_omr.add_argument("--key", help="原谱调号。老师确认的值会覆盖 OCR")
    p_omr.add_argument("--bpm", type=float, help="每分钟四分音符数")
    p_omr.add_argument("--title")
    p_omr.add_argument("--no-fragments", action="store_true", help="关掉单音碎片精修和重组，只留 Dolce 原始结果")
    p_omr.add_argument("--save-fragments", action="store_true", help="把存疑小节的单音切片诊断图存到 fragments/ 目录")
    p_omr.add_argument("-o", "--output-dir", default="./omr_out")

    p_sy = sub.add_parser("synth", help="简谱或 Score JSON 转人声唱音、钢琴 MP3 和 MIDI")
    p_sy.add_argument("source", help="Score .json 路径、简谱 .txt 路径或直接简谱文本")
    p_sy.add_argument("-o", "--output-dir", default="./audio_out")
    p_sy.add_argument("--title", default="简谱示范")
    p_sy.add_argument("--key", default="C")
    p_sy.add_argument("--bpm", type=float, default=100.0)
    p_sy.add_argument("--time-sig", default="4/4")
    p_sy.add_argument("--count-in", action="store_true")
    p_sy.add_argument("--metronome", action="store_true")

    p_en = sub.add_parser("engrave", help="简谱或 Score JSON 排版成竖版 A4 少儿大字 PDF")
    p_en.add_argument("source", help="Score .json 路径、简谱 .txt 路径或直接简谱文本")
    p_en.add_argument("-o", "--output-dir", default="./pdf_out")
    p_en.add_argument("--title", default="简谱")
    p_en.add_argument("--key", default="C")
    p_en.add_argument("--bpm", type=float, default=100.0)
    p_en.add_argument("--time-sig", default="4/4")
    p_en.add_argument("--source-layout", action="store_true", help="按原谱分行、分页，出双声部总谱")
    p_en.add_argument("--bars-per-row", type=int, default=None, help="双声部总谱每行固定小节数，字号自动放到最大（不重叠）")
    p_en.add_argument("--rows-per-page", type=int, default=None, help="双声部总谱每页谱行数")
    p_en.add_argument("--engraver", default=None, help="标题下方的制谱署名（例：某某工作室-制作）")
    p_en.add_argument("--mood", default=None, help="速度行情绪标记（例：阳光、活泼地）。会覆盖文本或 Score 里的 mood")
    p_en.add_argument("--final", action="store_true", help="定稿。去掉左下角“识谱复核样张”字样")
    p_en.add_argument("--output-name", default=None, help="自定义双声部总谱输出文件名（不含 .pdf）")
    p_en.add_argument("--footer-credit", default="", help="双声部总谱每页页码下方的署名")

    p_tc = sub.add_parser("theory-check", help="乐理体检：音值组合、音域、曲式、和弦")
    p_tc.add_argument("source", help="Score .json 路径、简谱 .txt 路径或直接简谱文本")
    p_tc.add_argument("--voice", choices=["children", "soprano", "alto", "tenor", "baritone"], default="children")
    p_tc.add_argument("--title", default="简谱")
    p_tc.add_argument("--key", default="C")
    p_tc.add_argument("--bpm", type=float, default=100.0)
    p_tc.add_argument("--time-sig", default="4/4")

    p_nm = sub.add_parser("normalize", help="按音值组合法规整乐谱，可选移调（无损）")
    p_nm.add_argument("source", help="Score .json 路径或简谱 .txt 路径")
    p_nm.add_argument("-o", "--output", required=True, help="输出规整后的 .score.json 路径")
    p_nm.add_argument("--transpose", default=None, help="目标调号（例：C、G、Bb）")
    p_nm.add_argument("--title", default="简谱")
    p_nm.add_argument("--key", default="C")
    p_nm.add_argument("--bpm", type=float, default=100.0)
    p_nm.add_argument("--time-sig", default="4/4")

    sub.add_parser("verify", help="跑 14 项端到端自检")

    args = parser.parse_args()

    try:
        if args.cmd == "verify":
            import verify_score
            return verify_score.run_all()

        if args.cmd == "theory-check":
            from theory import audit_score_theory
            score = load_score_input(args.source, args.title, args.key, args.bpm, args.time_sig)
            print(json.dumps(audit_score_theory(score, voice_type=args.voice), ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "normalize":
            from theory import normalize_score_rhythm, transpose_score
            score = load_score_input(args.source, args.title, args.key, args.bpm, args.time_sig)
            score = normalize_score_rhythm(score)
            if args.transpose:
                score = transpose_score(score, args.transpose)
            out_p = Path(args.output).expanduser().resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            score.to_json(out_p)
            print(json.dumps({"status": "ok", "key": f"1={score.key}", "output": str(out_p)}, ensure_ascii=False, indent=2))
            return 0

        out_dir = Path(args.output_dir).expanduser().resolve()

        if args.cmd == "omr":
            from omr import import_pages, recognize

            if bool(args.images) == bool(args.source_jsonl):
                raise ValueError("请提供原图或 --source-jsonl，二选一。")
            if args.source_jsonl:
                pages = [json.loads(line) for line in args.source_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                pages = recognize(args.images, args.recognizer)
            reference = Score.from_json(args.reference) if args.reference else None
            if reference and not 1 <= len(reference.parts) <= 2:
                raise ValueError("图片识谱入口目前支持一声部和二声部。")
            score, review = import_pages(
                pages, voices=args.voices or (len(reference.parts) if reference else 1),
                margin=args.left_edge, title=args.title, time_sig=args.time_sig, key=args.key,
                bpm=args.bpm, reference=reference,
                fragment_refine=not args.no_fragments,
                fragment_dir=(out_dir / "fragments") if args.save_fragments else None,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            score_path = out_dir / "识谱.score.json"
            review_path = out_dir / "识谱复核.json"
            raw_path = out_dir / "Dolce原始识别.jsonl"
            score.to_json(score_path)
            review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
            raw_path.write_text("".join(json.dumps(page, ensure_ascii=False) + "\n" for page in pages), encoding="utf-8")
            files = engrave_score_pdfs(score, out_dir, draft=True)
            for name, path in (("score_json", score_path), ("review", review_path), ("raw", raw_path)):
                files[name] = str(path)
            print(json.dumps({"status": "needs_review", "outputs": files}, ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "transcribe":
            score, tr_files = transcribe_audio_to_score(
                args.input, out_dir, title=args.title, separate=args.separate,
                mode=args.mode, key_override=args.key, bpm_override=args.bpm, time_sig=args.time_sig,
            )
            errs = score.validate()
            if errs:
                print(json.dumps({"status": "invalid_score", "errors": errs}, ensure_ascii=False, indent=2), file=sys.stderr)
                return 2
            pdf_files = engrave_score_pdfs(score, out_dir)
            print(json.dumps({"status": "ok", "key": f"1={score.key}", "bpm": score.bpm, "outputs": {**tr_files, **pdf_files}}, ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "synth":
            score = load_score_input(args.source, args.title, args.key, args.bpm, args.time_sig)
            errs = score.validate()
            if errs:
                print(json.dumps({"status": "invalid_score", "errors": errs}, ensure_ascii=False, indent=2), file=sys.stderr)
                return 2
            files = render_score_audio_bundle(score, out_dir, count_in=args.count_in, metronome=args.metronome)
            print(json.dumps({"status": "ok", "key": f"1={score.key}", "bpm": score.bpm, "outputs": files}, ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "engrave":
            if args.footer_credit and not args.source_layout:
                raise ValueError("--footer-credit 需要与 --source-layout 一起使用。")
            score = load_score_input(args.source, args.title, args.key, args.bpm, args.time_sig)
            if args.engraver is not None:
                score.engraver = args.engraver
            if args.mood is not None:
                score.mood = args.mood
            errs = score.validate()
            if errs:
                print(json.dumps({"status": "invalid_score", "errors": errs}, ensure_ascii=False, indent=2), file=sys.stderr)
                return 2
            if args.source_layout:
                from engrave_total import engrave_source_layout

                safe_title = score.title.replace("/", "／").replace(chr(92), "＼").strip() or "简谱"
                stem = (args.output_name or f"{safe_title}-双声部原谱排版").removesuffix(".pdf")
                output = engrave_source_layout(
                    score,
                    out_dir / f"{stem}.pdf",
                    bars_per_row=args.bars_per_row,
                    rows_per_page=args.rows_per_page,
                    draft=not args.final,
                    footer_credit=args.footer_credit,
                )
                files = {"total_score_pdf": str(output)}
            else:
                files = engrave_score_pdfs(score, out_dir)
            print(json.dumps({"status": "ok", "key": f"1={score.key}", "bpm": score.bpm, "outputs": files}, ensure_ascii=False, indent=2))
            return 0

        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR(2): {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR(3): {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

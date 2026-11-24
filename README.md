# 声乐乐谱工具（vocal-sheet-music-tool）

唱明白工作室出品

本地 100% 离线运行的声乐教学乐谱与练唱音频工具包：音频转简谱/五线谱、简谱转人声唱音与钢琴示范、双声部陪唱混音、竖版 A4 少儿大字简谱打印。

## 功能

四条主链路，共用同一套简谱领域模型（`Score → Part → Measure → BeatGroup → NoteToken`）：

1. **音频转谱**（`transcribe`）：音频/视频或公开非加密链接 → 可选 Demucs 人声分离 → Basic Pitch 转 MIDI → 节拍对齐、调性分析、和弦识别、音值组合规整 → MusicXML、五线谱 SVG、简谱打印 PDF、MIDI、Score JSON。
2. **图片识谱**（`omr`）：简谱图片 → Dolce 粗识别 → 碎片像素精修 → 拍值约束求解与音值组合重组 → 可复核的 Score JSON、复核报告、打印稿。只接受图片，不接受 PDF。
3. **简谱转音频**（`synth`）：简谱文本或 Score JSON → 离线合成人声唱音（`ah`/`oo` 共振峰 + 颤音）与钢琴示范 → MP3 + MIDI；双声部可出独唱、合唱、练唱版。
4. **简谱排版**（`engrave`）：Score JSON 或简谱文本 → 竖版 A4 少儿大字打印 PDF（单声部分谱、合订本、双声部总谱）。

另有 `theory-check`（音值组合、音域、曲式、和弦四大体检）、`normalize`（按音值组合法规整 + 无损移调）、`verify`（14 项确定性端到端自检）。

## 运行环境

- macOS Apple Silicon（主要验证平台；Linux 可用，字体见环境变量）
- Python 3.11 + 虚拟环境（建议 `python3.11 -m venv .venv`）
- ffmpeg（转码与 MP3 导出）
- Dolce `omr-cli`（仅图片识谱需要，**不随本工具发布**，需另行安装：https://github.com/lodebar2026/dolce ，或用 `--recognizer` / `DOLCE_OMR_CLI` 指定本机程序路径）

## 安装

```sh
git clone https://github.com/jiangaize-coder/vocal-sheet-music-tool.git
cd vocal-sheet-music-tool
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg        # macOS；Linux 请用系统包管理器
```

## 用法

入口是 `score_tool.py`（自动找解释器，见环境变量一节），也可直接用虚拟环境的 Python 跑 `cli.py`。

```sh
# 1. 运行端到端自检（14 项）
python3 score_tool.py verify

# 2. 音频转简谱打印版 + 五线谱 MusicXML/SVG + MIDI（带伴奏加 --separate）
python3 score_tool.py transcribe /path/to/song.mp3 -o ./score_out --title "歌名"

# 3. 简谱转人声唱音 + 钢琴示范 MP3
python3 score_tool.py synth "1=C 4/4 BPM=96 | 1 1 5 5 | 6 6 5 - |" -o ./audio_out --title "小星星" --count-in

# 4. 简谱或 Score JSON 导出竖版 A4 少儿大字打印 PDF
python3 score_tool.py engrave /path/to/song.score.json -o ./pdf_out

# 5. 简谱图片识谱（双声部加 --voices 2；拍号/调号可用老师确认值覆盖）
python3 score_tool.py omr 第一页.png 第二页.png --voices 2 --time-sig 2/2 --key Bb --bpm 64 -o ./omr_out
```

双声部总谱（按原谱分页，或固定每行小节数自动放大字号）：

```sh
# 1:1 原谱分页复核稿
python3 score_tool.py engrave /path/to/校订.score.json --source-layout -o ./pdf_out

# 固定每行 4 小节大字定稿（带制谱署名、去掉“识谱复核样张”底标）
python3 score_tool.py engrave /path/to/校订.score.json \
  --source-layout --bars-per-row 4 --engraver "某某工作室-制作" --final \
  --output-name "歌名-竖版A4-每行4小节大字版" -o ./pdf_out
```

（示例中的 `某某工作室-制作` 只是署名示例，请换成你自己的署名。）

## 简谱文本语法

```text
1=C 4/4 BPM=96 mood=阳光、活泼地
| 1 1 5 5 | 6 6 5 - | 4 4 3 3 | 2 2 1 - |
```

- 调号 `1=X`（如 `1=Bb`）、拍号（`4/4`、`3/4`、`2/2` 等）、速度 `BPM=96` 或 `♩=96`、情绪 `mood=`（单个无空格词，印在速度行）。
- 音符 `1`–`7`、休止 `0`；升降号 `#`/`b`；高八度 `'`、低八度 `,`；附点 `.`；增时线 `-`（每根 +1 拍）；`(x)` 八分音符、`((x))` 十六分音符；`/3` 三连音；`|` 小节线。
- 歌词、反复记号、转调标记（小节 `badge_note`，如 `转1=B`）走 Score JSON。

## 环境变量

| 变量 | 用途 |
|---|---|
| `SCORE_TOOL_VENV` | 启动器要用的虚拟环境目录（或 Python 可执行文件完整路径）；未设置时依次找仓库内 `.venv`、`venv`、当前解释器 |
| `DOLCE_OMR_CLI` | Dolce `omr-cli` 可执行文件路径（图片识谱用；未设置时找 PATH 上的 `omr-cli`，或用 `--recognizer` 指定） |
| `SCORE_TOOL_CJK_FONT` | 排版用中文字体文件路径（默认 macOS 系统字体：宋体/黑体/苹方） |
| `SCORE_TOOL_ROUND_FONT` | 排版用圆润数字字体文件路径（默认 Arial Rounded Bold） |

## 项目结构

| 文件 | 职责 |
|---|---|
| `score_tool.py` | 启动器：找解释器并转发参数给 `cli.py` |
| `cli.py` | 统一子命令入口（`transcribe` / `omr` / `synth` / `engrave` / `theory-check` / `normalize` / `verify`），退出码 0/2/3 |
| `domain.py` | 唯一强类型领域模型 + 简谱文本解析 + JSON 序列化 + `validate()` 校验 |
| `theory.py` | 音值组合法、音域体检、无损移调、曲式分段、拍级和弦识别 |
| `transcribe.py` | 音频 → WAV → 可选 Demucs 人声分离 → Basic Pitch → MIDI → 节拍对齐 → MusicXML + 五线谱 SVG + Score JSON |
| `omr.py` + `omr_fragments.py` | 图片识谱：调外部 Dolce 粗识别 → 碎片像素精修 → 拍值约束求解 |
| `synth.py` | 纯 numpy/scipy 合成（人声共振峰 + 颤音、钢琴泛音）+ 双声部混音 + MIDI 导出 + ffmpeg 转 MP3 |
| `engrave.py` | 竖版 A4 单声部/多声部分谱打印 PDF |
| `engrave_total.py` | 竖版 A4 双声部总谱（原谱分页或固定小节数自动降字号） |
| `verify_score.py` | 14 项确定性自检 |
| `乐理与识谱理论库.md` | 乐理参考：音值组合法、首调唱名、音域、曲式、和声 |

## 已知限制

- **转调**：简谱文本解析目前只认第一个调号（`1=`），小节级转调需走 Score JSON 的 `Measure.key`；音频转谱全曲只定一个调。转调曲目请人工核对转调后的小节。
- **图片识谱**：只接受图片（PNG/JPG），不接受 PDF；识别本身依赖外部 Dolce，高八度点、调号判定以复核报告为准，正式导出前请人工确认 Score。
- **声部**：合成只取前两个声部，第三个及以上声部会被忽略；识谱入口支持一或二声部。
- **和弦**：只认大小三和弦，不认七和弦；BPM 钳制在 50–180。
- **静默处理**：超出 MIDI 36–96 的音会被钳制；无法解析的 token 会被跳过；末小节不足会补休止垫满——请以 `verify` 与复核报告为准。

## 许可证

本仓库采用 Apache License 2.0（见 LICENSE 文件）。

第三方依赖许可（摘要，以各项目官方声明为准）：

| 依赖 | 许可 | 备注 |
|---|---|---|
| Verovio | LGPLv3 | 以 PyPI wheel 动态链接方式调用 |
| Basic Pitch（含 ICASSP 2022 模型 `nmp.onnx`） | Apache-2.0 | 代码与模型同许可，随包分发，商用可用 |
| Demucs 代码 | MIT | **`htdemucs` 模型权重无许可声明、仅限科研用途**；本工具不重新分发权重，但 `--separate` 的商用不在已知授权范围内 |
| music21、numpy、scipy、soundfile、reportlab | BSD-3-Clause | |
| librosa | ISC | |
| pretty_midi | MIT | |
| Pillow | HPND | |
| yt-dlp | Unlicense | 仅转谱接受 URL 时使用 |
| Dolce `omr-cli` | MIT | 不随本工具发布，需另行安装 |

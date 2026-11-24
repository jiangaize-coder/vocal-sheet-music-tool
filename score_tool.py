#!/usr/bin/env python3
"""
score_tool.py - Public launcher for 声乐乐谱工具 (vocal sheet music tool).

Finds a Python interpreter with the tool's dependencies and forwards all
arguments to cli.py (which lives next to this file).

Interpreter lookup order:
  1. $SCORE_TOOL_VENV (a venv directory, or a full path to a python binary)
  2. <repo>/.venv/bin/python
  3. <repo>/venv/bin/python
  4. The interpreter running this script (sys.executable)

Exit codes: 0 = ok (from cli.py), 2 = interpreter or cli.py not found.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
IMPL_CLI = REPO_ROOT / "cli.py"


def _candidate_pythons():
    env = os.environ.get("SCORE_TOOL_VENV", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            yield p / "bin" / "python"
        else:
            yield p
        return  # explicit setting wins; missing file is an error, not a fallback
    yield REPO_ROOT / ".venv" / "bin" / "python"
    yield REPO_ROOT / "venv" / "bin" / "python"
    yield Path(sys.executable)


def main() -> int:
    if not IMPL_CLI.exists():
        print(f"ERROR(2): 未找到工具入口: {IMPL_CLI}", file=sys.stderr)
        return 2
    for candidate in _candidate_pythons():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            proc = subprocess.run([str(candidate), str(IMPL_CLI), *sys.argv[1:]])
            return proc.returncode
    print(
        "ERROR(2): 未找到可用的 Python 解释器：请设置 SCORE_TOOL_VENV，"
        "或在仓库内创建 .venv / venv 并安装 requirements.txt",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

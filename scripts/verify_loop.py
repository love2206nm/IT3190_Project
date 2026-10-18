"""Vòng lặp tự kiểm chứng và tối ưu cục bộ (Local /loop Engine).

Script này đóng vai trò 'Giám khảo tự động' cho AI Agent:
1. Quét lỗi cú pháp & hàm cấm (Ruff).
2. Quét ranh giới Clean Architecture (Import Linter).
3. Quét comment rác từ AI.
4. Chạy toàn bộ Unit Tests (Pytest).

Nếu có bất kỳ bước nào thất bại, script trả về exit code 1 kèm traceback
để Agent tự đọc và tự sửa lại code cho đến khi 10/10 xanh.
"""

import os
import subprocess
import sys
from pathlib import Path

# Thiết lập UTF-8 cho Windows console tránh lỗi charmap encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent


def run_command(name: str, cmd: list[str]) -> bool:
    print(f"\n[KIEM TRA] {name}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        print(f"[CANH BAO] Khong tim thay lenh: {' '.join(cmd)}. Hay kiem tra thu vien da cai dat chua.")
        return False

    if result.returncode != 0:
        print(f"[FAIL] THAT BAI: {name}")
        print("=" * 60)
        if result.stdout:
            print("STDOUT:\n" + result.stdout)
        if result.stderr:
            print("STDERR:\n" + result.stderr)
        print("=" * 60)
        return False

    print(f"[PASS] THANH CONG: {name}")
    return True


def main() -> int:
    print("=" * 60)
    print("BAT DAU VONG LAP KIEM CHUNG TU DONG (/loop)")
    print("=" * 60)

    steps = [
        ("1. Cu phap & Rao chan Footguns (Ruff)", [sys.executable, "-m", "ruff", "check", "source/", "tests/"]),
        (
            "2. Ranh gioi Clean Architecture (Import Linter)",
            [
                sys.executable,
                "-c",
                "from importlinter.cli import lint_imports_command; lint_imports_command()",
            ],
        ),
        ("3. Rao chan Comment rac tu AI", [sys.executable, "scripts/check_no_ai_comments.py"]),
        ("4. Toan bo Unit Tests (Pytest)", [sys.executable, "-m", "pytest", "tests/", "-q"]),
    ]

    for name, cmd in steps:
        success = run_command(name, cmd)
        if not success:
            print("\n[VONG LAP CHUA DAT] Yeu cau Agent doc ky traceback o tren va tu sua code.")
            return 1

    print("\n" + "=" * 60)
    print("[CHUC MUNG] TAT CA 4 RAO CHAN DA DAT DIEM 10/10 XANH!")
    print("Ban da san sang de mo Pull Request")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

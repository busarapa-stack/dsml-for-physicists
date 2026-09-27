"""Run every numbered script of one chapter in order and save a log.

Usage:
    python run_chapter.py ch01_intro
    python run_chapter.py ch01_intro --clean     # delete data/ and outputs/ first
    python run_chapter.py ch02_programming --with-exercises   # also exercises/E*.py
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chapter", help="chapter folder, e.g. ch01_intro")
    ap.add_argument("--clean", action="store_true", help="remove data/ and outputs/ first (except files in keep_on_clean.txt)")
    ap.add_argument("--with-exercises", action="store_true",
                    help="also run exercises/E*.py after the numbered scripts")
    args = ap.parse_args()

    chdir = ROOT / args.chapter
    if not chdir.is_dir():
        sys.exit(f"not found: {chdir}")

    if args.clean:
        # files listed in keep_on_clean.txt (real downloaded data) survive --clean
        keep, saved = chdir / "keep_on_clean.txt", {}
        if keep.exists():
            for pattern in keep.read_text().split():
                for f in chdir.glob(pattern):
                    if f.is_file():
                        saved[f.relative_to(chdir)] = f.read_bytes()
        for sub in ("data", "outputs"):
            shutil.rmtree(chdir / sub, ignore_errors=True)
        for rel, content in saved.items():
            (chdir / rel).parent.mkdir(parents=True, exist_ok=True)
            (chdir / rel).write_bytes(content)
        if saved:
            print(f"kept {len(saved)} file(s) listed in keep_on_clean.txt")
    (chdir / "data").mkdir(exist_ok=True)
    (chdir / "outputs").mkdir(exist_ok=True)

    scripts = sorted(p for p in chdir.glob("[0-9][0-9]_*.py"))
    if args.with_exercises:
        key = lambda p: [(0, int(t), "") if t.isdigit() else (1, 0, t)
                         for t in re.split(r"(\d+)", p.stem) if t]
        scripts += sorted(chdir.glob("exercises/E*.py"), key=key)
    log_path = chdir / "outputs" / "run_log.txt"
    failed = []
    with open(log_path, "w", encoding="utf-8") as log:
        for s in scripts:
            t0 = time.time()
            name = s.relative_to(chdir).as_posix()
            print(f"=== {name} ===", flush=True)
            r = subprocess.run([sys.executable, str(s)], cwd=chdir,
                               capture_output=True, text=True)
            dt = time.time() - t0
            out = r.stdout + (("\n[stderr]\n" + r.stderr) if r.returncode else "")
            print(out, flush=True)
            log.write(f"=== {name}  ({dt:.1f} s, exit {r.returncode}) ===\n{out}\n")
            if r.returncode:
                failed.append(name)

    print(f"log saved to {log_path.relative_to(ROOT)}")
    if failed:
        sys.exit(f"FAILED: {', '.join(failed)}")
    print(f"all {len(scripts)} scripts finished OK")


if __name__ == "__main__":
    main()

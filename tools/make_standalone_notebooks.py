"""Build notebooks/standalone/chXX_*.ipynb: one self-contained Colab notebook per chapter.

Same cells as the notebooks made by make_colab_notebooks.py, but the chapter's code
(and the code of any earlier chapter it reuses) is packed into the notebook itself as a
compressed archive. The notebooks therefore run on Colab, or in any Jupyter, without
GitHub: open the .ipynb (upload it to Colab or Google Drive) and run the first cells.

The notebooks are GENERATED: edit the scripts, then run
    python tools/make_standalone_notebooks.py
"""
import base64
import io
import re
import sys
import zipfile
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_colab_notebooks as mc  # noqa: E402

ROOT = mc.ROOT
OUT_DIR = ROOT / "notebooks" / "standalone"
SKIP_DIRS = {"data", "outputs", "__pycache__", ".ipynb_checkpoints"}
KEEP_SUFFIX = {".py", ".md", ".txt", ".sh", ".json", ".ipynb"}
CHAPTERS = sorted(p.name for p in ROOT.glob("ch[0-9][0-9]_*") if p.is_dir())

INTRO_EXTRA = """
> **ไฟล์นี้ใช้ได้โดยไม่ต้องพึ่ง GitHub** โค้ดประกอบตำราของบทนี้ (และของบทก่อนหน้าที่บทนี้เรียกใช้) บรรจุอยู่ในเซลล์ **โค้ดประกอบตำรา** ในรูปข้อมูลบีบอัด
> เปิดไฟล์นี้บน Colab ได้สองทาง คืออัปโหลดผ่านเมนู File → Upload notebook หรือเก็บไว้ใน Google Drive แล้วเปิดด้วย Google Colaboratory
> จากนั้นรันเซลล์ **โค้ดประกอบตำรา** และเซลล์ **เตรียมโค้ด** ก่อนเซลล์อื่นทุกครั้ง
"""

BUNDLE = '''#@title โค้ดประกอบตำรา (ข้อมูลบีบอัด ไม่ต้องแก้ไข) — รันเซลล์นี้ก่อน
# บรรจุโฟลเดอร์ {folders} ของโค้ดประกอบตำรา ({nfiles} ไฟล์)
CODE_ZIP_B64 = "{b64}"
'''

SETUP = """# เตรียมโค้ด: แตกโค้ดที่บรรจุไว้ในโน้ตบุ๊กนี้ (ไม่ต้องใช้ GitHub) และติดตั้งเฉพาะไลบรารีที่ยังไม่มี
import os, sys, io, base64, zipfile, subprocess, importlib.util
CHAPTER = {chapter!r}
ROOT = os.path.abspath('dsml-for-physicists')
if not os.path.isdir(os.path.join(ROOT, CHAPTER)):
    zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_ZIP_B64))).extractall(ROOT)
os.chdir(os.path.join(ROOT, CHAPTER))
os.makedirs('data', exist_ok=True); os.makedirs('outputs', exist_ok=True)
NEED = {need!r}
missing = [req for mod, req in NEED.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
print('โฟลเดอร์ทำงาน:', os.getcwd())
print('ติดตั้งเพิ่ม:', ', '.join(missing) if missing else 'ไม่มี')
"""


def depends_on(chapter):
    """Chapters whose code this chapter reuses (transitively), including itself."""
    need, todo = {chapter}, [chapter]
    while todo:
        c = todo.pop()
        text = "\n".join(p.read_text(encoding="utf-8", errors="ignore")
                         for p in (ROOT / c).rglob("*.py"))
        for other in CHAPTERS:
            if other not in need and re.search(re.escape(other), text):
                need.add(other)
                todo.append(other)
    return sorted(need)


def bundle(chapters):
    buf, n = io.BytesIO(), 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in ["requirements.txt", "run_chapter.py", "README.md"]:
            z.write(ROOT / name, name); n += 1
        for c in chapters:
            for p in sorted((ROOT / c).rglob("*")):
                rel = p.relative_to(ROOT)
                if p.is_file() and p.suffix in KEEP_SUFFIX and not (SKIP_DIRS & set(rel.parts)):
                    z.write(p, rel.as_posix()); n += 1
    return base64.b64encode(buf.getvalue()).decode("ascii"), n


def build(chdir):
    ref = nbf.read(ROOT / "notebooks" / f"{chdir.name}.ipynb", as_version=4)   # the GitHub version
    chapters = depends_on(chdir.name)
    b64, n = bundle(chapters)
    cells = list(ref.cells)
    assert cells[1].cell_type == "code" and "git" in "".join(cells[1].source), "unexpected layout"
    cells[0] = nbf.v4.new_markdown_cell("".join(cells[0].source).rstrip() + "\n" + INTRO_EXTRA)
    zipcell = nbf.v4.new_code_cell(BUNDLE.format(folders=", ".join(chapters), nfiles=n, b64=b64))
    zipcell.metadata = {"cellView": "form", "jupyter": {"source_hidden": True}}
    setup = nbf.v4.new_code_cell(SETUP.format(chapter=chdir.name, need=mc.imports(chdir)))
    ref.cells = [cells[0], zipcell, setup] + cells[2:]
    for i, c in enumerate(ref.cells):      # fixed cell ids: regenerating gives an identical file
        c.id = f"{chdir.name[:4]}-s{i:03d}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{chdir.name}_standalone.ipynb"
    nbf.write(ref, out)
    return out, chapters, len(b64)


if __name__ == "__main__":
    for d in sorted(ROOT.glob("ch[0-9][0-9]_*")):
        if (ROOT / "notebooks" / f"{d.name}.ipynb").exists():
            out, chs, size = build(d)
            print(f"{out.relative_to(ROOT)}: bundles {', '.join(chs)} ({size / 1024:.0f} kB of code)")

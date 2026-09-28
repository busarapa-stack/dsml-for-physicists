"""Build notebooks/chXX_*.ipynb (one per chapter) from the chapter scripts.

The notebooks are GENERATED: edit the scripts, then run
    python tools/make_colab_notebooks.py
Every listing chunk of a script (between '# --- code:<label>' markers) becomes
its own cell headed with the number printed in the book (โค้ดที่ N.M).
The numbers come from tools/listing_numbers.json (made from the book's main.aux).
"""
import ast
import json
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/busarapa-stack/dsml-for-physicists.git"
NUM = json.loads((ROOT / "tools" / "listing_numbers.json").read_text(encoding="utf-8"))

TITLES = {
    0: "พื้นฐานการเขียนโปรแกรมภาษาไพทอนสำหรับนักฟิสิกส์",
    1: "บทนำและภูมิทัศน์ข้อมูลฟิสิกส์",
    2: "การโปรแกรมเชิงวิทยาศาสตร์และการทำงานที่ทำซ้ำได้",
    3: "การทำความสะอาดและการเตรียมลักษณะข้อมูลฟิสิกส์",
    4: "สถิติเชิงอนุมานและการวิเคราะห์ข้อมูลเชิงสำรวจ",
    5: "การถดถอยและแบบจำลองที่มีข้อจำกัดทางฟิสิกส์",
    6: "การจำแนกประเภทข้อมูลทางฟิสิกส์",
    7: "การเรียนรู้แบบไม่มีผู้สอนและการจัดกลุ่ม",
    8: "การลดมิติและตัวแปรรวมทางฟิสิกส์",
    9: "การประเมินแบบจำลองและการระบุความไม่แน่นอน",
    10: "โครงข่ายประสาทเทียมและการเรียนรู้ที่สอดคล้องกับฟิสิกส์",
    11: "โครงข่ายประสาทแบบคอนโวลูชันสำหรับภาพทางฟิสิกส์",
    12: "การบูรณาการโครงงาน จริยธรรม และการทำซ้ำได้",
}

# import name -> pip requirement (only packages that Colab may not have)
PIP = {
    "sklearn": "scikit-learn>=1.3", "seaborn": "seaborn>=0.12", "statsmodels": "statsmodels>=0.14",
    "imblearn": "imbalanced-learn>=0.12", "requests": "requests>=2.28", "h5py": "h5py>=3.8",
    "astropy": "astropy>=5.3", "uproot": "uproot>=5.0", "awkward": "awkward>=2.4", "pint": "pint>=0.22",
    "nbformat": "nbformat>=5.9", "nbclient": "nbclient>=0.8", "umap": "umap-learn>=0.5",
    "pymc": "pymc>=5.10", "arviz": "arviz>=0.17", "torch": "torch>=2.0",
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "matplotlib": "matplotlib",
}
EXTRA_FILES = {"code:gitignore": ["shell/gitignore_python.txt"]}   # listing files that do not name their label
SKIP = re.compile(r"claims_checks")          # author-side checks, not for students
MARK = re.compile(r"^# --- (code:[A-Za-z0-9_-]+|end of listings)")

INTRO = """# บทที่ {n} {title}
โน้ตบุ๊กประกอบตำรา *วิทยาศาสตร์ข้อมูลและการเรียนรู้ของเครื่องสำหรับนักฟิสิกส์*
โค้ดทุกเซลล์คัดลอกมาจากสคริปต์ในโฟลเดอร์ `{chapter}/` หัวเซลล์ **โค้ดที่ N.M** ตรงกับเลขในตำรา

**วิธีใช้**
1. รันเซลล์ **เตรียมโค้ด** ด้านล่างก่อนเสมอ (ทุกครั้งที่เปิดโน้ตบุ๊กหรือเริ่ม session ใหม่)
2. แต่ละหัวข้อเริ่มด้วยเซลล์ที่ขึ้นต้นด้วย `%reset -f` ซึ่งล้างตัวแปรเก่า ทำให้ผลเหมือนรันสคริปต์นั้นเดี่ยว ๆ
   ให้รันเซลล์ในหัวข้อเดียวกันเรียงจากบนลงล่าง (Shift+Enter) หรือใช้ Runtime → Run all
3. ใช้หน่วยประมวลผลกลาง (CPU) ตามค่าเริ่มต้น ไม่ต้องเปลี่ยนเป็น GPU
4. ไฟล์ที่สร้างขึ้นอยู่ใน `{chapter}/outputs/` และหายเมื่อ session จบ ถ้าต้องการเก็บ ให้คัดลอกไปยัง Google Drive

ตัวเลขส่วนใหญ่ควรตรงกับตำรา การคำนวณที่ไวต่อการปัดเศษ (การฝึกโครงข่ายประสาทเทียม
และมอนติคาร์โลในพิกัดต่อเนื่อง) อาจต่างเล็กน้อยเมื่อรันบนเครื่องอื่น ดู `{chapter}/README.md`
"""

SETUP = """# เตรียมโค้ด: ดึงโค้ดจาก GitHub (เมื่อรันบน Colab) และติดตั้งเฉพาะไลบรารีที่ยังไม่มี
import os, sys, subprocess, importlib.util
CHAPTER = {chapter!r}
REPO = {repo!r}
if 'google.colab' in sys.modules:
    ROOT = '/content/dsml-for-physicists'
    if not os.path.isdir(ROOT):
        subprocess.run(['git', 'clone', '-q', REPO, ROOT], check=True)
else:                                   # Jupyter บนเครื่องตนเอง: เปิดจากโฟลเดอร์ notebooks/
    ROOT = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
os.chdir(os.path.join(ROOT, CHAPTER))
os.makedirs('data', exist_ok=True); os.makedirs('outputs', exist_ok=True)
NEED = {need!r}
missing = [req for mod, req in NEED.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
print('โฟลเดอร์ทำงาน:', os.getcwd())
print('ติดตั้งเพิ่ม:', ', '.join(missing) if missing else 'ไม่มี')
"""

HEADER = """%reset -f
# ทำให้โค้ดทำงานเหมือนสั่ง  python {rel}  จากโฟลเดอร์ของบท
__file__ = {rel!r}
import os, sys, time; sys.argv = [__file__]; os.environ['NB_T0'] = str(time.time())
_root = os.path.dirname(os.getcwd())         # ล้างโมดูลและ path ของหัวข้อก่อนหน้า (เช่น _shared.py ของบทอื่น)
sys.path[:] = [os.path.dirname(os.path.abspath(__file__))] + [p for p in sys.path if not os.path.abspath(p or '.').startswith(_root + os.sep)]
for _m in [m for m, v in list(sys.modules.items()) if str(getattr(v, '__file__', '') or '').startswith(_root + os.sep)]:
    del sys.modules[_m]
"""

SHOW = """# แสดงรูปที่ส่วนนี้สร้าง (สคริปต์บันทึกรูปลงไฟล์ใน outputs/)
from IPython.display import Image, display
import glob, os
for _f in sorted(glob.glob('outputs/**/*.png', recursive=True)):
    if os.path.getmtime(_f) >= float(os.environ['NB_T0']):
        print(_f); display(Image(_f))
"""


def doc_line(src):
    try:
        d = ast.get_docstring(ast.parse(src)) or ""
    except SyntaxError:
        d = ""
    return d.strip().splitlines()[0] if d.strip() else ""


def imports(chdir):
    mods = set()
    for p in chdir.rglob("*.py"):
        if "optional" in p.parts:
            continue
        mods |= set(re.findall(r"^\s*(?:import|from) ([A-Za-z_]\w*)", p.read_text(encoding="utf-8"), re.M))
    return {m: PIP[m] for m in sorted(mods) if m in PIP}


def script_cells(chdir, path):
    rel = path.relative_to(chdir).as_posix()
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    chunks, cur, label = [], [], None
    for ln in lines:
        m = MARK.match(ln)
        if m:
            chunks.append((label, cur)); cur, label = [ln], m.group(1)
        else:
            cur.append(ln)
    chunks.append((label, cur))
    labels = [l for l, _ in chunks if l and l.startswith("code:")]
    if not labels:                      # no markers: use the labels named in the docstring
        labels = list(dict.fromkeys(re.findall(r"code:[A-Za-z0-9_-]+", ast.get_docstring(safe_parse(src)) or "")))
        labels += [l for l in dict.fromkeys(re.findall(r"^# .*?(code:[A-Za-z0-9_-]+)", src[:600], re.M)) if l not in labels]
    nums = [NUM[l] for l in labels if l in NUM]
    title = f"## `{rel}`" + (f" — โค้ดที่ {', '.join(nums)}" if nums else "")
    cells = [nbf.v4.new_markdown_cell(title + ("\n\n" + doc_line(src) if doc_line(src) else ""))]
    head = HEADER.format(rel=rel)
    first_label, first = chunks[0]
    cells.append(nbf.v4.new_code_cell(head + "".join(first).rstrip("\n")))
    for lab, body in chunks[1:]:
        text = "".join(body).rstrip("\n")
        if lab == "end of listings":
            if text.strip() != lines_strip(body[0]):
                cells.append(nbf.v4.new_markdown_cell("**ส่วนตรวจผล** (ไม่อยู่ในตำรา): พิมพ์ตัวเลขที่ตำรายกมาเพื่อเทียบ"))
                cells.append(nbf.v4.new_code_cell(text))
        else:
            cap = f"### โค้ดที่ {NUM[lab]}" if lab in NUM else f"### `{lab}`"
            cells.append(nbf.v4.new_markdown_cell(cap))
            cells.append(nbf.v4.new_code_cell(text))
    if "savefig" in src or "imsave" in src:
        cells.append(nbf.v4.new_code_cell(SHOW))
    return cells


def safe_parse(src):
    try:
        return ast.parse(src)
    except SyntaxError:
        return ast.parse("")


def lines_strip(s):
    return s.strip()


def build(chdir):
    n = int(chdir.name[2:4])
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}
    cells = [nbf.v4.new_markdown_cell(INTRO.format(n=n, title=TITLES[n], chapter=chdir.name)),
             nbf.v4.new_code_cell(SETUP.format(chapter=chdir.name, repo=REPO_URL, need=imports(chdir)))]
    for p in sorted(chdir.glob("[0-9][0-9]_*.py")):
        if not SKIP.search(p.name):
            cells += script_cells(chdir, p)
    key = lambda p: [(0, int(t), "") if t.isdigit() else (1, 0, t) for t in re.split(r"(\d+)", p.stem) if t]
    ex = sorted(chdir.glob("exercises/E*.py"), key=key)
    if ex:
        cells.append(nbf.v4.new_markdown_cell(
            "# แนวคำตอบของแบบฝึกหัดที่ต้องรันโค้ด\nควรลองทำเองก่อน แล้วจึงรันเซลล์เหล่านี้เพื่อเทียบคำตอบ"))
        for p in ex:
            cells += script_cells(chdir, p)
    text = "\n".join("".join(c.source) for c in cells)
    done = set(re.findall(r"โค้ดที่ ([0-9.]+)", text))
    left = [l for l, v in NUM.items() if v.split(".")[0] == str(n) and v not in done]
    if left:
        rows = []
        for l in sorted(left, key=lambda l: [int(x) for x in NUM[l].split(".")]):
            files = sorted({q.relative_to(chdir).as_posix() for q in chdir.rglob("*")
                            if q.is_file() and q.suffix in (".py", ".sh", ".txt") and l in q.read_text(encoding="utf-8", errors="ignore")
                            and q.name != "expected_output.txt"})
            files = files or EXTRA_FILES.get(l, [])
            rows.append(f"| โค้ดที่ {NUM[l]} | {', '.join('`'+f+'`' for f in files) or 'แสดงในตำราเท่านั้น'} |")
        cells.append(nbf.v4.new_markdown_cell(
            "# ตัวอย่างโค้ดที่ไม่ได้รันในโน้ตบุ๊กนี้\n"
            "เป็นคำสั่งเชลล์ ไฟล์ตั้งค่า โมดูลที่สคริปต์อื่นเรียกใช้ หรือโค้ดสำหรับข้อมูลจริง เปิดดูได้จากแถบไฟล์ของ Colab\n\n"
            "| ตัวอย่างโค้ด | ไฟล์ |\n|---|---|\n" + "\n".join(rows)))
    if (chdir / "optional").is_dir():
        cells.append(nbf.v4.new_markdown_cell(
            f"# ข้อมูลจริง (ไม่บังคับ)\nสคริปต์สำหรับดาวน์โหลดและอ่านข้อมูลจริงอยู่ใน `{chdir.name}/optional/` "
            f"วิธีดาวน์โหลดและผลที่ควรได้อยู่ใน `{chdir.name}/README.md` ข้อมูลเหล่านี้ไม่ได้อยู่ใน GitHub "
            "เพราะขนาดใหญ่หรือมีเงื่อนไขสัญญาอนุญาต"))
    for i, c in enumerate(cells):          # fixed cell ids: regenerating gives an identical file
        c.id = f"{chdir.name[:4]}-{i:03d}"
    nb.cells = cells
    out = ROOT / "notebooks" / f"{chdir.name}.ipynb"
    nbf.write(nb, out)
    return out, len(cells)


if __name__ == "__main__":
    for d in sorted(ROOT.glob("ch[0-9][0-9]_*")):
        out, k = build(d)
        print(f"{out.relative_to(ROOT)}: {k} cells")

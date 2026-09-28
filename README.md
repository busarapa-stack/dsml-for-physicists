# โค้ดประกอบตำรา "วิทยาศาสตร์ข้อมูลและการเรียนรู้ของเครื่องสำหรับนักฟิสิกส์"

ที่เก็บนี้ (https://github.com/busarapa-stack/dsml-for-physicists) เก็บโค้ดที่ใช้สร้างตัวเลข ตาราง และรูปทุกค่าที่ปรากฏในตำรา แยกตามบท ผู้อ่านรันโค้ดของแต่ละบทได้อิสระโดยไม่ต้องรันบทอื่นก่อน ตัวเลขที่ได้ควรตรงกับตัวเลขในตำรา หากไม่ตรง ให้ตรวจเวอร์ชันของไลบรารีก่อน (ดูหัวข้อ "เมื่อผลไม่ตรงกับตำรา")

## เปิดใน Google Colab (ไม่ต้องติดตั้งอะไร)

โน้ตบุ๊กมีสองแบบ ทั้งสองแบบมีเซลล์เหมือนกัน และหัวเซลล์ **โค้ดที่ N.M** ตรงกับเลขในตำรา

**แบบดึงโค้ดจาก GitHub** (คอลัมน์ที่สาม) เซลล์ **เตรียมโค้ด** ดึงโค้ดล่าสุดจากที่เก็บนี้และติดตั้งเฉพาะไลบรารีที่ Colab ยังไม่มี เหมาะเมื่อเชื่อมต่อ GitHub ได้

**แบบไฟล์เดียว ไม่พึ่ง GitHub** (คอลัมน์สุดท้าย, โฟลเดอร์ `notebooks/standalone/`) โค้ดของบทนั้นและของบทก่อนหน้าที่บทนั้นเรียกใช้บรรจุอยู่ในโน้ตบุ๊กเองในรูปข้อมูลบีบอัด ดาวน์โหลดไฟล์แล้วอัปโหลดขึ้น Colab (File → Upload notebook) หรือเก็บไว้ใน Google Drive แล้วเปิดด้วย Google Colaboratory ก็ได้ แจกให้นิสิตผ่านระบบจัดการการเรียนรู้หรือไดรฟ์ของรายวิชาได้โดยตรง รันเซลล์ **โค้ดประกอบตำรา** และ **เตรียมโค้ด** ก่อนเซลล์อื่น

| บท | เรื่อง | ดึงโค้ดจาก GitHub | ไฟล์เดียว (ไม่พึ่ง GitHub) |
|---|---|---|---|
| 0 | พื้นฐานการเขียนโปรแกรมภาษาไพทอนสำหรับนักฟิสิกส์ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch00_python_basics.ipynb) · ฉบับชั้นเรียน [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch00_python_basics_classroom.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch00_python_basics_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch00_python_basics_standalone.ipynb) |
| 1 | บทนำและภูมิทัศน์ข้อมูลฟิสิกส์ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch01_intro.ipynb) · ฉบับชั้นเรียน [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch01_intro_classroom.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch01_intro_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch01_intro_standalone.ipynb) |
| 2 | การโปรแกรมเชิงวิทยาศาสตร์และการทำงานที่ทำซ้ำได้ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch02_programming.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch02_programming_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch02_programming_standalone.ipynb) |
| 3 | การทำความสะอาดและการเตรียมลักษณะข้อมูลฟิสิกส์ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch03_cleaning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch03_cleaning_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch03_cleaning_standalone.ipynb) |
| 4 | สถิติเชิงอนุมานและการวิเคราะห์ข้อมูลเชิงสำรวจ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch04_statistics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch04_statistics_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch04_statistics_standalone.ipynb) |
| 5 | การถดถอยและแบบจำลองที่มีข้อจำกัดทางฟิสิกส์ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch05_regression.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch05_regression_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch05_regression_standalone.ipynb) |
| 6 | การจำแนกประเภทข้อมูลทางฟิสิกส์ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch06_classification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch06_classification_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch06_classification_standalone.ipynb) |
| 7 | การเรียนรู้แบบไม่มีผู้สอนและการจัดกลุ่ม | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch07_clustering.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch07_clustering_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch07_clustering_standalone.ipynb) |
| 8 | การลดมิติและตัวแปรรวมทางฟิสิกส์ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch08_dimred.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch08_dimred_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch08_dimred_standalone.ipynb) |
| 9 | การประเมินแบบจำลองและการระบุความไม่แน่นอน | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch09_evaluation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch09_evaluation_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch09_evaluation_standalone.ipynb) |
| 10 | โครงข่ายประสาทเทียมและการเรียนรู้ที่สอดคล้องกับฟิสิกส์ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch10_neural_networks.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch10_neural_networks_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch10_neural_networks_standalone.ipynb) |
| 11 | โครงข่ายประสาทแบบคอนโวลูชันสำหรับภาพทางฟิสิกส์ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch11_cnn.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch11_cnn_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch11_cnn_standalone.ipynb) |
| 12 | การบูรณาการโครงงาน จริยธรรม และการทำซ้ำได้ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/ch12_capstone.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/busarapa-stack/dsml-for-physicists/blob/main/notebooks/standalone/ch12_capstone_standalone.ipynb) · [ไฟล์](notebooks/standalone/ch12_capstone_standalone.ipynb) |

ข้อควรทราบ: ใช้ CPU ตามค่าเริ่มต้นของ Colab ไฟล์ที่สร้างขึ้นหายเมื่อ session จบ (คัดลอกไปยัง Google Drive ถ้าต้องการเก็บ) และตัวเลขของบทที่ 10–11 อาจต่างจากตำราเล็กน้อยตามที่อธิบายไว้ใน README ของบทนั้น โน้ตบุ๊กทั้งสองแบบสร้างจากสคริปต์ด้วย `tools/make_colab_notebooks.py` และ `tools/make_standalone_notebooks.py` ถ้าแก้สคริปต์ ให้รันทั้งสองไฟล์ใหม่ตามลำดับนี้ ส่วนฉบับชั้นเรียนของบทที่ 0 และ 1 เขียนแยกด้วยมือ ต้องแก้เอง

## โครงสร้างโฟลเดอร์

```
dsml-for-physicists/       (ในต้นฉบับตำราคือโฟลเดอร์ code/)
├── README.md              ← ไฟล์นี้
├── requirements.txt       ← ไลบรารีที่ต้องติดตั้ง
├── run_chapter.py         ← รันสคริปต์ทุกไฟล์ของบทหนึ่งตามลำดับ
├── notebooks/             ← โน้ตบุ๊กสำหรับ Google Colab บทละหนึ่งไฟล์ (standalone/ = แบบไฟล์เดียว ไม่พึ่ง GitHub)
├── tools/                 ← ตัวสร้างโน้ตบุ๊กและตารางเลขตัวอย่างโค้ด
├── ch00_python_basics/   ← บทปูพื้นฐานภาษาไพทอน (ข้อมูลลูกตุ้มจำลอง)
├── ch01_intro/
│   ├── README.md          ← ตารางจับคู่สคริปต์ ↔ หัวข้อ/ตัวอย่างโค้ดในตำรา
│   ├── 01_*.py, 02_*.py … ← สคริปต์ รันตามลำดับเลข
│   ├── optional/          ← สคริปต์เสริม (เช่น ดาวน์โหลดข้อมูลจริง)
│   ├── data/              ← ข้อมูลที่สคริปต์สร้างขึ้น (ลบทิ้งแล้วสร้างใหม่ได้)
│   ├── outputs/           ← รูปและบันทึกผลที่สคริปต์สร้าง
│   └── expected_output.txt← ผลลัพธ์ที่ได้จากการรันจริงตอนจัดทำตำรา
├── ch02_…/
└── …
```

## สถานะรายบท

| บท | โฟลเดอร์ | สคริปต์หลัก | แบบฝึกหัด | ตรวจกับตำรา |
|---|---|---|---|---|
| 0 พื้นฐานภาษาไพทอน | `ch00_python_basics/` | 7 | 6 รายการ | ตรวจแล้ว (27 ก.ย. 2569) |
| 1 บทนำ | `ch01_intro/` | 7 | – | ตรวจแล้ว รวมข้อมูลจริงจาก GWOSC, CMS และ Materials Project |
| 2 การโปรแกรมเชิงวิทยาศาสตร์ | `ch02_programming/` | 9 | 8 รายการ | ตรวจแล้ว |
| 3 การทำความสะอาดข้อมูลและลักษณะเด่น | `ch03_cleaning/` | 14 | 7 รายการ | ตรวจแล้ว |
| 4 สถิติเชิงอนุมานและ EDA | `ch04_statistics/` | 9 | 7 รายการ | ตรวจแล้ว |
| 5 การถดถอยและแบบจำลองที่มีข้อจำกัดทางฟิสิกส์ | `ch05_regression/` | 7 | 6 รายการ | ตรวจแล้ว |
| 6 การจำแนกประเภท | `ch06_classification/` | 8 | 9 รายการ | ตรวจแล้ว รวมกรณีศึกษา SDSS กับไฟล์ Kaggle จริง |
| 7 การเรียนรู้แบบไม่มีผู้สอนและการจัดกลุ่ม | `ch07_clustering/` | 7 | 6 รายการ | ตรวจแล้วสองรอบ (E7.6 ตรวจกับไฟล์ SDSS จริงด้วย) |
| 8 การลดมิติและตัวแปรรวม | `ch08_dimred/` | 6 | 6 รายการ | ตรวจแล้วสองรอบ (ต้องติดตั้ง umap-learn) |
| 9 การประเมินแบบจำลองและความไม่แน่นอน | `ch09_evaluation/` | 12 | 4 รายการ | ตรวจแล้วสองรอบ (ต้องติดตั้ง pymc และ arviz; ใช้โค้ดของบทที่ 4, 5, 6 และ 8) |
| 10 โครงข่ายประสาทเทียมและ PINN | `ch10_neural_networks/` | 9 | 4 รายการ | ตรวจแล้วสองรอบ (ต้องติดตั้ง torch รุ่น CPU; ใช้โค้ดของบทที่ 4, 5 และ 6) |
| 11 โครงข่ายคอนโวลูชันสำหรับภาพ | `ch11_cnn/` | 8 | 4 รายการ | ตรวจแล้วสองรอบ รวมไฟล์ Galaxy10 DECaLS จริง (ต้องติดตั้ง torch; E11.1 ใช้ข้อมูลของบทที่ 10) |
| 12 โครงงาน การทำซ้ำได้ และจริยธรรม | `ch12_capstone/` | 3 | – | ตรวจแล้วสองรอบ (ใช้ข้อมูลของบทที่ 6) |

## ข้อตกลงที่ใช้ทุกบท

1. **ลำดับการรัน** สคริปต์ในแต่ละบทขึ้นต้นด้วยเลขสองหลัก สคริปต์ที่เลขน้อยกว่าอาจสร้างไฟล์ใน `data/` ที่สคริปต์ถัดไปใช้ จึงควรรันตามลำดับ หรือใช้ `run_chapter.py`
2. **ข้อมูลจำลองกับข้อมูลจริง** ตัวอย่างหลักในตำราใช้ข้อมูลจำลอง (synthetic data) ที่สร้างขึ้นด้วยโค้ดและกำหนดค่าเมล็ดสุ่ม (random seed) ไว้ ผู้อ่านทุกคนจึงได้ผลเหมือนกันโดยไม่ต้องดาวน์โหลดไฟล์ขนาดใหญ่ ส่วนการเปิดข้อมูลจริงจากแหล่งข้อมูลเปิดอยู่ในโฟลเดอร์ `optional/` ของแต่ละบท
3. **เมล็ดสุ่ม** ทุกสคริปต์ที่มีการสุ่มกำหนดค่า `SEED` ไว้ที่ต้นไฟล์ การเปลี่ยนค่านี้เป็นแบบฝึกหัดที่ดีในการดูว่าผลเปลี่ยนมากเพียงใด (ดูบทที่ 12)
4. **ที่อยู่ไฟล์** ทุกสคริปต์อ้างถึง `data/` และ `outputs/` เทียบกับตำแหน่งของสคริปต์เอง จึงรันจากโฟลเดอร์ใดก็ได้
5. **หมายเลขตัวอย่างโค้ด** README ของแต่ละบทระบุเลขโค้ดตามที่พิมพ์ในตำรา เช่น โค้ดที่ 1.1 พร้อมชื่ออ้างอิงภายในของต้นฉบับ LaTeX ในวงเล็บ เช่น (`code:root-tree`) ชื่อหลังนี้เขียนไว้ในหัวไฟล์ของสคริปต์ที่เกี่ยวข้องด้วย จึงค้นด้วย `grep code:root-tree` ได้ เลขโค้ดอ้างอิงจากต้นฉบับฉบับพิมพ์ครั้งที่ 1 ถ้าเพิ่มหรือลบตัวอย่างโค้ดในตำรา ต้องปรับเลขใน README ตาม

## การติดตั้ง

ต้องใช้ Python 3.10 ขึ้นไป แนะนำให้สร้างสภาพแวดล้อมเสมือน (virtual environment) แยกไว้สำหรับตำรานี้

```bash
git clone https://github.com/busarapa-stack/dsml-for-physicists.git
cd dsml-for-physicists
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## การรัน

รันทั้งบท:

```bash
python run_chapter.py ch01_intro
python run_chapter.py ch02_programming --with-exercises   # รวมสคริปต์ในโฟลเดอร์ exercises/
```

รันทีละสคริปต์:

```bash
cd ch01_intro
python 01_make_sample_files.py
python 02_inspect_root.py
```

## เมื่อผลไม่ตรงกับตำรา

ตัวเลขในตำราได้จากการรันด้วยเวอร์ชันที่ระบุใน `expected_output.txt` ของแต่ละบท ความต่างในหลักทศนิยมท้าย ๆ อาจเกิดจากเวอร์ชันของ NumPy หรือ scikit-learn ที่ต่างกัน ซึ่งเป็นเรื่องปกติ แต่หากต่างกันมาก ให้ตรวจว่าไม่ได้แก้ค่า `SEED` และรันสคริปต์ตามลำดับแล้ว

## สัญญาอนุญาต

โค้ดในที่เก็บนี้เผยแพร่ภายใต้สัญญาอนุญาต MIT (ดูไฟล์ `LICENSE`) ข้อมูลจริงที่สคริปต์ใน `optional/` ดาวน์โหลดมาเป็นของเจ้าของแหล่งข้อมูล และมีเงื่อนไขการใช้งานตามตารางสัญญาอนุญาตในบทที่ 1

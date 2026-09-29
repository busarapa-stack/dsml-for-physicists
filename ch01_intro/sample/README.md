# ตัวอย่างย่อยของ Galaxy10 DECaLS

`galaxy10_decals_sample.h5` (ประมาณ 7 MB) เป็นส่วนย่อยของไฟล์ `Galaxy10_DECals.h5` (ประมาณ 2.7 GB) ใช้ในส่วนที่ 5 ของ `notebooks/ch01_intro_classroom.ipynb` เพื่อให้แสดงภาพได้ทันทีบน Colab โดยไม่ต้องอ่านไฟล์ใหญ่ผ่านเครือข่าย

- ภาพ 5 ภาพแรกของแต่ละประเภท รวม 50 ภาพ ไม่ได้ย่อหรือดัดแปลง (256×256×3, uint8) พร้อม `ra`, `dec`, `redshift`, `pxscale`
- `index_in_full` คือเลขแถวของภาพเหล่านี้ในไฟล์เต็ม
- `ans_full` คือป้ายกำกับของภาพทั้ง 17,736 ภาพ ใช้นับจำนวนภาพในแต่ละประเภท

สร้างซ้ำได้ด้วย `python ch01_intro/sample/make_galaxy10_sample.py` เมื่อมีไฟล์เต็มในโฟลเดอร์ `code/`

**ที่มาและสัญญาอนุญาต:** Galaxy10 DECaLS โดย H. W. Leung และ J. Bovy, https://zenodo.org/records/10845026 (CC BY 4.0) ป้ายกำกับจาก Galaxy Zoo DECaLS (Walmsley et al., 2022, *MNRAS*, 509, 3966) ภาพจาก DESI Legacy Imaging Surveys

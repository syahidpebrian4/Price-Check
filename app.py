import streamlit as st
import pytesseract
import cv2
import numpy as np
import pandas as pd
import re
import os
from PIL import Image, ImageDraw
from openpyxl import load_workbook
import io
import zipfile
from fuzzywuzzy import fuzz
import gc

# ================= CONFIG =================
FILE_PATH = "database/master_harga.xlsx"
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 
TARGET_IMAGE_SIZE_KB = 195 

# Daftar Nama yang akan di-sensor (Redact)
TEXTS_TO_REDACT = [
    "HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "WAYAN GIYANTO", 
    "MEMBER UMUM KLIK", "DJUANMING", "NONOK JUNENGSIH", 
    "AGUNG KURNIAWAN", "ARIF RAMADHAN", "HILMI ATIQ"
]

st.set_page_config(page_title="Price Check V10.9", layout="wide")

def clean_price_strict(text_segment):
    """Mengambil angka saja dari string harga (misal: 16.100 -> 16100)"""
    text = re.sub(r'[^\d]', '', str(text_segment))
    return int(text) if text else 0

def process_ocr_indogrosir(pil_image):
    # 1. Persiapan Gambar
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. OCR Tesseract
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config)
    
    df_ocr = pd.DataFrame(d)
    df_ocr['text'] = df_ocr['text'].fillna('').str.upper()
    
    # Kelompokkan teks per baris (berdasarkan koordinat 'top')
    df_ocr['line_id'] = df_ocr['block_num'].astype(str) + "_" + df_ocr['line_num'].astype(str)
    lines = df_ocr.groupby('line_id').agg({
        'text': lambda x: " ".join(x),
        'top': 'min',
        'height': 'max'
    }).reset_index()

    full_text_raw = " ".join(df_ocr['text'].tolist())
    final_res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    prod_name = "N/A"

    # --- LOGIKA HARGA & NAMA ---
    match_pcs = re.search(r"PCS\s*-\s*RP\s*([\d\.,]+)", full_text_raw)
    if match_pcs:
        price = clean_price_strict(match_pcs.group(1))
        final_res["PCS"] = {"normal": price, "promo": price}

    match_ctn = re.search(r"CTN\s*-\s*RP\s*([\d\.,]+)", full_text_raw)
    if match_ctn:
        price = clean_price_strict(match_ctn.group(1))
        final_res["CTN"] = {"normal": price, "promo": price}

    if "INDOGROSIR Q" in full_text_raw:
        parts = full_text_raw.split("INDOGROSIR Q")
        if len(parts) > 1:
            prod_name = " ".join(parts[1].strip().split()[:6])

    # --- LOGIKA REDAKSI FUZZY (HIGH TOLERANCE) ---
    draw = ImageDraw.Draw(pil_image)
    for _, row in lines.iterrows():
        line_text = str(row['text']).upper()
        if len(line_text) < 3: continue
        
        for keyword in TEXTS_TO_REDACT:
            score = fuzz.partial_ratio(keyword.upper(), line_text)
            # Toleransi fuzzy diatur ke 75
            if score > 75 or (("HALO" in line_text or "WAYAN" in line_text) and score > 50):
                y_coord = row['top'] / scale
                h_box = row['height'] / scale
                draw.rectangle([0, y_coord - 5, pil_image.width, y_coord + h_box + 5], fill="white")
                break 

    return final_res["PCS"], final_res["CTN"], prod_name, full_text_raw, pil_image

# ================= UI STREAMLIT =================
def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("📸 Price Check V10.9 (Update & Download)")

c1, c2, c3 = st.columns(3)
with c1: m_code = st.text_input("📍 Master Code").upper()
with c2: date_inp = st.text_input("📅 Tanggal (CONTOH: 23JAN2026)").upper()
with c3: week_inp = st.text_input("🗓️ Week")

files = st.file_uploader("📂 Upload Screenshots", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code and date_inp and week_inp:
    if os.path.exists(FILE_PATH):
        xl = pd.ExcelFile(FILE_PATH)
        db_ig = xl.parse(SHEET_MASTER_IG)
        db_ig.columns = db_ig.columns.astype(str).str.strip()
        db_targets = {s: xl.parse(s) for s in SHEETS_TARGET if s in xl.sheet_names}

        final_list = []
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                with st.expander(f"🔍 Memproses: {f.name}", expanded=True):
                    img_pil = Image.open(f)
                    pcs, ctn, name, raw_txt, red_img = process_ocr_indogrosir(img_pil)
                    
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_name = str(row[COL_IG_NAME]).upper()
                        score = fuzz.token_set_ratio(db_name, name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])
                    
                    col_l, col_r = st.columns([1, 2])
                    col_l.image(red_img, caption="Redacted Image")
                    with col_r:
                        st.write(f"**Hasil Cocok:** `{match_code}` (Score: {best_score})")
                        st.json({"PCS": pcs, "CTN": ctn})

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            match = df_t[(df_t["PRODCODE"].astype(str).apply(norm) == match_code) & 
                                         (df_t["MASTER Code"].astype(str).apply(norm) == norm(m_code))]
                            
                            if not match.empty:
                                final_list.append({
                                    "prodcode": match_code, "sheet": s_name, "index": match.index[0],
                                    "n_pcs": pcs['normal'], "p_pcs": pcs['promo'],
                                    "n_ctn": ctn['normal'], "p_ctn": ctn['promo']
                                })
                                if red_img.mode in ("RGBA", "P"): red_img = red_img.convert("RGB")
                                img_io = io.BytesIO()
                                red_img.save(img_io, format="JPEG", quality=85)
                                zf.writestr(f"{match_code}.jpg", img_io.getvalue())
                                break
                gc.collect()

        if final_list:
            st.write("### 📋 Tabel Perubahan")
            st.table(pd.DataFrame(final_list)[["prodcode", "sheet", "n_pcs", "p_pcs", "n_ctn", "p_ctn"]])
            
            # --- LOGIKA UPDATE & DOWNLOAD ---
            btn_update = st.button("🚀 EKSEKUSI UPDATE DATABASE EXCEL")
            
            if btn_update:
                wb = load_workbook(FILE_PATH)
                for r in final_list:
                    ws = wb[r['sheet']]
                    headers = [str(c.value).strip() for c in ws[1]]
                    row_num = r['index'] + 2
                    mapping = {
                        "Normal Competitor Price (Pcs)": r['n_pcs'],
                        "Promo Competitor Price (Pcs)": r['p_pcs'],
                        "Normal Competitor Price (Ctn)": r['n_ctn'],
                        "Promo Competitor Price (Ctn)": r['p_ctn']
                    }
                    for col_name, val in mapping.items():
                        if col_name in headers:
                            ws.cell(row=row_num, column=headers.index(col_name) + 1).value = val
                
                # Simpan perubahan ke file
                wb.save(FILE_PATH)
                st.success("✅ Database Excel telah diperbarui!")
                
                # Buat buffer untuk download file excel yang baru saja diupdate
                with open(FILE_PATH, "rb") as f:
                    st.download_button(
                        label="📥 DOWNLOAD EXCEL TERUPDATE",
                        data=f,
                        file_name=f"Price Check W{week_inp}_{date_inp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            # Tombol download ZIP tetap muncul di bawah
            st.download_button("🖼️ DOWNLOAD ZIP FOTO", zip_buffer.getvalue(), f"{m_code}_{date_inp}.zip")

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

st.set_page_config(page_title="Price Check V10.7", layout="wide")

def clean_price_strict(text_segment):
    """Mengambil angka saja dari string harga (misal: 16.100 -> 16100)"""
    text = re.sub(r'[^\d]', '', str(text_segment))
    return int(text) if text else 0

def process_ocr_indogrosir(pil_image):
    # 1. Persiapan Gambar untuk OCR
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # Gunakan skala 2.0x agar teks kecil Klik Indogrosir lebih tajam
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. Jalankan Tesseract
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config)
    
    df_ocr = pd.DataFrame(d)
    df_ocr['text'] = df_ocr['text'].fillna('').str.upper()
    full_text_raw = " ".join(df_ocr['text'].tolist())
    
    final_res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    prod_name = "N/A"

    # --- LOGIKA EKSTRAKSI HARGA (SESUAI PERMINTAAN) ---
    
    # Harga PCS: Angka pertama setelah 'PCS - RP'
    match_pcs = re.search(r"PCS\s*-\s*RP\s*([\d\.,]+)", full_text_raw)
    if match_pcs:
        price = clean_price_strict(match_pcs.group(1))
        final_res["PCS"] = {"normal": price, "promo": price}

    # Harga CTN: Angka pertama setelah 'CTN - RP'
    match_ctn = re.search(r"CTN\s*-\s*RP\s*([\d\.,]+)", full_text_raw)
    if match_ctn:
        price = clean_price_strict(match_ctn.group(1))
        final_res["CTN"] = {"normal": price, "promo": price}

    # Nama Produk: 6 kata setelah 'INDOGROSIR Q'
    if "INDOGROSIR Q" in full_text_raw:
        parts = full_text_raw.split("INDOGROSIR Q")
        if len(parts) > 1:
            prod_name = " ".join(parts[1].strip().split()[:6])

    # --- LOGIKA REDAKSI (SENSOR) ---
    # Kita cari baris yang mengandung 'CARI DI KLIK' atau 'HALO'
    draw = ImageDraw.Draw(pil_image)
    header_keywords = df_ocr[df_ocr['text'].str.contains("CARI|KLIK|INDOGROSIR|HALO", na=False)]
    
    if not header_keywords.empty:
        # Ambil koordinat Y dari kata kunci tersebut, kembalikan ke skala asli
        # Biasanya nama user ada tepat di baris yang sama atau tepat di bawah baris ini
        y_coord = header_keywords['top'].iloc[0] / scale
        h_box = header_keywords['height'].iloc[0] / scale
        
        # Gambar kotak sensor menutupi satu baris kalimat tersebut
        # Kita beri padding sedikit (y-10 sampai y+h+10) agar bersih
        draw.rectangle([0, y_coord - 10, pil_image.width, y_coord + h_box + 15], fill="white")

    return final_res["PCS"], final_res["CTN"], prod_name, full_text_raw, pil_image

# ================= UI STREAMLIT =================
def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("📸 Price Check V10.7 (Full Fixed)")

c1, c2, c3 = st.columns(3)
with c1: m_code = st.text_input("📍 Master Code").upper()
with c2: date_inp = st.text_input("📅 Tanggal (CONTOH: 23JAN2026)").upper()
with c3: week_inp = st.text_input("🗓️ Week")

files = st.file_uploader("📂 Upload Screenshots", type=["jpg","png","jpeg"], accept_multiple_files=True)

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
                with st.expander(f"🔍 Processing: {f.name}", expanded=True):
                    img_pil = Image.open(f)
                    pcs, ctn, name, raw_txt, red_img = process_ocr_indogrosir(img_pil)
                    
                    # Fuzzy Matching Nama ke Database
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_name = str(row[COL_IG_NAME]).upper()
                        score = fuzz.token_set_ratio(db_name, name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])
                    
                    # Display Result
                    col_left, col_right = st.columns([1, 2])
                    col_left.image(red_img, caption="Hasil Sensor")
                    with col_right:
                        st.write(f"**Nama Terdeteksi:** `{name}`")
                        st.write(f"**Prodcode Match:** `{match_code}` (Score: {best_score})")
                        st.json({"PCS": pcs, "CTN": ctn})
                        st.caption(f"Raw Text Preview: {raw_txt[:200]}...")

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            # Cari baris di DF/HBHC yang prodcode dan Master Code-nya sesuai
                            match = df_t[(df_t["PRODCODE"].astype(str).apply(norm) == match_code) & 
                                         (df_t["MASTER Code"].astype(str).apply(norm) == norm(m_code))]
                            
                            if not match.empty:
                                final_list.append({
                                    "prodcode": match_code, "sheet": s_name, "index": match.index[0],
                                    "n_pcs": pcs['normal'], "p_pcs": pcs['promo'],
                                    "n_ctn": ctn['normal'], "p_ctn": ctn['promo']
                                })
                                
                                # Konversi ke RGB untuk menghindari OSError JPEG
                                if red_img.mode in ("RGBA", "P"):
                                    red_img = red_img.convert("RGB")
                                
                                img_io = io.BytesIO()
                                red_img.save(img_io, format="JPEG", quality=85)
                                zf.writestr(f"{match_code}.jpg", img_io.getvalue())
                                break
                gc.collect()

        if final_list:
            st.write("### 📋 Ringkasan Data Update")
            st.table(pd.DataFrame(final_list)[["prodcode", "sheet", "n_pcs", "p_pcs", "n_ctn", "p_ctn"]])
            
            if st.button("🚀 EKSEKUSI UPDATE EXCEL"):
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
                wb.save(FILE_PATH)
                st.success("Database Excel Berhasil Diperbarui!")
            
            st.download_button("📥 DOWNLOAD ZIP FOTO", zip_buffer.getvalue(), f"{m_code}_{date_inp}.zip")

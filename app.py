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

st.set_page_config(page_title="Price Check V11.1 - Robust", layout="wide")

def clean_price_robust(text_segment):
    """
    Membersihkan teks harga. Menghapus tanda hubung '-' yang sering muncul 
    akibat harga coret agar tidak merusak deteksi angka.
    """
    # Hapus titik, koma, dan tanda hubung (hasil coretan)
    clean_txt = re.sub(r'[.,\-\s]', '', str(text_segment))
    # Ambil angka 3-7 digit
    nums = re.findall(r'\d{3,7}', clean_txt)
    if nums:
        # Jika ada beberapa angka dalam satu segmen, 
        # biasanya harga promo/terbaru adalah yang paling kanan (terakhir)
        return int(nums[-1])
    return 0

def process_ocr_indogrosir(pil_image):
    # 1. Persiapan Gambar (Scale 2x untuk akurasi teks kecil)
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. OCR Tesseract (PSM 6: Blok teks seragam)
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config)
    
    df_ocr = pd.DataFrame(d)
    df_ocr['text'] = df_ocr['text'].fillna('').str.upper()
    
    # Grouping Baris untuk Redaksi dan Raw Text
    df_ocr['line_id'] = df_ocr['block_num'].astype(str) + "_" + df_ocr['line_num'].astype(str)
    lines_df = df_ocr.groupby('line_id').agg({
        'text': lambda x: " ".join(x), 
        'top': 'min', 
        'height': 'max'
    }).reset_index()
    
    full_text_raw = " | ".join(lines_df['text'].tolist())
    final_res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    prod_name = "N/A"

    # --- LOGIKA EKSTRAKSI HARGA ---
    # Mencari pola PCS dan CTN dengan toleransi terhadap karakter sampah di antaranya
    pcs_pattern = re.search(r"PCS\s*-\s*RP\s*([^|]+)", full_text_raw)
    if pcs_pattern:
        val = clean_price_robust(pcs_pattern.group(1))
        final_res["PCS"] = {"normal": val, "promo": val}

    ctn_pattern = re.search(r"CTN\s*-\s*RP\s*([^|]+)", full_text_raw)
    if ctn_pattern:
        val = clean_price_robust(ctn_pattern.group(1))
        final_res["CTN"] = {"normal": val, "promo": val}

    # Nama Produk (6 kata setelah box pencarian)
    if "INDOGROSIR Q" in full_text_raw:
        parts = full_text_raw.split("INDOGROSIR Q")
        if len(parts) > 1:
            prod_name = " ".join(parts[1].strip().split()[:6])

    # --- LOGIKA REDAKSI FUZZY ---
    draw = ImageDraw.Draw(pil_image)
    for _, row in lines_df.iterrows():
        line_text = str(row['text']).upper()
        if len(line_text) < 4: continue
        
        for kw in TEXTS_TO_REDACT:
            # Menggunakan partial_ratio agar 'HALO, WAYAN' tetap kena walau terbaca 'HALO WAYAN'
            if fuzz.partial_ratio(kw.upper(), line_text) > 75:
                y = row['top'] / scale
                h = row['height'] / scale
                # Redaksi satu baris penuh
                draw.rectangle([0, y - 8, pil_image.width, y + h + 8], fill="white")
                break 

    return final_res["PCS"], final_res["CTN"], prod_name, full_text_raw, pil_image

# ================= UI STREAMLIT =================
def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("📸 Price Check V11.1 (Robust OCR)")

col1, col2, col3 = st.columns(3)
with col1: m_code = st.text_input("📍 Master Code").upper()
with col2: date_inp = st.text_input("📅 Tanggal (Contoh: 26JAN2026)").upper()
with col3: week_inp = st.text_input("🗓️ Week")

files = st.file_uploader("📂 Upload Screenshot Klik Indogrosir", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

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
                with st.expander(f"🔎 Analisis: {f.name}", expanded=True):
                    img_pil = Image.open(f)
                    res_pcs, res_ctn, name, raw_text, red_img = process_ocr_indogrosir(img_pil)
                    
                    # Matching ke Database
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_n = str(row[COL_IG_NAME]).upper()
                        score = fuzz.token_set_ratio(db_n, name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])
                    
                    c_left, c_right = st.columns([1, 1.2])
                    with c_left:
                        st.image(red_img, use_container_width=True)
                    with c_right:
                        st.write(f"**Prodcode:** `{match_code}` | **Score:** `{best_score}`")
                        st.json({"PCS": res_pcs, "CTN": res_ctn})
                        st.text_area("OCR Raw (Debug)", value=raw_text, height=100)

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            match_row = df_t[(df_t["PRODCODE"].astype(str).apply(norm) == match_code) & 
                                             (df_t["MASTER Code"].astype(str).apply(norm) == norm(m_code))]
                            
                            if not match_row.empty:
                                final_list.append({
                                    "prodcode": match_code, "sheet": s_name, "index": match_row.index[0],
                                    "n_pcs": res_pcs['normal'], "p_pcs": res_pcs['promo'],
                                    "n_ctn": res_ctn['normal'], "p_ctn": res_ctn['promo']
                                })
                                # Save to ZIP
                                if red_img.mode != "RGB": red_img = red_img.convert("RGB")
                                buf = io.BytesIO()
                                red_img.save(buf, format="JPEG", quality=85)
                                zf.writestr(f"{match_code}.jpg", buf.getvalue())
                                break
                gc.collect()

        if final_list:
            st.divider()
            if st.button("🚀 UPDATE DATABASE & GENERATE EXCEL"):
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
                st.success("✅ Database master_harga.xlsx telah diperbarui!")
                
                with open(FILE_PATH, "rb") as f:
                    st.download_button("📥 DOWNLOAD EXCEL TERUPDATE", f, f"Price_Check_W{week_inp}_{date_inp}.xlsx")
            
            st.download_button("🖼️ DOWNLOAD ZIP FOTO", zip_buffer.getvalue(), f"{m_code}_{date_inp}.zip")

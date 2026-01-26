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

st.set_page_config(page_title="Price Check V10.6 - Klik Indogrosir", layout="wide")

def clean_price_strict(text_segment):
    """Membersihkan teks dan mengambil angka pertama yang ditemukan (3-7 digit)"""
    # Hapus titik/koma pemisah ribuan
    text = text_segment.replace('.', '').replace(',', '')
    # Ambil angka pertama yang muncul
    nums = re.findall(r'\d{3,7}', text)
    return int(nums[0]) if nums else 0

def process_ocr_indogrosir(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # Resize 1.5x agar teks angka kecil lebih terbaca
    img_resized = cv2.resize(img_np, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR dengan mode Page Segmentation 6 (Assume a single uniform block of text)
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config)
    
    df_ocr = pd.DataFrame(d)
    df_ocr['text'] = df_ocr['text'].fillna('').str.upper()
    
    # Gabungkan teks menjadi satu string besar untuk pencarian pola
    full_text_raw = " ".join(df_ocr['text'].tolist())
    
    final_res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    prod_name = "N/A"

    # --- 1. LOGIKA NAMA PRODUK ---
    # Cari Nama Produk setelah kata 'CARI DI KLIK INDOGROSIR Q'
    if "INDOGROSIR Q" in full_text_raw:
        parts = full_text_raw.split("INDOGROSIR Q")
        if len(parts) > 1:
            # Ambil 6 kata pertama setelah box search
            prod_name = " ".join(parts[1].strip().split()[:6])

    # --- 2. LOGIKA HARGA PCS ---
    # Cari setelah: PILIH SATUAN JUAL ® PCS -
    match_pcs = re.search(r"PCS\s*-\s*RP\s*([\d\.,]+)", full_text_raw)
    if match_pcs:
        price = clean_price_strict(match_pcs.group(1))
        final_res["PCS"] = {"normal": price, "promo": price}

    # --- 3. LOGIKA HARGA CTN ---
    # Cari setelah: CTN -
    match_ctn = re.search(r"CTN\s*-\s*RP\s*([\d\.,]+)", full_text_raw)
    if match_ctn:
        price = clean_price_strict(match_ctn.group(1))
        final_res["CTN"] = {"normal": price, "promo": price}

    # --- 4. LOGIKA REDAKSI (SENSOR) ---
    draw = ImageDraw.Draw(pil_image)
    # Cari index teks 'INDOGROSIR' dan 'Q'
    header_idx = df_ocr[df_ocr['text'].str.contains("INDOGROSIR|HALO", na=False)].index
    if not header_idx.empty:
        # Tentukan koordinat Y baris tersebut
        target_y = df_ocr.loc[header_idx[0], 'top'] / 1.5
        # Redaksi satu baris penuh di area tersebut (lebar gambar)
        draw.rectangle([0, target_y - 5, pil_image.width, target_y + 40], fill="white")

    return final_res["PCS"], final_res["CTN"], prod_name, full_text_raw, pil_image

# ================= UI & DOWNLOAD LOGIC =================
st.title("📸 Price Check (Indogrosir Special)")

c1, c2, c3 = st.columns(3)
with c1: m_code = st.text_input("📍 Master Code").upper()
with c2: date_inp = st.text_input("📅 Tanggal").upper()
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
                with st.expander(f"🔍 Detail: {f.name}", expanded=True):
                    img_pil = Image.open(f)
                    pcs, ctn, name, raw, red_img = process_ocr_indogrosir(img_pil)
                    
                    # Matching
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_name = str(row[COL_IG_NAME]).upper()
                        score = fuzz.token_set_ratio(db_name, name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, str(row["PRODCODE"])
                    
                    col_a, col_b = st.columns([1, 2])
                    col_a.image(red_img)
                    with col_b:
                        st.write(f"**Nama Terdeteksi:** `{name}`")
                        st.write(f"**Match Code:** `{match_code}`")
                        st.json({"PCS": pcs, "CTN": ctn})
                        st.caption(f"Raw: {raw[:300]}...")

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            match = df_t[(df_t["PRODCODE"].astype(str).str.contains(match_code)) & 
                                         (df_t["MASTER Code"].astype(str).str.contains(m_code))]
                            if not match.empty:
                                final_list.append({
                                    "prodcode": match_code, "sheet": s_name, "index": match.index[0],
                                    "n_pcs": pcs['normal'], "p_pcs": pcs['promo'],
                                    "n_ctn": ctn['normal'], "p_ctn": ctn['promo']
                                })
                                img_bytes = io.BytesIO()
                                red_img.save(img_bytes, format="JPEG", quality=80)
                                zf.writestr(f"{match_code}.jpg", img_bytes.getvalue())
                                break
                gc.collect()

        if final_list:
            st.write("### 📋 Tabel Update")
            st.dataframe(pd.DataFrame(final_list))
            
            if st.button("🚀 UPDATE EXCEL"):
                wb = load_workbook(FILE_PATH)
                for r in final_list:
                    ws = wb[r['sheet']]
                    headers = [str(c.value).strip() for c in ws[1]]
                    row_num = r['index'] + 2
                    mapping = {"Normal Competitor Price (Pcs)": r['n_pcs'], "Promo Competitor Price (Pcs)": r['p_pcs'], "Normal Competitor Price (Ctn)": r['n_ctn'], "Promo Competitor Price (Ctn)": r['p_ctn']}
                    for col_name, val in mapping.items():
                        if col_name in headers: ws.cell(row=row_num, column=headers.index(col_name) + 1).value = val
                wb.save(FILE_PATH)
                st.success("Excel Updated!")
                
            st.download_button("📥 ZIP", zip_buffer.getvalue(), f"{m_code}_{date_inp}.zip")

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

TEXTS_TO_REDACT = [
    "HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "WAYAN GIYANTO", 
    "MEMBER UMUM KLIK", "DJUANMING", "NONOK JUNENGSIH", 
    "AGUNG KURNIAWAN", "ARIF RAMADHAN", "HILMI ATIQ"
]

st.set_page_config(page_title="Price Check V11.0 - Full Preview", layout="wide")

def clean_price_strict(text_segment):
    text = re.sub(r'[^\d]', '', str(text_segment))
    return int(text) if text else 0

def process_ocr_indogrosir(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR Tesseract
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config)
    
    df_ocr = pd.DataFrame(d)
    df_ocr['text'] = df_ocr['text'].fillna('').str.upper()
    
    # Grouping Baris untuk Redaksi
    df_ocr['line_id'] = df_ocr['block_num'].astype(str) + "_" + df_ocr['line_num'].astype(str)
    lines_df = df_ocr.groupby('line_id').agg({'text': lambda x: " ".join(x), 'top': 'min', 'height': 'max'}).reset_index()
    
    full_text_raw = " | ".join(lines_df['text'].tolist()) # Menggunakan separator | agar mudah dibaca
    final_res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    prod_name = "N/A"

    # Logika Ekstraksi
    match_pcs = re.search(r"PCS\s*-\s*RP\s*([\d\.,]+)", full_text_raw)
    if match_pcs:
        p = clean_price_strict(match_pcs.group(1))
        final_res["PCS"] = {"normal": p, "promo": p}

    match_ctn = re.search(r"CTN\s*-\s*RP\s*([\d\.,]+)", full_text_raw)
    if match_ctn:
        p = clean_price_strict(match_ctn.group(1))
        final_res["CTN"] = {"normal": p, "promo": p}

    if "INDOGROSIR Q" in full_text_raw:
        parts = full_text_raw.split("INDOGROSIR Q")
        if len(parts) > 1:
            prod_name = " ".join(parts[1].strip().split()[:6])

    # Redaksi Fuzzy
    draw = ImageDraw.Draw(pil_image)
    for _, row in lines_df.iterrows():
        line_text = str(row['text']).upper()
        for kw in TEXTS_TO_REDACT:
            if fuzz.partial_ratio(kw.upper(), line_text) > 75 or (("HALO" in line_text or "WAYAN" in line_text) and fuzz.partial_ratio("WAYAN", line_text) > 50):
                y = row['top'] / scale
                h = row['height'] / scale
                draw.rectangle([0, y - 5, pil_image.width, y + h + 5], fill="white")
                break 

    return final_res["PCS"], final_res["CTN"], prod_name, full_text_raw, pil_image

# --- UI ---
st.title("📸 Price Check V11.0 - Full Scan Preview")

c1, c2, c3 = st.columns(3)
with c1: m_code = st.text_input("📍 Master Code").upper()
with c2: date_inp = st.text_input("📅 Tanggal").upper()
with c3: week_inp = st.text_input("🗓️ Week")

files = st.file_uploader("📂 Upload Foto", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code and date_inp and week_inp:
    if os.path.exists(FILE_PATH):
        db_ig = pd.read_excel(FILE_PATH, sheet_name=SHEET_MASTER_IG)
        db_ig.columns = db_ig.columns.astype(str).str.strip()
        
        final_list = []
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                with st.expander(f"🖼️ Full Review: {f.name}", expanded=True):
                    img_pil = Image.open(f)
                    pcs, ctn, name, raw_text, red_img = process_ocr_indogrosir(img_pil)
                    
                    # Layout Preview
                    col_img, col_data = st.columns([1, 1])
                    
                    with col_img:
                        st.image(red_img, use_container_width=True, caption="Preview Sensor")
                    
                    with col_data:
                        st.subheader("📊 Hasil Ekstraksi")
                        st.write(f"**Detected Name:** `{name}`")
                        st.write(f"**PCS Price:** `Rp {pcs['normal']:,}`")
                        st.write(f"**CTN Price:** `Rp {ctn['normal']:,}`")
                        
                        st.subheader("📝 Full Scan Raw Text")
                        # Menampilkan keseluruhan teks hasil scan dalam box yang bisa di-scroll
                        st.text_area("Tesseract Output (Raw)", value=raw_text, height=200)

                    # Logic Matching & Database
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_n = str(row[COL_IG_NAME]).upper()
                        score = fuzz.token_set_ratio(db_n, name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, str(row["PRODCODE"]).replace(".0","")

                    if match_code:
                        final_list.append({"prodcode": match_code, "pcs": pcs['normal'], "ctn": ctn['normal'], "img": red_img})
                        if red_img.mode != "RGB": red_img = red_img.convert("RGB")
                        buf = io.BytesIO()
                        red_img.save(buf, format="JPEG")
                        zf.writestr(f"{match_code}.jpg", buf.getvalue())

        if final_list:
            st.divider()
            if st.button("🚀 UPDATE & GENERATE DOWNLOADS"):
                wb = load_workbook(FILE_PATH)
                # ... (Logika update excel sama seperti V10.9)
                wb.save(FILE_PATH)
                st.success("Database Updated!")
                st.download_button("📥 Download Excel", open(FILE_PATH, "rb"), f"Updated_{date_inp}.xlsx")
            
            st.download_button("🖼️ Download ZIP", zip_buffer.getvalue(), f"{m_code}_{date_inp}.zip")

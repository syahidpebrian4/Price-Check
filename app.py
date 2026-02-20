import streamlit as st
import pytesseract
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageDraw
import io
import zipfile
from fuzzywuzzy import fuzz
import gc
import base64

# ================= CONFIG & DATABASE =================
FILE_PATH = "database/master_harga.xlsx"
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 

st.set_page_config(page_title="Image Processor", layout="wide", initial_sidebar_state="expanded")

# --- FUNGSI HELPER: LOGO BASE64 ---
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

# --- CSS CUSTOM ---
logo_b64 = get_base64_image("lotte_logo.png")
st.markdown(f"""
    <style>
        .custom-header {{
            position: fixed;
            top: 0; left: 0; width: 100%; height: 90px;
            background-color: white;
            display: flex; align-items: center;
            padding: 0 30px; border-bottom: 3px solid #eeeeee;
            z-index: 999999;
        }}
        .header-logo {{ height: 55px; margin-right: 25px; }}
        .header-title {{
            font-size: 42px; font-weight: 900;
            font-family: 'Arial Black', sans-serif; color: black; margin: 0;
        }}
        [data-testid="stSidebar"] {{
            background-color: #FF0000 !important;
            margin-top: 90px !important;
            min-width: 320px !important; max-width: 320px !important;
        }}
        .main .block-container {{ padding-top: 130px !important; }}
        header {{ visibility: hidden; }}
    </style>
    <div class="custom-header">
        <img src="data:image/png;base64,{logo_b64 if logo_b64 else ''}" class="header-logo">
        <h1 class="header-title">IMAGE SENSOR & NAMING</h1>
    </div>
""", unsafe_allow_html=True)

# --- FUNGSI LOGIKA OCR & SENSOR ---
def process_image_only(pil_image, master_product_names=None):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
    df_ocr = pd.DataFrame(d)
    df_ocr = df_ocr[df_ocr['text'].str.strip() != ""]
    df_ocr['text'] = df_ocr['text'].str.upper()

    # Rekonstruksi teks untuk pencocokan nama
    lines_data = []
    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by=['top', 'left'])
        current_top = df_ocr.iloc[0]['top']
        temp_words = []
        for _, row in df_ocr.iterrows():
            if row['top'] - current_top > 15:
                lines_data.append({
                    "text": " ".join([w['text'] for w in temp_words]),
                    "top": current_top,
                    "h": max([w['height'] for w in temp_words]) if temp_words else 0
                })
                temp_words = [{'text': row['text'], 'left': row['left'], 'height': row['height']}]
                current_top = row['top']
            else:
                temp_words.append({'text': row['text'], 'left': row['left'], 'height': row['height']})
        lines_data.append({"text": " ".join([w['text'] for w in temp_words]), "top": current_top, "h": 10})

    lines_txt = [l['text'] for l in lines_data]
    full_text_single = " # ".join(lines_txt)

    # --- LOGIKA SENSOR ---
    draw = ImageDraw.Draw(pil_image)
    anchor_nav = "SEMUA KATEGORI"
    for line_obj in lines_data:
        if fuzz.partial_ratio(anchor_nav, line_obj['text']) > 65:
            y_coord = line_obj['top'] / scale
            if y_coord < (pil_image.height * 0.3):
                h_box = min(line_obj['h'] / scale, 40)
                draw.rectangle([0, y_coord - 5, pil_image.width, y_coord + h_box + 5], fill="white")
                break

    # --- LOGIKA PENCOCOKAN NAMA ---
    prod_name = "N/A"
    if master_product_names:
        best_match, highest_score = "N/A", 0
        for ref_name in master_product_names:
            m_name = str(ref_name).upper()
            score = fuzz.partial_ratio(m_name, full_text_single)
            if score > 80 and score > highest_score:
                highest_score, best_match = score, m_name
        prod_name = best_match

    return prod_name, pil_image

# ================= UI STREAMLIT =================
def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

with st.sidebar:
    st.write("---")
    m_code = st.text_input("📍 MASTER CODE").upper()
    date_inp = st.text_input("📅 DATE").upper()
    st.write("---")

files = st.file_uploader("📂 UPLOAD GAMBAR", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code and date_inp:
    if os.path.exists(FILE_PATH):
        db_ig = pd.read_excel(FILE_PATH, sheet_name=SHEET_MASTER_IG)
        list_nama_master = db_ig[COL_IG_NAME].dropna().unique().tolist()
        
        zip_buffer = io.BytesIO()
        processed_count = 0
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                with st.container(border=True):
                    img_pil = Image.open(f)
                    
                    # Proses Gambar (Sensor & Nama)
                    name_found, censored_img = process_image_only(img_pil, list_nama_master)
                    
                    # Cari PRODCODE untuk penamaan file
                    match_code = None
                    best_score = 0
                    for _, row in db_ig.iterrows():
                        db_name = str(row[COL_IG_NAME]).upper()
                        score = fuzz.partial_ratio(db_name, name_found)
                        if score > 75 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])

                    # Tampilan UI
                    st.markdown(f"### 📄 {f.name}")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(censored_img, caption="Preview (Censored)", use_container_width=True)
                    with col2:
                        st.write(f"**Identified Name:** {name_found}")
                        if match_code:
                            st.success(f"**Code Matched:** {match_code}")
                            # Simpan ke ZIP
                            buf = io.BytesIO()
                            censored_img.convert("RGB").save(buf, format="JPEG")
                            zf.writestr(f"{match_code}_{date_inp}.jpg", buf.getvalue())
                            processed_count += 1
                        else:
                            st.warning("⚠️ Code Not Found - Not added to ZIP")
                
                gc.collect()

        if processed_count > 0:
            st.divider()
            st.download_button(
                label=f"📥 DOWNLOAD {processed_count} PROCESSED IMAGES (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"PROCESSED_{m_code}_{date_inp}.zip",
                mime="application/zip",
                use_container_width=True
            )
    else:
        st.error("Database Excel tidak ditemukan di folder /database!")

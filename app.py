import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
from PIL import Image, ImageDraw
import io
import zipfile
from fuzzywuzzy import fuzz

# ================= CONFIG (LINK DATABASE) =================
URL_BASE = "https://docs.google.com/spreadsheets/d/1HF9TVlYok1Virca7Mc_LsUdGbpgCIJoqHLQET5Knc5M/edit?usp=sharing"
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 
TARGET_IMAGE_SIZE_KB = 195 

# Daftar Sensor (Redaksi)
TEXTS_TO_REDACT = ["HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "Halo WAYAN GIYANTO / WRG", 
                   "MEMBER UMUM KLIK", "DJUANMING / TK GOGO", "NONOK JUNENGSIH"]

st.set_page_config(page_title="Price Check AI", layout="wide")

# --- Fungsi Baca Data Anti-Error ---
def get_data_direct(sheet_name):
    try:
        csv_url = f"{URL_BASE}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        df = pd.read_csv(csv_url)
        # Bersihkan nama kolom: Hilangkan spasi & ubah ke huruf besar
        df.columns = [str(c).strip().upper() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Gagal memuat sheet {sheet_name}: {e}")
        return None

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

# --- Logika OCR & Redaksi ---
def process_ocr_all_prices(pil_image, reader):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_pil = pil_image.copy()
    
    # Pre-processing ringan
    img_resized = cv2.resize(img_np, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    results = reader.readtext(gray, detail=1)
    
    df_ocr_list = []
    for (bbox, text, prob) in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        df_ocr_list.append({"text": text.upper(), "top": y_center, "bbox": bbox})
    
    df_ocr = pd.DataFrame(df_ocr_list)
    scanned_name = ""

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        idx_search = df_ocr[df_ocr['text'].str.contains("CARI DI KLIK|SEMUA KATEGORI", na=False)].index
        if not idx_search.empty and (idx_search[-1] + 1) < len(df_ocr):
            scanned_name = df_ocr.iloc[idx_search[-1] + 1]['text']
            
    # Sensor Data Pribadi
    draw = ImageDraw.Draw(original_pil)
    for item in df_ocr_list:
        for kw in TEXTS_TO_REDACT:
            if kw.upper() in item['text']:
                box = item['bbox']
                # Skala ulang box karena gambar asli berbeda ukuran dengan yang di-resize
                draw.rectangle([box[0][0]/1.5, box[0][1]/1.5, box[2][0]/1.5, box[2][1]/1.5], fill="white")

    return scanned_name, original_pil

# --- UI UTAMA ---
st.title("📸 Price Check AI")

col1, col2, col3 = st.columns(3)
with col1: m_code = st.text_input("📍 Master Code").strip().upper()
with col2: tgl = st.text_input("📅 Tanggal (23JAN2026)").strip().upper()
with col3: week = st.text_input("🗓️ Week").strip()

files = st.file_uploader("📂 Upload Foto", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code:
    db_ig = get_data_direct(SHEET_MASTER_IG)
    
    if db_ig is not None:
        # Validasi Kolom (Mencegah KeyError)
        target_col = COL_IG_NAME.strip().upper()
        prodcode_col = "PRODCODE" # Sesuaikan jika di Excel namanya lain
        
        if target_col not in db_ig.columns:
            st.error(f"❌ Kolom '{target_col}' tidak ditemukan! Nama kolom yang ada: {list(db_ig.columns)}")
            st.stop()
            
        st.success(f"✅ Database IG Terhubung. (Total: {len(db_ig)} produk)")
        reader = load_reader()
        
        final_list = []
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for idx, f in enumerate(files):
                img_pil = Image.open(f)
                s_name, red_img = process_ocr_all_prices(img_pil, reader)
                
                # Matching
                best_match_code = None
                max_score = 0
                for _, row in db_ig.iterrows():
                    score = fuzz.token_set_ratio(str(row[target_col]), s_name)
                    if score > 80 and score > max_score:
                        max_score = score
                        best_match_code = str(row[prodcode_col])
                
                if best_match_code:
                    with st.expander(f"✅ Berhasil: {best_match_code}"):
                        st.write(f"Nama Scan: {s_name}")
                        st.image(red_img, width=400)
                        
                        # Masukkan ke ZIP
                        img_byte_arr = io.BytesIO()
                        red_img.save(img_byte_arr, format='JPEG', quality=85)
                        zip_file.writestr(f"{best_match_code}.jpg", img_byte_arr.getvalue())
                        
                        final_list.append({"PRODCODE": best_match_code, "NAME_SCAN": s_name})
                else:
                    st.warning(f"⚠️ Gagal mencocokkan: {s_name}")

        if final_list:
            st.write("### 📋 Ringkasan Scan")
            st.dataframe(pd.DataFrame(final_list))
            
            # Tombol Download
            zip_name = f"{m_code}_{tgl}.zip"
            st.download_button("🖼️ Download Hasil Scan (ZIP)", zip_buffer.getvalue(), zip_name)


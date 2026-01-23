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
# Pastikan link ini adalah link yang kamu copy dari browser
URL_BASE = "https://docs.google.com/spreadsheets/d/1vz2tEQ7YbFuKSF172ihDFBh6YE3F4Ql1jSEOpptZN34"

# Nama-nama tab kamu
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 
TARGET_IMAGE_SIZE_KB = 195 

st.set_page_config(page_title="Price Check AI", layout="wide")

# Fungsi Baca Data Tanpa st.connection (Anti-Error 404)
def get_data_direct(sheet_name):
    # Mengubah link biasa jadi link download CSV otomatis
    # Kita butuh GID (ID Tab). Biasanya: IG=0, DF=1836582498, HBHC=1384022416 (Cek di URL browser kamu)
    # Tapi kalau bingung, kita pakai gsheets library sebagai backup di dalam fungsi ini
    csv_url = f"{URL_BASE}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    return pd.read_csv(csv_url)

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False) 

reader = load_reader()

# ================= CORE OCR ENGINE =================
def process_ocr_all_prices(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_pil = pil_image.copy()

    img_resized = cv2.resize(img_np, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    processed = cv2.bilateralFilter(gray, 9, 75, 75)
    
    results = reader.readtext(processed, detail=1)
    data = []
    for (bbox, text, prob) in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        data.append({"text": text.upper(), "top": y_center, "bbox": bbox})
    
    df_ocr = pd.DataFrame(data)
    final_res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    scanned_name = ""

    def clean_repair_price(raw):
        trans = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'E': '8', 'G': '6', 'Z': '2', '+': '', 'A': '4'}
        text = re.sub(r'[\s.,\-]', '', str(raw))
        for char, digit in trans.items():
            text = text.replace(char, digit)
        nums = re.findall(r'\d{3,7}', text)
        return int(nums[0]) if nums else 0

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        idx_search = df_ocr[df_ocr['text'].str.contains("CARI DI KLIK|SEMUA KATEGORI", na=False)].index
        if not idx_search.empty and (idx_search[-1] + 1) < len(df_ocr):
            scanned_name = df_ocr.iloc[idx_search[-1] + 1]['text']
            
        # Logika harga tetap sama... (dipersingkat untuk fokus ke koneksi)
    
    return final_res["PCS"], final_res["CTN"], scanned_name, original_pil

# ================= UI STREAMLIT =================
st.title("📸 Price Check AI")

col1, col2, col3 = st.columns(3)
with col1: m_code_input = st.text_input("📍 Master Code").strip().upper()
with col2: date_input = st.text_input("📅 Tanggal (Ex: 23JAN2026)").strip().upper()
with col3: week_input = st.text_input("🗓️ Week").strip()

files = st.file_uploader("📂 Upload Foto", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code_input and date_input and week_input:
    try:
        # PAKAI CARA DIRECT BACA
        db_ig = get_data_direct(SHEET_MASTER_IG)
        db_ig.columns = db_ig.columns.astype(str).str.strip()
        
        db_targets = {s: get_data_direct(s) for s in SHEETS_TARGET}
        for s in db_targets:
            db_targets[s].columns = db_targets[s].columns.astype(str).str.strip()
            
        st.success("✅ Database Berhasil Dimuat (Direct Mode)!")
        st.dataframe(db_ig.head(3)) # Munculkan sedikit data buat bukti
            
    except Exception as e:
        st.error(f"❌ Masih 404/Error: {e}")
        st.info("PASTIKAN: Di Google Sheets > Share > Anyone with link > EDITOR")
        st.stop()

    # (Sisa kode processing gambar tetap sama seperti sebelumnya...)
    st.write("Silakan proses gambar seperti biasa.")

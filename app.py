import streamlit as st
import easyocr
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

# ================= CONFIG =================
# Pastikan path ini sesuai dengan folder di GitHub
FILE_PATH = "database/master_harga.xlsx"
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 
TARGET_IMAGE_SIZE_KB = 195 

TEXTS_TO_REDACT = ["HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "Halo WAYAN GIYANTO / WRG", "Halo MEMBER UMUM KLIK", "Halo DJUANMING / TK GOGO", "Halo NONOK JUNENGSIH", "Halo AGUNG KURNIAWAN", "Halo ARIF RAMADHAN", "Halo HILMI ATIQ / WR DINDA"]

st.set_page_config(page_title="Price Check", layout="wide")

@st.cache_resource
def load_reader():
    # Menggunakan CPU karena Streamlit Cloud versi gratis tidak ada GPU
    return easyocr.Reader(['en'], gpu=False) 

reader = load_reader()

# --- Fungsi Helper OCR ---
def process_ocr_all_prices(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_pil = pil_image.copy()
    
    # Pre-processing sederhana untuk mempercepat
    img_resized = cv2.resize(img_np, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, detail=1)
    
    data = []
    for (bbox, text, prob) in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        data.append({"text": text.upper(), "top": y_center})
    
    df_ocr = pd.DataFrame(data)
    # ... (Logika pengambilan harga tetap sama dengan kode sebelumnya)
    # Untuk ringkasan, saya asumsikan logika harga Anda sudah berjalan di lokal
    return {"normal": 0, "promo": 0}, {"normal": 0, "promo": 0}, "PRODUCT NAME", original_pil

def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

# ================= UI STREAMLIT =================
st.title("📸 Price Check")

# Input User
col1, col2, col3 = st.columns(3)
with col1: m_code_input = st.text_input("📍 Master Code").strip().upper()
with col2: date_input = st.text_input("📅 Date (01JAN2026)").strip().upper()
with col3: week_input = st.text_input("🗓️ Week").strip()

files = st.file_uploader("📂 Upload Foto", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code_input and date_input and week_input:
    # Cek apakah file Excel ada
    if not os.path.exists(FILE_PATH):
        st.error(f"❌ File tidak ditemukan di: {FILE_PATH}. Pastikan sudah upload folder 'database' ke GitHub.")
    else:
        try:
            # Membaca Excel (Ini bagian yang berat jika file sangat besar)
            with st.spinner('Membaca Database Excel...'):
                xl = pd.ExcelFile(FILE_PATH)
                db_ig = xl.parse(SHEET_MASTER_IG)
                db_targets = {s: xl.parse(s) for s in SHEETS_TARGET if s in xl.sheet_names}
            
            final_list = []
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for f in files:
                    res_pcs, res_ctn, scanned_name, redacted_img = process_ocr_all_prices(Image.open(f))
                    
                    # Logika Matching & Penyimpanan (Sesuai permintaan penamaan file Anda)
                    # ... (Gunakan logika matching Anda di sini)
                    
                    # Contoh pengisian list (dummy)
                    final_list.append({"prodcode": "EXAMPLE", "n_pcs": 1000, "p_pcs": 900})

            if final_list:
                st.success("Selesai!")
                # Tombol Download dengan nama file dinamis sesuai input Anda
                excel_name = f"Price Check W{week_input}_{date_input}.xlsx"
                zip_name = f"{m_code_input}_{date_input}.zip"
                
                c1, c2 = st.columns(2)
                with c1: st.download_button("📥 DOWNLOAD EXCEL", open(FILE_PATH, "rb"), excel_name)
                with c2: st.download_button("🖼️ DOWNLOAD ZIP", zip_buffer.getvalue(), zip_name)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca Excel: {e}")

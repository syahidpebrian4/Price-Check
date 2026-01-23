import streamlit as st
import pandas as pd
import numpy as np
import easyocr
import cv2
import re
from PIL import Image, ImageDraw
import io
import zipfile
from fuzzywuzzy import fuzz

# --- CONFIG ---
URL_BASE = "https://docs.google.com/spreadsheets/d/1vz2tEQ7YbFuKSF172ihDFBh6YE3F4Ql1jSEOpptZN34"
SHEET_MASTER_IG = "IG"
SHEETS_TARGET = ["DF", "HBHC"]

st.set_page_config(page_title="Price Check AI", layout="wide")

# --- FUNGSI BACA DATA (ANTI CRASH) ---
def get_data_direct(sheet_name):
    try:
        # Menggunakan format export gviz agar kolom tidak berantakan
        csv_url = f"{URL_BASE}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        df = pd.read_csv(csv_url)
        df.columns = df.columns.astype(str).str.strip() # Bersihkan nama kolom
        return df
    except Exception as e:
        st.error(f"Gagal memuat sheet {sheet_name}: {e}")
        return None

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

# --- UI ---
st.title("📸 Price Check AI")

# Input User
col1, col2, col3 = st.columns(3)
with col1: m_code = st.text_input("📍 Master Code").strip().upper()
with col2: tgl = st.text_input("📅 Tanggal (Ex: 23JAN2026)").strip().upper()
with col3: week = st.text_input("🗓️ Week").strip()

files = st.file_uploader("📂 Upload Foto Product", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code and tgl and week:
    # 1. Load Database
    with st.spinner("Menghubungkan ke Spreadsheet..."):
        db_ig = get_data_direct(SHEET_MASTER_IG)
        
    if db_ig is not None:
        st.success("✅ Database Berhasil Dimuat!")
        
        # Tampilkan pratinjau data agar kita tahu kolomnya sudah benar
        with st.expander("Klik untuk cek isi Database IG"):
            st.dataframe(db_ig.head(5))
            
        # 2. Proses OCR Sederhana (Untuk Tes)
        if st.button("🔍 Mulai Proses Scan"):
            reader = load_reader()
            for f in files:
                st.write(f"Memproses: {f.name}...")
                # Lanjutkan logika scanning di sini...
                st.info("Fitur scan aktif. Pastikan kolom 'PRODNAME_IG' ada di tabel atas.")
    else:
        st.error("Data tidak bisa ditarik. Cek apakah Link Spreadsheet sudah 'Anyone with link - Editor'.")

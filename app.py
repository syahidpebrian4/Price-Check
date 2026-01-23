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

# ================= CONFIG =================
URL_BASE = "https://docs.google.com/spreadsheets/d/1vz2tEQ7YbFuKSF172ihDFBh6YE3F4Ql1jSEOpptZN34"
SHEET_MASTER_IG = "IG"
TEXTS_TO_REDACT = ["HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "WAYAN GIYANTO", 
                   "MEMBER UMUM KLIK", "DJUANMING", "NONOK JUNENGSIH"]

st.set_page_config(page_title="Price Check AI", layout="wide")

# --- Fungsi Baca Data dengan Proteksi Tipe Data ---
def get_data_fixed(sheet_name):
    try:
        csv_url = f"{URL_BASE}/export?format=csv&sheet={sheet_name}"
        
        # Gunakan usecols=[0,1] untuk membatasi hanya kolom A & B (mencegah tokenizing error)
        df = pd.read_csv(csv_url, usecols=[0, 1], on_bad_lines='skip', engine='python')
        
        # Berikan nama kolom yang konsisten
        df.columns = ["PRODCODE", "PRODNAME_IG"]
        
        # Buang baris yang benar-benar kosong (Mengatasi Baris 2 kosong di GSheets)
        df = df.dropna(subset=["PRODCODE", "PRODNAME_IG"]).reset_index(drop=True)
        
        # PERBAIKAN: Memastikan semua data adalah string sebelum menggunakan fungsi teks
        df["PRODCODE"] = df["PRODCODE"].astype(str).str.strip()
        
        # Gunakan apply lambda untuk menghindari error 'Series' object has no attribute 'upper'
        df["PRODNAME_IG"] = df["PRODNAME_IG"].astype(str).apply(lambda x: x.strip().upper())
        
        return df
    except Exception as e:
        st.error(f"⚠️ Gagal Memproses Sheet {sheet_name}: {e}")
        return None

@st.cache_resource
def load_reader():
    # Load model OCR ke RAM (sekali saja)
    return easyocr.Reader(['en'], gpu=False)

# --- Logika OCR & Sensor (Redaksi) ---
def process_ocr_all_prices(pil_image, reader):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_pil = pil_image.copy()
    
    # Pre-processing: Resize 1.5x untuk keseimbangan akurasi & kecepatan
    img_resized = cv2.resize(img_np, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    results = reader.readtext(gray, detail=1)
    
    data_ocr = []
    for (bbox, text, prob) in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        data_ocr.append({"text": text.upper(), "top": y_center, "bbox": bbox})
    
    df_ocr = pd.DataFrame(data_ocr)
    scanned_name = ""

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        # Mencari nama produk tepat di bawah header 'Cari di Klik'
        idx_ref = df_ocr[df_ocr['text'].str.contains("CARI DI KLIK|SEMUA KATEGORI", na=False)].index
        if not idx_ref.empty and (idx_ref[-1] + 1) < len(df_ocr):
            scanned_name = df_ocr.iloc[idx_ref[-1] + 1]['text']
            
    # Auto-Redact (Sensor Putih)
    draw = ImageDraw.Draw(original_pil)
    for item in data_ocr:
        for kw in TEXTS_TO_REDACT:
            if fuzz.partial_ratio(kw.upper(), item['text']) > 80:
                box = item['bbox']
                # Skala box kembali ke ukuran asli gambar
                draw.rectangle([box[0][0]/1.5 - 5, box[0][1]/1.5 - 5, 
                                box[2][0]/1.5 + 5, box[2][1]/1.5 + 5], fill="white")
    
    return scanned_name, original_pil

# --- UI UTAMA ---
st.title("📸 Price Check AI (Stabil)")

col_input = st.columns(3)
with col_input[0]: m_code_in = st.text_input("📍 Master Code").strip().upper()
with col_input[1]: tgl_in = st.text_input("📅 Tanggal (Ex: 23JAN2026)").strip().upper()
with col_input[2]: week_in = st.text_input("🗓️ Week").strip()

files = st.file_uploader("📂 Upload Foto", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code_in:
    db_ig = get_data_fixed(SHEET_MASTER_IG)
    
    if db_ig is not None:
        st.success(f"✅ Database IG Terhubung. (Total: {len(db_ig)} Produk)")
        
        # Tabel preview untuk memastikan data tidak hancur lagi
        with st.expander("Lihat Preview Database"):
            st.table(db_ig.head(10))
            
        reader = load_reader()
        final_results = []
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for f in files:
                img_pil = Image.open(f)
                s_name, red_img = process_ocr_all_prices(img_pil, reader)
                
                # Mencocokkan hasil scan dengan database menggunakan Fuzzy Matching
                best_code, max_score = None, 0
                for _, row in db_ig.iterrows():
                    score = fuzz.token_set_ratio(row["PRODNAME_IG"], s_name)
                    if score > 80 and score > max_score:
                        max_score, best_code = score, row["PRODCODE"]
                
                if best_code:
                    with st.expander(f"✅ Berhasil: {best_code}"):
                        st.write(f"Hasil Scan: **{s_name}**")
                        st.image(red_img, width=300)
                        
                        # Simpan ke memori untuk ZIP
                        img_io = io.BytesIO()
                        red_img.save(img_io, format='JPEG', quality=85)
                        zip_file.writestr(f"{best_code}.jpg", img_io.getvalue())
                        
                        final_results.append({"PRODCODE": best_code, "MATCH_SCORE": max_score})
                else:
                    st.warning(f"⚠️ Produk Tidak Dikenali: {s_name}")

        if final_results:
            st.write("### 📋 Ringkasan Hasil")
            st.dataframe(pd.DataFrame(final_results))
            
            # Tombol Download ZIP
            zip_filename = f"{m_code_in}_{tgl_in}.zip"
            st.download_button("📥 Download Hasil Scan (ZIP)", zip_buffer.getvalue(), zip_filename)

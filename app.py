import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
import os
from PIL import Image, ImageDraw
import io
import zipfile
from fuzzywuzzy import fuzz

# ================= CONFIG =================
FILE_PATH = "database/master_harga.xlsx"
st.set_page_config(page_title="Price Check Lite", layout="wide")

# Gunakan cache hanya untuk Reader agar tidak boros RAM
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False, download_enabled=True)

reader = load_reader()

def process_lite(pil_image):
    # Kecilkan skala resize agar RAM tidak meledak
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_small = cv2.resize(img_np, None, fx=1.2, fy=1.2) # Skala diperkecil
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    
    # Langsung OCR tanpa filter berat
    results = reader.readtext(gray)
    
    # Logika sederhana ambil teks
    all_text = " ".join([res[1].upper() for res in results])
    
    # Sensor Nama (Redaksi)
    draw = ImageDraw.Draw(pil_image)
    for (bbox, text, prob) in results:
        if any(x in text.upper() for x in ["YUYUN", "SUMARNI", "HALO", "MEMBER"]):
            # Skala balik koordinat karena tadi diresize
            box = np.array(bbox) / 1.2
            draw.rectangle([tuple(box[0]), tuple(box[2])], fill="white")
            
    return all_text, pil_image

# ================= UI =================
st.title("📸 Price Check AI (Lite Mode)")

m_code = st.sidebar.text_input("Master Code")
tgl = st.sidebar.text_input("Tanggal")
week = st.sidebar.text_input("Week")

uploaded_files = st.file_uploader("Upload Foto", accept_multiple_files=True)

if uploaded_files and m_code:
    if os.path.exists(FILE_PATH):
        try:
            # Pindahkan pembacaan Excel ke luar loop agar tidak berulang
            df_ig = pd.read_excel(FILE_PATH, sheet_name="IG")
            
            final_results = []
            for f in uploaded_files:
                text_detected, img_redacted = process_lite(Image.open(f))
                
                # Matching sederhana
                final_results.append({
                    "File": f.name,
                    "Detected": text_detected[:50] + "...",
                    "Status": "Processed"
                })
                st.image(img_redacted, width=400)
            
            st.table(pd.DataFrame(final_results))
            
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("File database/master_harga.xlsx tidak ditemukan di GitHub.")

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
URL_BASE = "https://docs.google.com/spreadsheets/d/1vz2tEQ7YbFuKSF172ihDFBh6YE3F4Ql1jSEOpptZN34"
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 
TARGET_IMAGE_SIZE_KB = 195 

TEXTS_TO_REDACT = ["HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "Halo WAYAN GIYANTO / WRG", 
                   "Halo MEMBER UMUM KLIK", "Halo DJUANMING / TK GOGO", "Halo NONOK JUNENGSIH"]

st.set_page_config(page_title="Price Check AI", layout="wide")

# Fungsi Baca Data Direct
def get_data_direct(sheet_name):
    try:
        csv_url = f"{URL_BASE}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        df = pd.read_csv(csv_url)
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Gagal memuat sheet {sheet_name}: {e}")
        return None

@st.cache_resource
def load_reader():
    # Load model sekali saja untuk hemat RAM
    return easyocr.Reader(['en'], gpu=False) 

# ================= CORE OCR ENGINE =================
def process_ocr_all_prices(pil_image, reader):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_pil = pil_image.copy()
    
    # Pre-processing ringan agar tidak crash
    img_resized = cv2.resize(img_np, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    results = reader.readtext(gray, detail=1)
    data = []
    for (bbox, text, prob) in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        data.append({"text": text.upper(), "top": y_center, "bbox": bbox})
    
    df_ocr = pd.DataFrame(data)
    final_res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    scanned_name = ""

    def clean_repair_price(raw):
        trans = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'E': '8', 'G': '6', 'Z': '2'}
        text = re.sub(r'[\s.,\-]', '', str(raw))
        for char, digit in trans.items(): text = text.replace(char, digit)
        nums = re.findall(r'\d{3,7}', text)
        return int(nums[0]) if nums else 0

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        idx_search = df_ocr[df_ocr['text'].str.contains("CARI DI KLIK|SEMUA KATEGORI", na=False)].index
        if not idx_search.empty and (idx_search[-1] + 1) < len(df_ocr):
            scanned_name = df_ocr.iloc[idx_search[-1] + 1]['text']
            
    # Sensor Redaksi Sederhana
    draw = ImageDraw.Draw(original_pil)
    for item in data:
        for kw in TEXTS_TO_REDACT:
            if kw.upper() in item['text']:
                bbox = item['bbox']
                draw.rectangle([bbox[0][0]/1.5, bbox[0][1]/1.5, bbox[2][0]/1.5, bbox[2][1]/1.5], fill="white")

    return final_res["PCS"], final_res["CTN"], scanned_name, original_pil

# ================= UI =================
st.title("📸 Price Check AI (Stable Mode)")

col1, col2, col3 = st.columns(3)
with col1: m_code = st.text_input("📍 Master Code").strip().upper()
with col2: tgl = st.text_input("📅 Tanggal").strip().upper()
with col3: week = st.text_input("🗓️ Week").strip()

files = st.file_uploader("📂 Upload Foto", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code:
    db_ig = get_data_direct(SHEET_MASTER_IG)
    
    if db_ig is not None:
        st.success("✅ Database Terhubung!")
        reader = load_reader()
        
        final_list = []
        for f in files:
            img_pil = Image.open(f)
            res_pcs, res_ctn, s_name, red_img = process_ocr_all_prices(img_pil, reader)
            
            # Fuzzy Matching
            best_match = None
            max_score = 0
            for _, row in db_ig.iterrows():
                score = fuzz.token_set_ratio(str(row[COL_IG_NAME]), s_name)
                if score > 80 and score > max_score:
                    max_score = score
                    best_match = str(row["PRODCODE"])
            
            if best_match:
                st.write(f"✔️ Cocok: **{best_match}** ({s_name})")
                final_list.append({"PRODCODE": best_match, "NAME": s_name})
            else:
                st.warning(f"❓ Tidak dikenal: {s_name}")
            
            st.image(red_img, width=300)

        if final_list:
            st.info("Logika update GSheets siap. Gunakan 'Deploy as Web App' di Apps Script untuk simpan permanen.")

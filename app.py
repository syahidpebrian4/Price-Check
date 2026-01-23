import streamlit as st
import pytesseract
import cv2
import numpy as np
import pandas as pd
import re
from PIL import Image, ImageDraw
import io
import zipfile
from fuzzywuzzy import fuzz
import gc

# ================= CONFIG =================
URL_BASE = "https://docs.google.com/spreadsheets/d/1vz2tEQ7YbFuKSF172ihDFBh6YE3F4Ql1jSEOpptZN34"
SHEET_MASTER_IG = "IG"

TEXTS_TO_REDACT = [
    "HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "WAYAN GIYANTO", 
    "MEMBER UMUM KLIK", "DJUANMING", "NONOK JUNENGSIH", 
    "AGUNG KURNIAWAN", "ARIF RAMADHAN", "HILMI ATIQ"
]

st.set_page_config(page_title="Price Check Tesseract V1.0", layout="wide")

# --- Logika Pembersihan Harga ---
def clean_price(raw):
    trans = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'E': '8', 'G': '6', 'Z': '2', 'A': '4'}
    txt = re.sub(r'[\s.,\-]', '', str(raw))
    for c, d in trans.items(): 
        txt = txt.replace(c, d)
    nums = re.findall(r'\d{3,7}', txt)
    return int(nums[0]) if nums else 0

def extract_prices_smart(text_content):
    text_content = text_content.upper()
    text_content = re.sub(r'\(.*?\)|ISI\s*\d+', '', text_content)
    found_prices = []
    parts = text_content.split('RP')
    for part in parts[1:]:
        p = clean_price(part.split('/')[0])
        if p > 500: 
            found_prices.append(p)
    
    if not found_prices: return {"normal": 0, "promo": 0}
    if len(found_prices) >= 2:
        return {"normal": max(found_prices), "promo": min(found_prices)}
    return {"normal": found_prices[0], "promo": found_prices[0]}

# --- Fungsi OCR Utama ---
def process_with_tesseract(pil_img):
    # Pre-processing ringan
    img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    # Ambil data dari Tesseract
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    
    full_text_list = []
    for i in range(len(d['text'])):
        txt = d['text'][i].strip().upper()
        if txt:
            full_text_list.append({
                "text": txt,
                "top": d['top'][i],
                "left": d['left'][i],
                "width": d['width'][i],
                "height": d['height'][i]
            })

    df_ocr = pd.DataFrame(full_text_list)
    res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    p_name = ""

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        full_string = " ".join(df_ocr['text'].tolist())
        
        # Cari Nama Produk
        idx_search = df_ocr[df_ocr['text'].str.contains("CARI|KLIK|SEMUA", na=False)].index
        if not idx_search.empty:
            start_idx = idx_search[-1] + 1
            p_name = " ".join(df_ocr.iloc[start_idx : start_idx+5]['text'].tolist())

        # Cari Harga
        if "SATUAN" in full_string or "PCS" in full_string:
            res["PCS"] = extract_prices_smart(full_string)
        if "CTN" in full_string or "KARTON" in full_string or "DUS" in full_string:
            res["CTN"] = extract_prices_smart(full_string)

    # Redaksi (Sensor Putih)
    draw = ImageDraw.Draw(pil_img)
    for i in range(len(d['text'])):
        word = d['text'][i].upper()
        for kw in TEXTS_TO_REDACT:
            if word and word in kw.upper():
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                draw.rectangle([x-2, y-2, x+w+2, y+h+2], fill="white")

    return res, p_name, pil_img

# --- UI STREAMLIT ---
st.title("📸 Price Check (Tesseract Mode)")

with st.sidebar:
    m_code = st.text_input("Master Code").upper()
    tgl = st.text_input("Tanggal").upper()

files = st.file_uploader("Upload Foto", type=["jpg","jpeg","png"], accept_multiple_files=True)

if files and m_code:
    # Get Database from Google Sheets
    try:
        csv_url = f"{URL_BASE}/export?format=csv&sheet={SHEET_MASTER_IG}"
        db = pd.read_csv(csv_url, usecols=[0, 1], on_bad_lines='skip', engine='python')
        db.columns = ["PRODCODE", "PRODNAME_IG"]
        db = db.dropna().reset_index(drop=True)
        db["PRODNAME_IG"] = db["PRODNAME_IG"].astype(str).str.upper()
        
        zip_buffer = io.BytesIO()
        final_table = []

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                with st.status(f"Memproses {f.name}...") as s:
                    img_obj = Image.open(f)
                    res_prices, scanned_name, red_img = process_with_tesseract(img_obj)
                    
                    # Fuzzy Matching
                    match_code, best_score = None, 0
                    for _, row in db.iterrows():
                        score = fuzz.token_set_ratio(row["PRODNAME_IG"], scanned_name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, row["PRODCODE"]
                    
                    if match_code:
                        img_io = io.BytesIO()
                        red_img.save(img_io, format="JPEG", quality=80)
                        zf.writestr(f"{match_code}.jpg", img_io.getvalue())
                        
                        final_table.append({
                            "PRODCODE": match_code,
                            "N_PCS": res_prices["PCS"]["normal"],
                            "P_PCS": res_prices["PCS"]["promo"]
                        })
                        s.update(label=f"✅ {match_code}", state="complete")
                    else:
                        st.warning(f"Nama produk tak terdeteksi di {f.name}")
                    
                    gc.collect()

        if final_table:
            st.dataframe(pd.DataFrame(final_table))
            st.download_button("📥 Download ZIP", zip_buffer.getvalue(), f"{m_code}_{tgl}.zip")
            
    except Exception as e:
        st.error(f"Error: {e}")

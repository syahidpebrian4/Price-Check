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

st.set_page_config(page_title="Price Check AI V10.5 Full Output", layout="wide")

def clean_price(raw):
    trans = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'E': '8', 'G': '6', 'Z': '2', 'A': '4'}
    txt = re.sub(r'[\s.,\-]', '', str(raw))
    for c, d in trans.items(): txt = txt.replace(c, d)
    nums = re.findall(r'\d{3,7}', txt)
    return int(nums[0]) if nums else 0

def extract_prices_smart(text_content):
    text_content = text_content.upper()
    text_content = re.sub(r'\(.*?\)|ISI\s*\d+', '', text_content)
    found_prices = []
    parts = text_content.split('RP')
    for part in parts[1:]:
        segment = part.split('/')[0].split('\n')[0]
        p = clean_price(segment)
        if p > 500: found_prices.append(p)
    
    if not found_prices: return {"normal": 0, "promo": 0}
    if len(found_prices) >= 2:
        return {"normal": max(found_prices), "promo": min(found_prices)}
    return {"normal": found_prices[0], "promo": found_prices[0]}

def process_with_tesseract(pil_img):
    img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_np = cv2.resize(img_np, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    
    full_text_list = []
    for i in range(len(d['text'])):
        txt = d['text'][i].strip().upper()
        if txt:
            full_text_list.append({"text": txt, "top": d['top'][i], "left": d['left'][i], "width": d['width'][i], "height": d['height'][i]})

    df_ocr = pd.DataFrame(full_text_list)
    res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    p_name = "TIDAK TERDETEKSI"

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        all_words = df_ocr['text'].tolist()
        full_string = " ".join(all_words)
        
        idx_found = -1
        for i, word in enumerate(all_words):
            if any(key in word for key in ["CARI", "KLIK", "SEMUA"]):
                idx_found = i
                break
        
        if idx_found != -1 and (idx_found + 1) < len(all_words):
            p_name = " ".join(all_words[idx_found + 1 : idx_found + 7])

        if any(x in full_string for x in ["PCS", "SATUAN", "PCK", "RCG"]):
            res["PCS"] = extract_prices_smart(full_string)
        if any(x in full_string for x in ["CTN", "KARTON", "DUS"]):
            res["CTN"] = extract_prices_smart(full_string)

    draw = ImageDraw.Draw(pil_img)
    for i in range(len(d['text'])):
        word = d['text'][i].upper()
        for kw in TEXTS_TO_REDACT:
            if word and word in kw.upper() and len(word) > 3:
                x, y, w, h = d['left'][i]/1.5, d['top'][i]/1.5, d['width'][i]/1.5, d['height'][i]/1.5
                draw.rectangle([x-2, y-2, x+w+2, y+h+2], fill="white")

    return res, p_name, pil_img

# --- UI ---
st.title("📸 Price Check AI - Full Result View")

with st.sidebar:
    st.header("📋 Data Input")
    m_code = st.text_input("Master Code").upper()
    tgl = st.text_input("Tanggal").upper()

files = st.file_uploader("📂 Upload Foto", type=["jpg","jpeg","png"], accept_multiple_files=True)

if files and m_code:
    try:
        csv_url = f"{URL_BASE}/export?format=csv&sheet={SHEET_MASTER_IG}"
        db = pd.read_csv(csv_url, usecols=[0, 1], on_bad_lines='skip', engine='python')
        db.columns = ["PRODCODE", "PRODNAME_IG"]
        db = db.dropna().reset_index(drop=True)
        db["PRODNAME_IG"] = db["PRODNAME_IG"].astype(str).str.upper()
        
        zip_buffer = io.BytesIO()
        final_table = []

        # Area Tampilan Processing
        st.write("### ⚙️ Processing Details")
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                img_obj = Image.open(f)
                res_prices, scanned_name, red_img = process_with_tesseract(img_obj)
                
                match_code, best_score = "N/A", 0
                for _, row in db.iterrows():
                    score = fuzz.partial_ratio(row["PRODNAME_IG"], scanned_name)
                    if score > 65 and score > best_score:
                        best_score, match_code = score, row["PRODCODE"]
                
                # Menampilkan BOX untuk setiap gambar
                with st.expander(f"🖼️ File: {f.name} | Match: {match_code}", expanded=True):
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.image(red_img, use_container_width=True)
                    with col_b:
                        st.write(f"**Nama Terbaca OCR:** `{scanned_name}`")
                        st.write(f"**Confidence Score:** `{best_score}%`")
                        
                        st.markdown("---")
                        c_pcs, c_ctn = st.columns(2)
                        c_pcs.metric("PCS (Normal)", f"Rp{res_prices['PCS']['normal']:,}")
                        c_pcs.metric("PCS (Promo)", f"Rp{res_prices['PCS']['promo']:,}")
                        c_ctn.metric("CTN (Normal)", f"Rp{res_prices['CTN']['normal']:,}")
                        c_ctn.metric("CTN (Promo)", f"Rp{res_prices['CTN']['promo']:,}")

                if match_code != "N/A":
                    img_io = io.BytesIO()
                    red_img.save(img_io, format="JPEG", quality=80)
                    zf.writestr(f"{match_code}.jpg", img_io.getvalue())
                    
                    final_table.append({
                        "FILE": f.name,
                        "PRODCODE": match_code,
                        "SCAN_NAME": scanned_name,
                        "N_PCS": res_prices["PCS"]["normal"],
                        "P_PCS": res_prices["PCS"]["promo"],
                        "N_CTN": res_prices["CTN"]["normal"],
                        "P_CTN": res_prices["CTN"]["promo"]
                    })
                gc.collect()

        if final_table:
            st.write("### 📋 Tabel Ringkasan Akhir")
            st.dataframe(pd.DataFrame(final_table), use_container_width=True)
            
            st.download_button(
                label="📥 DOWNLOAD HASIL ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"{m_code}_{tgl}.zip",
                mime="application/zip",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"Error Sistem: {e}")

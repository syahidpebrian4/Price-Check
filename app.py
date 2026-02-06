import streamlit as st
import pytesseract
import cv2
import numpy as np
import pandas as pd
import re
import os
from PIL import Image
from openpyxl import load_workbook
import io
import zipfile
from fuzzywuzzy import fuzz
import gc
import base64

# ================= CONFIG & DATABASE =================
FILE_PATH = "database/master_harga.xlsx"
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 

st.set_page_config(page_title="Price Check Pro", layout="wide", initial_sidebar_state="expanded")

# --- CSS CUSTOM HEADER ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #FF0000 !important; }
        [data-testid="stSidebar"] * { color: white !important; font-weight: bold; }
        .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 10px; border: 1px solid #ddd; }
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI MEMBERSIHKAN HARGA ---
def clean_to_int(raw_str):
    if not raw_str: return 0
    s = str(raw_str).upper()
    # Koreksi karakter typo OCR yang sering muncul di angka
    s = s.replace('O', '0').replace('S', '5').replace('I', '1').replace('B', '8').replace('L', '1').replace('G', '6')
    digits = re.sub(r'[^\d]', '', s)
    return int(digits) if digits else 0

# --- LOGIKA BARU: CARI DI ANTARA KATA KUNCI ---
def get_prices_between_keywords(text, is_ctn=False):
    """
    Mencari angka yang berada SETELAH PCS/RCG/BOX 
    dan SEBELUM / atau ISI
    """
    prices = []
    
    # 1. Hapus isi kurung dulu agar tidak mengganggu
    text = re.sub(r'\(.*?\)', ' ', text)
    
    # 2. Tentukan Keyword (Aliasing untuk salah baca OCR)
    if is_ctn:
        keywords = ["CTN", "CIN", "CTH"]
    else:
        keywords = ["PCS", "PES", "PC5", "RCG", "RC6", "PCK", "BOX", "B0X"]

    for kw in keywords:
        # Regex: Cari keyword -> ambil teks sampai ketemu '/' atau 'ISI' atau 'RP' berikutnya
        # Pattern: KW + (apapun di antaranya) + (/ atau ISI)
        pattern = rf"{kw}(.*?)(?=/|ISI|RP|$)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for m in matches:
            # Cari angka di dalam potongan teks tersebut
            found_nums = re.findall(r'[\d\.,OSIBLG]{4,12}', m)
            for n in found_nums:
                val = clean_to_int(n)
                if 400 <= val <= 3000000:
                    if val not in prices: prices.append(val)
    
    return prices

def process_ocr_targeted(pil_img, master_names):
    # Pre-processing
    img = np.array(pil_img.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # OCR - Gunakan PSM 6 karena price tag biasanya berpola kolom/baris
    raw_data = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
    full_text = raw_data.upper().replace('\n', '  ')

    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}
    
    # --- PROSES PCS ---
    pcs_vals = get_prices_between_keywords(full_text, is_ctn=False)
    if len(pcs_vals) >= 2:
        res["PCS"]["n"], res["PCS"]["p"] = pcs_vals[0], pcs_vals[1]
    elif len(pcs_vals) == 1:
        res["PCS"]["n"] = res["PCS"]["p"] = pcs_vals[0]

    # --- PROSES CTN ---
    ctn_vals = get_prices_between_keywords(full_text, is_ctn=True)
    if len(ctn_vals) >= 2:
        res["CTN"]["n"], res["CTN"]["p"] = ctn_vals[0], ctn_vals[1]
    elif len(ctn_vals) == 1:
        res["CTN"]["n"] = res["CTN"]["p"] = ctn_vals[0]

    # Fuzzy Matching Nama
    best_name = "N/A"
    if master_names:
        match_res = [(fuzz.partial_ratio(str(m).upper(), full_text), m) for m in master_names]
        top_match = max(match_res, key=lambda x: x[0])
        if top_match[0] > 70: best_name = top_match[1]

    return res["PCS"], res["CTN"], best_name, raw_data

# ================= UI STREAMLIT (LENGKAP) =================
with st.sidebar:
    st.header("📍 DATA INPUT")
    m_code = st.text_input("MASTER CODE")
    date_inp = st.text_input("DATE")
    week_inp = st.text_input("WEEK")

files = st.file_uploader("📂 UPLOAD FOTO PRICE TAG", accept_multiple_files=True)

if files and m_code:
    if os.path.exists(FILE_PATH):
        db_ig = pd.read_excel(FILE_PATH, sheet_name=SHEET_MASTER_IG)
        db_targets = {s: pd.read_excel(FILE_PATH, sheet_name=s) for s in SHEETS_TARGET}
        list_names = db_ig[COL_IG_NAME].dropna().unique().tolist()
        
        final_list, zip_buffer = [], io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                with st.container(border=True):
                    img_pil = Image.open(f)
                    pcs, ctn, name, raw_txt = process_ocr_targeted(img_pil, list_names)
                    
                    # Match Prodcode
                    match_code = None
                    scores = [(fuzz.partial_ratio(str(r[COL_IG_NAME]).upper(), name), r["PRODCODE"]) for _, r in db_ig.iterrows()]
                    if scores:
                        best = max(scores, key=lambda x: x[0])
                        if best[0] > 75: match_code = str(best[1]).replace('.0','')
                    
                    st.write(f"#### 📄 {f.name}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("UNIT (Norm/Prom)", f"{pcs['n']:,}", f"{pcs['p']:,}")
                    c2.metric("CTN (Norm/Prom)", f"{ctn['n']:,}", f"{ctn['p']:,}")
                    c3.info(f"**Detect:** {name}\n\n**Code:** {match_code}")

                    with st.expander("🔍 Debug Raw Text"):
                        st.code(raw_txt)

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            m_row = df_t[(df_t["PRODCODE"].astype(str).str.replace('.0','') == match_code) & 
                                         (df_t["MASTER Code"].astype(str) == m_code)]
                            if not m_row.empty:
                                final_list.append({"sheet":s_name, "idx":m_row.index[0], "np":pcs['n'], "pp":pcs['p'], "nc":ctn['n'], "pc":ctn['p']})
                                buf = io.BytesIO()
                                img_pil.convert("RGB").save(buf, format="JPEG")
                                zf.writestr(f"{match_code}.jpg", buf.getvalue())
                                break

        if final_list:
            if st.button("🚀 UPDATE DATABASE SEKARANG", use_container_width=True):
                wb = load_workbook(FILE_PATH)
                for r in final_list:
                    ws = wb[r['sheet']]
                    headers = [str(c.value).strip() for c in ws[1]]
                    row_idx = r['idx'] + 2
                    mapping = {
                        "Normal Competitor Price (Pcs)": r['np'], 
                        "Promo Competitor Price (Pcs)": r['pp'], 
                        "Normal Competitor Price (Ctn)": r['nc'], 
                        "Promo Competitor Price (Ctn)": r['pc']
                    }
                    for col_name, val in mapping.items():
                        if col_name in headers:
                            ws.cell(row=row_idx, column=headers.index(col_name)+1).value = val if val != 0 else None
                wb.save(FILE_PATH)
                st.success("✅ Database Updated!")
                st.download_button("📥 Download Excel", open(FILE_PATH, "rb"), f"RESULT_{date_inp}.xlsx")
                st.download_button("🖼️ Download ZIP", zip_buffer.getvalue(), "BUKTI_FOTO.zip")
    else:
        st.error("File database tidak ditemukan!")

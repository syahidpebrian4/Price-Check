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

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

# --- CSS HEADER (TETAP SESUAI DESAIN ASLI) ---
logo_b64 = get_base64_image("lotte_logo.png")
st.markdown(f"""
    <style>
        .custom-header {{
            position: fixed; top: 0; left: 0; width: 100%; height: 90px;
            background-color: white; display: flex; align-items: center;
            padding: 0 30px; border-bottom: 3px solid #eeeeee; z-index: 999999;
        }}
        .header-logo {{ height: 55px; margin-right: 25px; }}
        .header-title {{ font-size: 38px; font-weight: 900; color: black; margin: 0; font-family: sans-serif; }}
        [data-testid="stSidebar"] {{ background-color: #FF0000 !important; margin-top: 90px !important; }}
        [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {{ color: white !important; font-weight: bold; }}
        .main .block-container {{ padding-top: 130px !important; }}
        header {{ visibility: hidden; }}
    </style>
    <div class="custom-header">
        <img src="data:image/png;base64,{logo_b64 if logo_b64 else ''}" class="header-logo">
        <h1 class="header-title">PRICE CHECK SYSTEM</h1>
    </div>
""", unsafe_allow_html=True)

# --- FUNGSI MEMBERSIHKAN ANGKA ---
def clean_val(s):
    if not s: return 0
    s = str(s).upper().replace('O','0').replace('S','5').replace('I','1').replace('B','8').replace('L','1').replace('G','6')
    res = re.sub(r'[^\d]', '', s)
    return int(res) if res else 0

# --- LOGIKA POSISI (URUTAN MUNCUL) ---
def get_prices_by_position(line_text):
    """Mengambil angka berdasarkan urutan kemunculan di baris"""
    # 1. Buang isi dalam kurung dulu
    clean_line = re.sub(r'\(.*?\)', ' ', line_text)
    # 2. Cari semua kandidat angka (4-15 digit)
    candidates = re.findall(r'[\d\.,OSIBLG]{4,15}', clean_line)
    
    extracted = []
    for c in candidates:
        val = clean_val(c)
        if 400 <= val <= 5000000:
            extracted.append(val)
    
    # Return urutan: [Normal, Promo]
    if len(extracted) >= 2:
        return extracted[0], extracted[1]
    elif len(extracted) == 1:
        return extracted[0], extracted[0]
    return 0, 0

def process_ocr_final(pil_img, master_names):
    img = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # OCR - Ambil baris demi baris
    raw_data = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
    lines = [l.strip().upper() for l in raw_data.split('\n') if l.strip()]
    
    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}
    
    # Kata kunci untuk identifikasi baris
    pcs_keywords = ["PCS", "RCG", "PCK", "BOX", "PES", "RC6", "B0X", "UNIT"]
    ctn_keywords = ["CTN", "CIN", "CTH", "CASE", "KARTON", "ISI"]

    for line in lines:
        # Jika baris mengandung keyword PCS
        if any(kw in line for kw in pcs_keywords):
            # Ambil berdasarkan POSISI kemunculan di teks
            n, p = get_prices_by_position(line)
            if n > 0:
                res["PCS"]["n"], res["PCS"]["p"] = n, p
        
        # Jika baris mengandung keyword CTN
        elif any(kw in line for kw in ctn_keywords):
            # Ambil berdasarkan POSISI kemunculan di teks
            n, p = get_prices_by_position(line)
            if n > 0:
                res["CTN"]["n"], res["CTN"]["p"] = n, p

    # Fuzzy Match Nama
    full_txt = " ".join(lines)
    best_name = "N/A"
    if master_names:
        scores = [(fuzz.partial_ratio(str(m).upper(), full_txt), m) for m in master_names]
        best_name = max(scores, key=lambda x: x[0])[1] if scores else "N/A"

    return res["PCS"], res["CTN"], best_name, "\n".join(lines)

# ================= UI STREAMLIT =================
with st.sidebar:
    m_code = st.text_input("📍 MASTER CODE").upper()
    date_inp = st.text_input("📅 DATE (DDMMYY)").upper()
    week_inp = st.text_input("🗓️ WEEK")

files = st.file_uploader("📂 UPLOAD GAMBAR", accept_multiple_files=True)

if files and m_code and date_inp:
    if os.path.exists(FILE_PATH):
        db_ig = pd.read_excel(FILE_PATH, sheet_name=SHEET_MASTER_IG)
        list_nama_master = db_ig[COL_IG_NAME].dropna().unique().tolist()
        db_targets = {s: pd.read_excel(FILE_PATH, sheet_name=s) for s in SHEETS_TARGET}
        
        final_list, zip_buffer = [], io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                with st.container(border=True):
                    img_pil = Image.open(f)
                    pcs, ctn, name, raw_txt = process_ocr_final(img_pil, list_nama_master)
                    
                    st.write(f"### 📄 {f.name}")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("PCS NORMAL", f"{pcs['n']:,}")
                        st.metric("PCS PROMO", f"{pcs['p']:,}")
                    with c2:
                        st.metric("CTN NORMAL", f"{ctn['n']:,}")
                        st.metric("CTN PROMO", f"{ctn['p']:,}")
                    with c3:
                        st.info(f"Produk: {name}")

                    with st.expander("HASIL SCAN"):
                        st.code(raw_txt)
                    
                    # Simpan data
                    match_code = None
                    scores = [(fuzz.partial_ratio(str(row[COL_IG_NAME]).upper(), name), row["PRODCODE"]) for _, row in db_ig.iterrows()]
                    if scores:
                        best = max(scores, key=lambda x: x[0])
                        if best[0] > 75: match_code = str(best[1]).replace('.0','')
                    
                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            m_row = df_t[(df_t["PRODCODE"].astype(str).str.replace('.0','') == match_code) & (df_t["MASTER Code"].astype(str) == m_code)]
                            if not m_row.empty:
                                final_list.append({"sheet":s_name, "idx":m_row.index[0], "np":pcs['n'], "pp":pcs['p'], "nc":ctn['n'], "pc":ctn['p']})
                                buf = io.BytesIO()
                                img_pil.convert("RGB").save(buf, format="JPEG")
                                zf.writestr(f"{match_code}.jpg", buf.getvalue())
                                break

        if final_list and st.button("🚀 UPDATE DATABASE", use_container_width=True):
            wb = load_workbook(FILE_PATH)
            for r in final_list:
                ws = wb[r['sheet']]
                headers = [str(c.value).strip() for c in ws[1]]
                for col_name, val in {"Normal Competitor Price (Pcs)": r['np'], "Promo Competitor Price (Pcs)": r['pp'], "Normal Competitor Price (Ctn)": r['nc'], "Promo Competitor Price (Ctn)": r['pc']}.items():
                    if col_name in headers: ws.cell(row=r['idx']+2, column=headers.index(col_name)+1).value = val if val != 0 else None
            wb.save(FILE_PATH)
            st.success("DATABASE UPDATED!")
            st.download_button("📥 DOWNLOAD EXCEL", open(FILE_PATH, "rb"), "RESULT.xlsx")
            st.download_button("🖼️ DOWNLOAD ZIP", zip_buffer.getvalue(), "FOTO.zip")

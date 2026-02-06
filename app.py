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

logo_b64 = get_base64_image("lotte_logo.png")
st.markdown(f"""
    <style>
        .custom-header {{
            position: fixed; top: 0; left: 0; width: 100%; height: 90px;
            background-color: white; display: flex; align-items: center;
            padding: 0 30px; border-bottom: 3px solid #eeeeee; z-index: 999999;
        }}
        .header-logo {{ height: 55px; margin-right: 25px; }}
        .header-title {{ font-size: 38px; font-weight: 900; color: black; margin: 0; }}
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

# --- CORE LOGIC ---
def clean_to_int(raw_str):
    if not raw_str: return 0
    s = str(raw_str).upper()
    # Koreksi karakter typo OCR
    s = s.replace('O', '0').replace('S', '5').replace('I', '1').replace('B', '8').replace('L', '1')
    digits = re.sub(r'[^\d]', '', s)
    return int(digits) if digits else 0

def extract_valid_prices(text_block):
    """Fungsi ekstraksi harga yang bersih dari kurung dan pemisah ISI"""
    if not text_block: return []
    # 1. Buang kurung (...)
    text_block = re.sub(r'\(.*?\)', ' ', text_block)
    # 2. Buang setelah tanda / atau kata ISI
    text_block = re.split(r'/|ISI', text_block, flags=re.IGNORECASE)[0]
    # 3. Cari angka minimal 4 digit (mencegah ambil angka promo beli 2 gratis 1)
    matches = re.findall(r'[\d\.,OSIBL]{4,12}', text_block)
    
    results = []
    for m in matches:
        val = clean_to_int(m)
        if 500 <= val <= 3000000: # Range harga masuk akal
            results.append(val)
    return results

def process_ocr_precision(pil_img, master_names):
    # Image Prep
    img = np.array(pil_img.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Pre-processing untuk teks kecil
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # OCR Data Frame (untuk tahu posisi koordinat teks)
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    df = pd.DataFrame(d)
    df = df[df['text'].str.strip() != ""]
    
    # Sort berdasarkan baris (top) lalu kolom (left)
    df['line_group'] = (df['top'] / 15).astype(int) # Grouping baris tipis
    df = df.sort_values(['top', 'left'])
    
    full_text_lines = []
    pcs_candidates = ""
    ctn_candidates = ""
    
    # Pisahkan teks ke kategori PCS atau CTN berdasarkan konten baris
    current_line_text = ""
    last_top = -1
    
    for _, row in df.iterrows():
        if last_top == -1 or abs(row['top'] - last_top) <= 15:
            current_line_text += " " + row['text']
        else:
            line_upper = current_line_text.upper()
            full_text_lines.append(line_upper)
            if "CTN" in line_upper:
                ctn_candidates += " " + line_upper
            else:
                pcs_candidates += " " + line_upper
            current_line_text = row['text']
        last_top = row['top']
    
    # Tambah baris terakhir
    line_upper = current_line_text.upper()
    full_text_lines.append(line_upper)
    if "CTN" in line_upper: ctn_candidates += " " + line_upper
    else: pcs_candidates += " " + line_upper

    # Ekstraksi Harga
    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}
    
    p_prices = extract_valid_prices(pcs_candidates)
    if len(p_prices) >= 2:
        res["PCS"]["n"], res["PCS"]["p"] = p_prices[0], p_prices[1]
    elif len(p_prices) == 1:
        res["PCS"]["n"] = res["PCS"]["p"] = p_prices[0]

    c_prices = extract_valid_prices(ctn_candidates)
    if len(c_prices) >= 1:
        res["CTN"]["n"] = res["CTN"]["p"] = c_prices[0]
        if len(c_prices) >= 2: res["CTN"]["p"] = c_prices[1]

    # Fuzzy Matching Nama
    full_txt_all = " ".join(full_text_lines)
    best_name = "N/A"
    if master_names:
        match_res = [(fuzz.partial_ratio(str(m).upper(), full_txt_all), m) for m in master_names]
        top_match = max(match_res, key=lambda x: x[0])
        if top_match[0] > 70: best_name = top_match[1]

    return res["PCS"], res["CTN"], best_name, "\n".join(full_text_lines)

# ================= UI =================
with st.sidebar:
    st.header("CONFIG")
    m_code = st.text_input("MASTER CODE")
    date_inp = st.text_input("DATE")
    week_inp = st.text_input("WEEK")

files = st.file_uploader("Upload Foto", accept_multiple_files=True)

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
                    pcs, ctn, name, raw_txt = process_ocr_precision(img_pil, list_names)
                    
                    # Match Prodcode
                    match_code = None
                    scores = [(fuzz.partial_ratio(str(r[COL_IG_NAME]).upper(), name), r["PRODCODE"]) for _, r in db_ig.iterrows()]
                    if scores:
                        best = max(scores, key=lambda x: x[0])
                        if best[0] > 75: match_code = norm(best[1]) if 'norm' in globals() else str(best[1])
                    
                    st.write(f"### {f.name}")
                    col1, col2 = st.columns(2)
                    col1.metric("PCS (N/P)", f"{pcs['n']:,} / {pcs['p']:,}")
                    col2.metric("CTN (N/P)", f"{ctn['n']:,} / {ctn['p']:,}")
                    st.write(f"**Detect:** {name} | **Code:** {match_code}")

                    with st.expander("🔍 Debug Raw Text"):
                        st.text(raw_txt)

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            m_row = df_t[(df_t["PRODCODE"].astype(str) == match_code) & 
                                         (df_t["MASTER Code"].astype(str) == m_code)]
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
                row_idx = r['idx'] + 2
                mapping = {"Normal Competitor Price (Pcs)": r['np'], "Promo Competitor Price (Pcs)": r['pp'], "Normal Competitor Price (Ctn)": r['nc'], "Promo Competitor Price (Ctn)": r['pc']}
                for col_name, val in mapping.items():
                    if col_name in headers:
                        ws.cell(row=row_idx, column=headers.index(col_name)+1).value = val if val != 0 else None
            wb.save(FILE_PATH)
            st.success("Updated!")
            st.download_button("Download Excel", open(FILE_PATH, "rb"), "RESULT.xlsx")
            st.download_button("Download ZIP", zip_buffer.getvalue(), "FOTO.zip")

import streamlit as st
import pytesseract
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
import gc
import base64

# ================= CONFIG & DATABASE =================
FILE_PATH = "database/master_harga.xlsx"
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 

st.set_page_config(page_title="Price Check Pro", layout="wide", initial_sidebar_state="expanded")

# --- FUNGSI HELPER ---
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

# --- CSS CUSTOM ---
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
        [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {{ color: white !important; }}
        .main .block-container {{ padding-top: 130px !important; }}
        header {{ visibility: hidden; }}
    </style>
    <div class="custom-header">
        <img src="data:image/png;base64,{logo_b64 if logo_b64 else ''}" class="header-logo">
        <h1 class="header-title">PRICE CHECK SYSTEM</h1>
    </div>
""", unsafe_allow_html=True)

def clean_price_val(raw_str):
    if not raw_str: return 0
    # Koreksi karakter umum yang sering salah baca oleh OCR
    s = str(raw_str).upper()
    s = s.replace('O', '0').replace('S', '5').replace('I', '1').replace('B', '8')
    clean = re.sub(r'[^\d]', '', s)
    return int(clean) if clean else 0

def get_prices_smart(text_segment):
    """Fungsi ekstraksi harga dengan pembersihan karakter non-harga"""
    if not text_segment: return []
    # 1. Hapus teks di dalam kurung agar tidak mengambil (RP 14.800 /PCS)
    text_segment = re.sub(r'\(.*?\)', '', text_segment)
    # 2. Hapus teks setelah kata ISI atau tanda /
    text_segment = re.split(r"/|ISI", text_segment)[0]
    # 3. Cari angka
    found = re.findall(r"(?:RP|R9|BP|RD|P)?\s?([\d\.,]{4,10})", text_segment)
    valid = []
    for f in found:
        val = clean_price_val(f)
        if 400 < val < 5000000: # Range diperluas
            valid.append(val)
    return valid

def process_ocr_final(pil_image, master_product_names=None):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
    df_ocr = pd.DataFrame(d)
    df_ocr = df_ocr[df_ocr['text'].str.strip() != ""]
    df_ocr['text'] = df_ocr['text'].str.upper()

    # Sorting urutan baca
    df_ocr = df_ocr.sort_values(by=['top', 'left'])
    lines_txt = []
    if not df_ocr.empty:
        curr_top = df_ocr.iloc[0]['top']
        line = []
        for _, row in df_ocr.iterrows():
            if row['top'] - curr_top > 15:
                lines_txt.append(" ".join(line))
                line = [row['text']]
                curr_top = row['top']
            else:
                line.append(row['text'])
        lines_txt.append(" ".join(line))

    full_text = " ".join(lines_txt)
    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}
    
    # --- PEMBAGIAN AREA (PARTITIONING) ---
    # Kita bagi teks menjadi bagian PCS dan bagian CTN
    if "CTN" in full_text:
        parts = full_text.split("CTN")
        pcs_section = parts[0]
        ctn_section = "CTN " + parts[1]
    else:
        pcs_section = full_text
        ctn_section = ""

    # --- EKSTRAKSI PCS ---
    # Cari angka di area PCS saja
    all_pcs_prices = get_prices_smart(pcs_section)
    if len(all_pcs_prices) >= 2:
        res["PCS"]["n"], res["PCS"]["p"] = all_pcs_prices[0], all_pcs_prices[1]
    elif len(all_pcs_prices) == 1:
        res["PCS"]["n"] = res["PCS"]["p"] = all_pcs_prices[0]

    # --- EKSTRAKSI CTN ---
    if ctn_section:
        all_ctn_prices = get_prices_smart(ctn_section)
        if len(all_ctn_prices) >= 2:
            res["CTN"]["n"], res["CTN"]["p"] = all_ctn_prices[0], all_ctn_prices[1]
        elif len(all_ctn_prices) == 1:
            res["CTN"]["n"] = res["CTN"]["p"] = all_ctn_prices[0]

    # --- NAMA PRODUK & PROMO ---
    prod_name = "N/A"
    if master_product_names:
        scores = [(fuzz.partial_ratio(str(name).upper(), full_text), name) for name in master_product_names]
        best = max(scores, key=lambda x: x[0])
        if best[0] > 70: prod_name = best[1]

    promo_desc = "-"
    m_promo = re.search(r"(BELI\s\d+\sGRATIS\s\d+|MIN\.\sBELI\s\d+)", full_text)
    if m_promo: promo_desc = m_promo.group(0)

    return res["PCS"], res["CTN"], prod_name, "\n".join(lines_txt), pil_image, promo_desc

# ================= UI STREAMLIT =================
def norm(val): return str(val).replace(".0", "").replace(" ", "").strip().upper()

with st.sidebar:
    st.header("INPUT DATA")
    m_code = st.text_input("📍 MASTER CODE").upper()
    date_inp = st.text_input("📅 DATE").upper()
    week_inp = st.text_input("🗓️ WEEK")

files = st.file_uploader("📂 UPLOAD GAMBAR", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code and date_inp and week_inp:
    if os.path.exists(FILE_PATH):
        db_ig = pd.read_excel(FILE_PATH, sheet_name=SHEET_MASTER_IG)
        db_targets = {s: pd.read_excel(FILE_PATH, sheet_name=s) for s in SHEETS_TARGET}
        list_nama_master = db_ig[COL_IG_NAME].dropna().unique().tolist()
        final_list, zip_buffer = [], io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                with st.container(border=True):
                    img_pil = Image.open(f)
                    pcs, ctn, name, raw_txt, red_img, p_desc = process_ocr_final(img_pil, list_nama_master)
                    
                    # Match Product Code
                    match_code = None
                    scores = [(fuzz.partial_ratio(str(row[COL_IG_NAME]).upper(), name), row["PRODCODE"]) for _, row in db_ig.iterrows()]
                    best_match = max(scores, key=lambda x: x[0])
                    if best_match[0] > 75: match_code = norm(best_match[1])
                    
                    st.markdown(f"### 📄 {f.name}")
                    col1, col2, col3 = st.columns([1,1,1])
                    col1.metric("UNIT (Norm/Prom)", f"{pcs['n']:,}", f"{pcs['p']:,}", delta_color="inverse")
                    col2.metric("CTN (Norm/Prom)", f"{ctn['n']:,}", f"{ctn['p']:,}", delta_color="inverse")
                    col3.info(f"**Matched:** {match_code}")
                    
                    with st.expander("🔍 DEBUG SCAN"):
                        st.code(raw_txt)

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            m_row = df_t[(df_t["PRODCODE"].astype(str).apply(norm) == match_code) & 
                                         (df_t["MASTER Code"].astype(str).apply(norm) == norm(m_code))]
                            if not m_row.empty:
                                final_list.append({
                                    "sheet": s_name, "idx": m_row.index[0], "p_desc": p_desc,
                                    "np": pcs['n'], "pp": pcs['p'], "nc": ctn['n'], "pc": ctn['p']
                                })
                                buf = io.BytesIO()
                                red_img.convert("RGB").save(buf, format="JPEG")
                                zf.writestr(f"{match_code}.jpg", buf.getvalue())
                                break
                gc.collect()

        if final_list:
            if st.button("🚀 UPDATE DATABASE & EXCEL"):
                wb = load_workbook(FILE_PATH)
                for r in final_list:
                    ws = wb[r['sheet']]
                    headers = [str(c.value).strip() for c in ws[1]]
                    row = r['idx'] + 2
                    mapping = {
                        "Normal Competitor Price (Pcs)": r['np'],
                        "Promo Competitor Price (Pcs)": r['pp'],
                        "Normal Competitor Price (Ctn)": r['nc'],
                        "Promo Competitor Price (Ctn)": r['pc'],
                        "Promosi Competitor": r['p_desc'] if r['p_desc'] != "-" else None
                    }
                    for col_name, val in mapping.items():
                        if col_name in headers:
                            ws.cell(row=row, column=headers.index(col_name) + 1).value = val if val != 0 else None
                wb.save(FILE_PATH)
                st.success("DATABASE UPDATED!")
                with open(FILE_PATH, "rb") as f:
                    st.download_button("📥 DOWNLOAD EXCEL", f, f"RESULT_{date_inp}.xlsx")
                st.download_button("🖼️ DOWNLOAD ZIP FOTO", zip_buffer.getvalue(), "images.zip")
    else:
        st.error("Excel database not found!")

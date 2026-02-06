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

# --- FUNGSI HELPER: LOGO BASE64 ---
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

# --- CSS CUSTOM HEADER (TAMPILAN TETAP SAMA) ---
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

# --- LOGIKA MEMBERSIHKAN ANGKA ---
def clean_price_val(raw_str):
    if not raw_str: return 0
    s = str(raw_str).upper()
    s = s.replace('O', '0').replace('S', '5').replace('I', '1').replace('B', '8').replace('G', '6').replace('L', '1')
    clean = re.sub(r'[^\d]', '', s)
    return int(clean) if clean else 0

# --- LOGIKA TARGET KEYWORDS (PCS, RCG, PCK, BOX) ---
def get_prices_by_anchors(text, is_ctn=False):
    found_prices = []
    text = re.sub(r'\(.*?\)', ' ', text) # Tetap buang isi kurung
    
    if is_ctn:
        anchors = ["CTN", "CIN", "CTH"]
    else:
        anchors = ["PCS", "PES", "PC5", "RCG", "RC6", "PCK", "BOX", "B0X"]

    for anchor in anchors:
        # Cari angka SETELAH anchor, berhenti SEBELUM / atau ISI
        pattern = rf"{anchor}(.*?)(?=/|ISI|RP|$)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for m in matches:
            nums = re.findall(r'[\d\.,OSIBLG]{4,15}', m)
            for n in nums:
                val = clean_price_val(n)
                if 400 <= val <= 4000000:
                    if val not in found_prices:
                        found_prices.append(val)
    return found_prices

def process_ocr_final(pil_image, master_product_names=None):
    w, h = pil_image.size
    img_resized = pil_image.resize((w*2, h*2), Image.Resampling.LANCZOS)
    img_np = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    raw_data = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
    full_text = raw_data.upper().replace('\n', '  ')

    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}
    
    # Ambil Harga PCS (Normal & Promo)
    pcs_list = get_prices_by_anchors(full_text, is_ctn=False)
    if len(pcs_list) >= 2:
        res["PCS"]["n"], res["PCS"]["p"] = pcs_list[0], pcs_list[1]
    elif len(pcs_list) == 1:
        res["PCS"]["n"] = res["PCS"]["p"] = pcs_list[0]

    # Ambil Harga CTN (Normal & Promo)
    ctn_list = get_prices_by_anchors(full_text, is_ctn=True)
    if len(ctn_list) >= 2:
        res["CTN"]["n"], res["CTN"]["p"] = ctn_list[0], ctn_list[1]
    elif len(ctn_list) == 1:
        res["CTN"]["n"] = res["CTN"]["p"] = ctn_list[0]

    # Fuzzy Match Nama Produk
    prod_name = "N/A"
    if master_product_names:
        scores = [(fuzz.partial_ratio(str(m).upper(), full_text), m) for m in master_product_names]
        if scores:
            best = max(scores, key=lambda x: x[0])
            if best[0] > 70: prod_name = best[1]

    return res["PCS"], res["CTN"], prod_name, raw_data

# ================= UI STREAMLIT (FULL) =================
def norm(val): return str(val).replace(".0", "").replace(" ", "").strip().upper()

with st.sidebar:
    st.header("INPUT DATA")
    m_code = st.text_input("📍 MASTER CODE").upper()
    date_inp = st.text_input("📅 DATE (DDMMYY)").upper()
    week_inp = st.text_input("🗓️ WEEK")
    st.divider()

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
                    pcs, ctn, name, raw_txt = process_ocr_final(img_pil, list_nama_master)
                    
                    match_code = None
                    scores = [(fuzz.partial_ratio(str(row[COL_IG_NAME]).upper(), name), row["PRODCODE"]) for _, row in db_ig.iterrows()]
                    if scores:
                        best_match = max(scores, key=lambda x: x[0])
                        if best_match[0] > 75: match_code = norm(best_match[1])
                    
                    st.markdown(f"### 📄 {f.name}")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        st.metric("PCS NORMAL", f"{pcs['n']:,}")
                        st.metric("PCS PROMO", f"{pcs['p']:,}")
                    with col2:
                        st.metric("CTN NORMAL", f"{ctn['n']:,}")
                        st.metric("CTN PROMO", f"{ctn['p']:,}")
                    with col3:
                        st.info(f"**Produk:** {name}")
                        st.success(f"**Code:** {match_code}")
                    
                    with st.expander("🔍 LIHAT HASIL SCAN (DEBUG)"):
                        st.code(raw_txt)

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            m_row = df_t[(df_t["PRODCODE"].astype(str).apply(norm) == match_code) & 
                                         (df_t["MASTER Code"].astype(str).apply(norm) == norm(m_code))]
                            if not m_row.empty:
                                final_list.append({
                                    "sheet": s_name, "idx": m_row.index[0],
                                    "np": pcs['n'], "pp": pcs['p'], "nc": ctn['n'], "pc": ctn['p']
                                })
                                buf = io.BytesIO()
                                img_pil.convert("RGB").save(buf, format="JPEG")
                                zf.writestr(f"{match_code}.jpg", buf.getvalue())
                                break
                gc.collect()

        if final_list:
            st.divider()
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
                            target_col = headers.index(col_name) + 1
                            ws.cell(row=row_idx, column=target_col).value = val if val != 0 else None
                
                wb.save(FILE_PATH)
                st.success("✅ DATABASE BERHASIL DIUPDATE!")
                
                with open(FILE_PATH, "rb") as f_excel:
                    st.download_button("📥 DOWNLOAD EXCEL HASIL", f_excel, f"RESULT_W{week_inp}_{date_inp}.xlsx", use_container_width=True)
                st.download_button("🖼️ DOWNLOAD ZIP BUKTI FOTO", zip_buffer.getvalue(), f"BUKTI_W{week_inp}.zip", use_container_width=True)
    else:
        st.error(f"File {FILE_PATH} tidak ditemukan.")

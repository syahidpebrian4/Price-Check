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

# --- CSS HEADER LOTTE (TETAP SAMA) ---
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

def clean_val(s):
    if not s: return 0
    s = str(s).upper().replace('O','0').replace('S','5').replace('I','1').replace('B','8').replace('L','1')
    res = re.sub(r'[^\d]', '', s)
    return int(res) if res else 0

def process_ocr_geometry(pil_img, master_names):
    img = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Ambil data OCR beserta koordinat (lebar, tinggi, posisi x, posisi y)
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['text'])
    
    anchors = {"PCS": [], "CTN": []}
    price_candidates = []

    for i in range(n_boxes):
        text = d['text'][i].upper()
        if not text.strip(): continue
        
        # Koordinat tengah box
        curr_x = d['left'][i] + (d['width'][i] / 2)
        curr_y = d['top'][i] + (d['height'][i] / 2)

        # 1. Cari Anchor (PCS atau CTN)
        if any(kw in text for kw in ["PCS", "RCG", "PCK", "BOX", "PES"]):
            anchors["PCS"].append((curr_x, curr_y))
        elif any(kw in text for kw in ["CTN", "CIN", "CASE", "KARTON"]):
            anchors["CTN"].append((curr_x, curr_y))
        
        # 2. Cari Angka
        val = clean_val(text)
        if 400 <= val <= 5000000:
            price_candidates.append({'val': val, 'x': curr_x, 'y': curr_y})

    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}

    # Fungsi untuk cari harga terdekat dari anchor
    def get_closest_prices(anchor_list, candidates):
        if not anchor_list: return []
        # Ambil anchor pertama yang ketemu
        ax, ay = anchor_list[0]
        # Hitung jarak Euclidean ke semua angka
        sorted_by_dist = sorted(candidates, key=lambda c: ((c['x']-ax)**2 + (c['y']-ay)**2)**0.5)
        # Ambil 2 terdekat
        return [c['val'] for c in sorted_by_dist[:2]]

    pcs_found = get_closest_prices(anchors["PCS"], price_candidates)
    if len(pcs_found) >= 1:
        res["PCS"]["n"] = pcs_found[0]
        res["PCS"]["p"] = pcs_found[1] if len(pcs_found) > 1 else pcs_found[0]

    ctn_found = get_closest_prices(anchors["CTN"], price_candidates)
    if len(ctn_found) >= 1:
        res["CTN"]["n"] = ctn_found[0]
        res["CTN"]["p"] = ctn_found[1] if len(ctn_found) > 1 else ctn_found[0]

    # Matching Nama (Full Text)
    full_text = " ".join(d['text'])
    best_name = "N/A"
    if master_names:
        scores = [(fuzz.partial_ratio(str(m).upper(), full_text.upper()), m) for m in master_names]
        best_name = max(scores, key=lambda x: x[0])[1] if scores else "N/A"

    return res["PCS"], res["CTN"], best_name, full_text

# ================= UI STREAMLIT =================
with st.sidebar:
    m_code = st.text_input("📍 MASTER CODE").upper()
    date_inp = st.text_input("📅 DATE").upper()
    week_inp = st.text_input("🗓️ WEEK")

files = st.file_uploader("📂 UPLOAD GAMBAR", accept_multiple_files=True)

if files and m_code:
    if os.path.exists(FILE_PATH):
        db_ig = pd.read_excel(FILE_PATH, sheet_name=SHEET_MASTER_IG)
        list_names = db_ig[COL_IG_NAME].dropna().unique().tolist()
        db_targets = {s: pd.read_excel(FILE_PATH, sheet_name=s) for s in SHEETS_TARGET}
        
        final_list, zip_buffer = [], io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                with st.container(border=True):
                    pcs, ctn, name, raw_txt = process_ocr_geometry(Image.open(f), list_names)
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
                    
                    # Pencarian Prodcode & Simpan
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
                                Image.open(f).convert("RGB").save(buf, format="JPEG")
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

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

st.set_page_config(page_title="Price Check", layout="wide", initial_sidebar_state="expanded")

# --- FUNGSI HELPER: LOGO BASE64 ---
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

# --- CSS CUSTOM: FIXED HEADER & SIDEBAR MERAH PERMANEN ---
logo_b64 = get_base64_image("lotte_logo.png")

st.markdown(f"""
    <style>
        /* 1. Header Putih Fixed di Atas */
        .custom-header {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 90px;
            background-color: white;
            display: flex;
            align-items: center;
            padding: 0 30px;
            border-bottom: 3px solid #eeeeee;
            z-index: 999999;
        }}
        .header-logo {{
            height: 55px;
            margin-right: 25px;
        }}
        .header-title {{
            font-size: 42px;
            font-weight: 900;
            font-family: 'Arial Black', sans-serif;
            color: black;
            margin: 0;
        }}

        /* 2. Sidebar Merah Permanen (Tidak bisa ditutup) */
        [data-testid="stSidebar"] {{
            background-color: #FF0000 !important;
            margin-top: 90px !important;
            min-width: 320px !important;
            max-width: 320px !important;
        }}
        
        /* Sembunyikan tombol toggle/tutup sidebar */
        [data-testid="stSidebarNav"] + div, button[kind="headerNoSpacing"] {{
            display: none !important;
        }}

        /* 3. Teks & Input Sidebar agar Putih */
        [data-testid="stSidebar"] .stMarkdown p, 
        [data-testid="stSidebar"] label {{
            color: white !important;
            font-weight: bold !important;
            font-size: 14px;
        }}
        
        /* 4. Padding Area Utama agar tidak tertutup header */
        .main .block-container {{
            padding-top: 130px !important;
        }}

        /* 5. Sembunyikan Header bawaan Streamlit */
        header {{visibility: hidden;}}
    </style>
    
    <div class="custom-header">
        <img src="data:image/png;base64,{logo_b64 if logo_b64 else ''}" class="header-logo">
        <h1 class="header-title">PRICE CHECK</h1>
    </div>
""", unsafe_allow_html=True)

# --- FUNGSI LOGIKA OCR (SESUAI REQUEST ANDA) ---
def clean_price_val(raw_str):
    if not raw_str: return 0
    clean = re.sub(r'[^\d]', '', str(raw_str))
    return int(clean) if clean else 0

def process_ocr_final(pil_image, master_product_names=None):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
    df_ocr = pd.DataFrame(d)
    df_ocr = df_ocr[df_ocr['text'].str.strip() != ""]
    df_ocr['text'] = df_ocr['text'].str.upper()

    df_ocr = df_ocr.sort_values(by=['top', 'left'])
    lines_data = []
    if not df_ocr.empty:
        current_top = df_ocr.iloc[0]['top']
        temp_words = []
        for _, row in df_ocr.iterrows():
            if row['top'] - current_top > 15:
                temp_words.sort(key=lambda x: x['left'])
                lines_data.append({
                    "text": " ".join([w['text'] for w in temp_words]),
                    "top": current_top,
                    "h": max([w['height'] for w in temp_words])
                })
                temp_words = [{'text': row['text'], 'left': row['left'], 'height': row['height']}]
                current_top = row['top']
            else:
                temp_words.append({'text': row['text'], 'left': row['left'], 'height': row['height']})
        temp_words.sort(key=lambda x: x['left'])
        lines_data.append({"text": " ".join([w['text'] for w in temp_words]), "top": current_top, "h": 10})

    lines_txt = [l['text'] for l in lines_data]
    full_text_single = " ".join(lines_txt)
    raw_ocr_output = "\n".join(lines_txt)

    prod_name, promo_desc = "N/A", "-"
    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}
    draw = ImageDraw.Draw(pil_image)

    if master_product_names:
        best_match, highest_score = "N/A", 0
        for ref_name in master_product_names:
            score = fuzz.partial_ratio(str(ref_name).upper(), full_text_single)
            if score > 80 and score > highest_score:
                highest_score, best_match = score, str(ref_name).upper()
        prod_name = best_match

    anchor_nav = "SEMUA KATEGORI"
    for i, line in enumerate(lines_txt):
        if fuzz.partial_ratio(anchor_nav, line) > 65:
            y_coord = lines_data[i]['top'] / scale
            if y_coord < (pil_image.height * 0.3):
                h_box = min(lines_data[i]['h'] / scale, 40)
                draw.rectangle([0, y_coord - 5, pil_image.width, y_coord + h_box + 5], fill="white")
                break

    def get_prices(text_segment):
        found = re.findall(r"(?:RP|R9|BP|RD|P)?\s?([\d\.,]{4,9})", text_segment)
        return [clean_price_val(f) for f in found if 500 < clean_price_val(f) < 2000000]

    pcs_area = re.split(r"(PILIH SATUAN|TERMURAH|PCS|RCG|PCH|PCK)", full_text_single)
    if len(pcs_area) > 1:
        prices = get_prices(" ".join(pcs_area[1:]))
        if len(prices) >= 2: res["PCS"]["n"], res["PCS"]["p"] = max(prices[:2]), min(prices[:2])
        elif len(prices) == 1: res["PCS"]["n"] = res["PCS"]["p"] = prices[0]

    if "CTN" in full_text_single:
        prices_ctn = get_prices(full_text_single.split("CTN")[-1])
        if prices_ctn: res["CTN"]["n"] = res["CTN"]["p"] = prices_ctn[0]

    anchor_promo = "MAU LEBIH UNTUNG? CEK MEKANISME PROMO BERIKUT"
    for i, line in enumerate(lines_txt):
        if anchor_promo in line:
            promo_lines = [lines_txt[j] for j in range(i + 1, min(i + 3, len(lines_txt)))]
            promo_desc = " ".join(promo_lines).split("=")[0].strip()
            break
    
    if promo_desc == "-":
        m_promo = re.search(r"(BELI\s\d+\sGRATIS\s\d+|MIN\.\sBELI\s\d+)", full_text_single)
        if m_promo: promo_desc = m_promo.group(0)

    return res["PCS"], res["CTN"], prod_name, raw_ocr_output, pil_image, promo_desc

# ================= UI SIDEBAR & CONTENT =================

def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

# Sidebar Merah Fixed
with st.sidebar:
    st.write("---")
    m_code = st.text_input("📍 MASTER CODE", value="6002").upper()
    date_inp = st.text_input("📅 DAY", value="22JAN2026").upper()
    week_inp = st.text_input("🗓️ WEEK", value="2")
    st.write("---")

# Konten Utama
st.markdown("### 📂 UPLOAD SCREENSHOTS")
files = st.file_uploader("", type=["jpg", "png", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")

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
                    
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        score = fuzz.partial_ratio(str(row[COL_IG_NAME]).upper(), name)
                        if score > 75 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])
                    
                    st.markdown(f"**📄 File:** {f.name} | **Match:** `{name}`")
                    c1, c2, c3 = st.columns([1, 1, 2])
                    c1.metric("PCS (N/P)", f"{pcs['n']:,} / {pcs['p']:,}")
                    c2.metric("CTN", f"{ctn['n']:,}")
                    c3.success(f"Promo: {p_desc}")

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            match_row = df_t[(df_t["PRODCODE"].astype(str).apply(norm) == match_code) & 
                                             (df_t["MASTER Code"].astype(str).apply(norm) == norm(m_code))]
                            if not match_row.empty:
                                final_list.append({
                                    "prodcode": match_code, "sheet": s_name, "index": match_row.index[0],
                                    "n_pcs": pcs['n'], "p_pcs": pcs['p'], "n_ctn": ctn['n'], "p_ctn": ctn['p'], "p_desc": p_desc
                                })
                                buf = io.BytesIO()
                                red_img.convert("RGB").save(buf, format="JPEG")
                                zf.writestr(f"{match_code}.jpg", buf.getvalue())
                                break
                gc.collect()

        if final_list:
            st.divider()
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🚀 UPDATE DATABASE", use_container_width=True):
                    wb = load_workbook(FILE_PATH)
                    for r in final_list:
                        ws = wb[r['sheet']]
                        headers = [str(c.value).strip() for c in ws[1]]
                        row_num = r['index'] + 2
                        
                        def empty_if_zero(val): return val if val != 0 else None

                        mapping = {
                            "Normal Competitor Price (Pcs)": empty_if_zero(r['n_pcs']),
                            "Promo Competitor Price (Pcs)": empty_if_zero(r['p_pcs']),
                            "Normal Competitor Price (Ctn)": empty_if_zero(r['n_ctn']),
                            "Promo Competitor Price (Ctn)": empty_if_zero(r['p_ctn']),
                            "Promosi Competitor": r['p_desc'] if r['p_desc'] != "-" else None
                        }
                        for col_name, val in mapping.items():
                            if col_name in headers:
                                ws.cell(row=row_num, column=headers.index(col_name) + 1).value = val
                    
                    wb.save(FILE_PATH)
                    st.success("✅ DATABASE UPDATED!")
                    with open(FILE_PATH, "rb") as f_excel:
                        st.download_button("📥 DOWNLOAD EXCEL", f_excel, f"Update_{date_inp}.xlsx", use_container_width=True)
            with col_btn2:
                st.download_button("🖼️ DOWNLOAD FOTO", zip_buffer.getvalue(), f"{m_code}.zip", use_container_width=True)
    else:
        st.error("Database Excel tidak ditemukan!")

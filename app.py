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

# ================= CONFIG & DATABASE =================
FILE_PATH = "database/master_harga.xlsx"
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 

st.set_page_config(page_title="Price Check", layout="wide")

# CSS: Membuat area logo putih dan sisa sidebar merah
st.markdown("""
    <style>
        /* Samping Sidebar (Background Merah) */
        [data-testid="stSidebar"] {
            background-color: #FF0000;
        }
        
        /* Area Logo di Sidebar (Background Putih) */
        .logo-container {
            background-color: white;
            padding: 20px;
            margin: -60px -20px 20px -20px;
            text-align: center;
            border-bottom: 5px solid #E5E5E5;
        }

        /* Teks Label di Sidebar agar Putih */
        [data-testid="stSidebar"] .stMarkdown p, 
        [data-testid="stSidebar"] label {
            color: white !important;
            font-weight: bold;
            font-size: 16px;
        }
        
        /* Tombol di sidebar agar lebar penuh */
        .stButton>button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

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

    # --- A. NAMA PRODUK ---
    if master_product_names:
        best_match, highest_score = "N/A", 0
        for ref_name in master_product_names:
            score = fuzz.partial_ratio(str(ref_name).upper(), full_text_single)
            if score > 80 and score > highest_score:
                highest_score, best_match = score, str(ref_name).upper()
        prod_name = best_match

    # --- B. SENSOR ---
    for i, line in enumerate(lines_txt):
        if any(k in line for k in ["SEMUA", "KATEGORI", "CARI", "INDOGROSIR"]):
            y_coord = lines_data[i]['top'] / scale
            if y_coord < (pil_image.height * 0.3):
                h_box = min(lines_data[i]['h'] / scale, 40)
                draw.rectangle([0, y_coord-5, pil_image.width, y_coord+h_box+5], fill="white")
                break

    # --- C. HARGA ---
    def get_prices(segment):
        found = re.findall(r"(?:RP|R9|BP|RD|P)?\s?([\d\.,]{4,9})", segment)
        return [clean_price_val(f) for f in found if 500 < clean_price_val(f) < 2000000]

    pcs_split = re.split(r"(PILIH SATUAN|TERMURAH|PCS|RCG|PCH|PCK)", full_text_single)
    if len(pcs_split) > 1:
        prices = get_prices(" ".join(pcs_split[1:]))
        if len(prices) >= 2: res["PCS"]["n"], res["PCS"]["p"] = max(prices[:2]), min(prices[:2])
        elif len(prices) == 1: res["PCS"]["n"] = res["PCS"]["p"] = prices[0]

    if "CTN" in full_text_single:
        c_prices = get_prices(full_text_single.split("CTN")[-1])
        if c_prices: res["CTN"]["n"] = res["CTN"]["p"] = c_prices[0]

    # --- D. PROMO ---
    if "MAU LEBIH UNTUNG" in full_text_single:
        for i, line in enumerate(lines_txt):
            if "MAU LEBIH UNTUNG" in line:
                p_txt = " ".join(lines_txt[i+1:i+3]).split("=")[0].strip()
                promo_desc = re.sub(r'\bRAP\b|\|', '', p_txt).strip()
                break
    else:
        m_promo = re.search(r"(BELI\s\d+\sGRATIS\s\d+|MIN\.\sBELI\s\d+)", full_text_single)
        if m_promo: promo_desc = m_promo.group(0)

    return res["PCS"], res["CTN"], prod_name, raw_ocr_output, pil_image, promo_desc

# ================= UI STREAMLIT =================

# --- SIDEBAR ---
with st.sidebar:
    # Mengubah Link Google Drive menjadi Direct Image Link
    file_id = "1N733EZVansjReDoxh1xVp-_fkrVY1q5C"
    direct_link = f"https://drive.google.com/uc?export=view&id={file_id}"
    
    # Area Logo dengan background putih
    st.markdown(f"""
        <div class="logo-container">
            <img src="{direct_link}" width="150">
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; color: white;'>PRICE CHECK SYSTEM</p>", unsafe_allow_html=True)
    st.divider()
    
    m_code = st.text_input("🔑 MASTER CODE", placeholder="6002").upper()
    date_inp = st.text_input("📅 DAY", placeholder="22JAN2026").upper()
    week_inp = st.text_input("🗓️ WEEK", placeholder="2")
    
    st.divider()

# --- MAIN PAGE ---
st.title("📂 Upload Screenshots")
files = st.file_uploader("", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

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
                            best_score, match_code = score, str(row["PRODCODE"]).replace(".0","").strip()
                    
                    st.markdown(f"#### 📄 {f.name}")
                    c1, c2 = st.columns([2, 1])
                    with c1: st.write(f"**OCR Name:** `{name}`")
                    with c2: 
                        if match_code: st.success(f"**Matched Code:** {match_code}")
                        else: st.warning("⚠️ Code Not Found")

                    m1, m2, m3 = st.columns([1, 1, 2])
                    m1.metric("UNIT", f"{pcs['n']:,} / {pcs['p']:,}")
                    m2.metric("CTN", f"{ctn['n']:,} / {ctn['p']:,}")
                    m3.success(f"**Promo:** {p_desc}")

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            match_row = df_t[(df_t["PRODCODE"].astype(str).str.contains(match_code)) & 
                                             (df_t["MASTER Code"].astype(str).str.contains(m_code))]
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
                        def e_z(v): return v if v != 0 else None
                        mapping = {
                            "Normal Competitor Price (Pcs)": e_z(r['n_pcs']),
                            "Promo Competitor Price (Pcs)": e_z(r['p_pcs']),
                            "Normal Competitor Price (Ctn)": e_z(r['n_ctn']),
                            "Promo Competitor Price (Ctn)": e_z(r['p_ctn']),
                            "Promosi Competitor": r['p_desc'] if r['p_desc'] != "-" else None
                        }
                        for col_name, val in mapping.items():
                            if col_name in headers:
                                ws.cell(row=row_num, column=headers.index(col_name) + 1).value = val
                    wb.save(FILE_PATH)
                    st.success("✅ DATABASE UPDATED!")
                    st.download_button("📥 DOWNLOAD EXCEL", open(FILE_PATH, "rb"), f"Price Check W{week_inp}_{date_inp}.xlsx", use_container_width=True)
            with col_btn2:
                st.download_button("🖼️ DOWNLOAD FOTO", zip_buffer.getvalue(), f"{m_code}.zip", use_container_width=True)

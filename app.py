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

# --- CSS CUSTOM HEADER LOTTE ---
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

# --- FUNGSI LOGIKA OCR ---
def clean_price_val(raw_str):
    """Membersihkan teks dan memperbaiki salah baca OCR (O=0, S=5, I=1, dll)"""
    if not raw_str: return 0
    s = str(raw_str).upper()
    # Kamus koreksi karakter yang sering tertukar
    s = s.replace('O', '0').replace('S', '5').replace('I', '1').replace('B', '8').replace('G', '6').replace('L', '1')
    clean = re.sub(r'[^\d]', '', s)
    return int(clean) if clean else 0

def get_prices_smart(text_segment):
    """Logika: Abaikan kurung, abaikan teks setelah / atau ISI"""
    if not text_segment: return []
    
    # 1. Hapus teks di dalam kurung (...) beserta kurungnya
    text_segment = re.sub(r'\(.*?\)', ' ', text_segment)
    
    # 2. Ambil hanya bagian sebelum tanda / atau kata ISI
    text_segment = re.split(r"/|ISI", text_segment)[0]
    
    # 3. Cari deretan angka (Regex ini mengizinkan karakter salah baca untuk ditangkap dulu)
    found = re.findall(r"[\d\.,O S I B L G]{4,15}", text_segment)
    
    valid = []
    for f in found:
        val = clean_price_val(f)
        # Filter range harga masuk akal (400 perak - 5 juta)
        if 400 <= val <= 5000000:
            valid.append(val)
    return valid

def process_ocr_final(pil_image, master_product_names=None):
    # Pre-processing: Perbesar 2x dan GrayScale untuk akurasi
    w, h = pil_image.size
    img_resized = pil_image.resize((w*2, h*2), Image.Resampling.LANCZOS)
    img_np = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    # OCR
    raw_data = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
    lines = [l.strip().upper() for l in raw_data.split('\n') if l.strip()]
    full_text = " ".join(lines)

    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}
    
    # --- PEMISAHAN AREA ---
    if "CTN" in full_text:
        parts = re.split(r"CTN", full_text, maxsplit=1)
        pcs_section = parts[0]
        ctn_section = "CTN " + parts[1]
    else:
        pcs_section = full_text
        ctn_section = ""

    # Harga PCS (1st=Normal, 2nd=Promo)
    prices_pcs = get_prices_smart(pcs_section)
    if len(prices_pcs) >= 2:
        res["PCS"]["n"], res["PCS"]["p"] = prices_pcs[0], prices_pcs[1]
    elif len(prices_pcs) == 1:
        res["PCS"]["n"] = res["PCS"]["p"] = prices_pcs[0]

    # Harga CTN (1st=Normal, 2nd=Promo)
    if ctn_section:
        prices_ctn = get_prices_smart(ctn_section)
        if len(prices_ctn) >= 2:
            res["CTN"]["n"], res["CTN"]["p"] = prices_ctn[0], prices_ctn[1]
        elif len(prices_ctn) == 1:
            res["CTN"]["n"] = res["CTN"]["p"] = prices_ctn[0]

    # Fuzzy Match Nama Produk
    prod_name = "N/A"
    if master_product_names:
        scores = [(fuzz.partial_ratio(str(m).upper(), full_text), m) for m in master_product_names]
        if scores:
            best = max(scores, key=lambda x: x[0])
            if best[0] > 70: prod_name = best[1]

    # Deteksi Promo (Mekanisme)
    promo_desc = "-"
    m_promo = re.search(r"(BELI\s\d+\sGRATIS\s\d+|MIN\.\sBELI\s\d+)", full_text)
    if m_promo: promo_desc = m_promo.group(0)

    return res["PCS"], res["CTN"], prod_name, "\n".join(lines), pil_image, promo_desc

# ================= UI STREAMLIT =================
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
        # Load Database
        db_ig = pd.read_excel(FILE_PATH, sheet_name=SHEET_MASTER_IG)
        db_targets = {s: pd.read_excel(FILE_PATH, sheet_name=s) for s in SHEETS_TARGET}
        list_nama_master = db_ig[COL_IG_NAME].dropna().unique().tolist()
        
        final_list, zip_buffer = [], io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                with st.container(border=True):
                    img_pil = Image.open(f)
                    pcs, ctn, name, raw_txt, enhanced_img, p_desc = process_ocr_final(img_pil, list_nama_master)
                    
                    # Match Product Code dari database IG
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
                        st.info(f"**Nama Produk:** {name}")
                        st.success(f"**Prod Code:** {match_code}")
                        st.warning(f"**Promo:** {p_desc}")
                    
                    with st.expander("🔍 LIHAT HASIL SCAN (DEBUG)"):
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
                                # Simpan foto yang diproses ke ZIP
                                buf = io.BytesIO()
                                img_pil.save(buf, format="JPEG")
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
                        "Promo Competitor Price (Ctn)": r['pc'],
                        "Promosi Competitor": r['p_desc'] if r['p_desc'] != "-" else None
                    }
                    
                    for col_name, val in mapping.items():
                        if col_name in headers:
                            target_col = headers.index(col_name) + 1
                            ws.cell(row=row_idx, column=target_col).value = val if val != 0 else None
                
                wb.save(FILE_PATH)
                st.success("✅ DATABASE BERHASIL DIUPDATE!")
                
                # Sediakan tombol download setelah update
                with open(FILE_PATH, "rb") as f_excel:
                    st.download_button("📥 DOWNLOAD EXCEL HASIL", f_excel, f"RESULT_W{week_inp}_{date_inp}.xlsx", use_container_width=True)
                st.download_button("🖼️ DOWNLOAD ZIP BUKTI FOTO", zip_buffer.getvalue(), f"BUKTI_W{week_inp}.zip", use_container_width=True)
    else:
        st.error(f"File {FILE_PATH} tidak ditemukan. Pastikan folder 'database' ada.")

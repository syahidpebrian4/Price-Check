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

st.set_page_config(page_title="Price Check V13.2 - Full Scan Preview", layout="wide")

def clean_price_val(raw_str):
    if not raw_str: return 0
    clean = re.sub(r'[^\d]', '', str(raw_str))
    return int(clean) if clean else 0

def process_ocr_strict(pil_image):
    # 1. Preprocessing
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. OCR Execution
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
    df_ocr = pd.DataFrame(d)
    df_ocr = df_ocr[df_ocr['text'].str.strip() != ""]
    df_ocr['text'] = df_ocr['text'].str.upper()

    # 3. Baris demi Baris (Line Grouping Logic)
    df_ocr = df_ocr.sort_values(by=['top', 'left'])
    lines_data = []
    if not df_ocr.empty:
        current_top = df_ocr.iloc[0]['top']
        temp_line_words = []
        for _, row in df_ocr.iterrows():
            if row['top'] - current_top > 15:
                temp_line_words.sort(key=lambda x: x['left'])
                lines_data.append({
                    "text": " ".join([w['text'] for w in temp_line_words]),
                    "top": current_top,
                    "height": max([w['height'] for w in temp_line_words])
                })
                temp_line_words = [{'text': row['text'], 'left': row['left'], 'height': row['height']}]
                current_top = row['top']
            else:
                temp_line_words.append({'text': row['text'], 'left': row['left'], 'height': row['height']})
        temp_line_words.sort(key=lambda x: x['left'])
        lines_data.append({
            "text": " ".join([w['text'] for w in temp_line_words]),
            "top": current_top,
            "height": max([w['height'] for w in temp_line_words])
        })

    lines_txt = [l['text'] for l in lines_data]
    full_text_single = " ".join(lines_txt)
    raw_ocr_output = "\n".join(lines_txt) # Ini yang akan ditampilkan di preview

    # 4. Inisialisasi Extraction
    prod_name, promo_desc = "N/A", "-"
    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}
    draw = ImageDraw.Draw(pil_image)

    # A. Nama & Sensor (Anchor: SEMUA KATEGORI)
    anchor_header = "SEMUA KATEGORI ~ CARI DI KLIK INDOGROSIR"
    for i, line in enumerate(lines_txt):
        if fuzz.partial_ratio(anchor_header, line) > 80:
            y, h = lines_data[i]['top'] / scale, lines_data[i]['height'] / scale
            draw.rectangle([0, y - 5, pil_image.width, y + h + 5], fill="white")
            if i + 1 < len(lines_txt): prod_name = lines_txt[i+1].strip()

    # B. Harga PCS (Anchor: PILIH SATUAN JUAL)
    if "PILIH SATUAN JUAL" in full_text_single:
        after_unit = full_text_single.split("PILIH SATUAN JUAL")[1]
        unit_match = re.search(r"(PCS|RCG|BOX)(.*?)/", after_unit)
        if unit_match:
            p = re.findall(r"RP\s*([\d\-\.,%]+)", unit_match.group(2))
            if len(p) >= 2: res["PCS"]["n"], res["PCS"]["p"] = clean_price_val(p[0]), clean_price_val(p[1])
            elif len(p) == 1: res["PCS"]["n"] = res["PCS"]["p"] = clean_price_val(p[0])

    # C. Harga CTN
    if "CTN" in full_text_single:
        ctn_part = full_text_single.split("CTN")[1].split("/")[0]
        p_ctn = re.findall(r"RP\s*([\d\-\.,%]+)", ctn_part)
        if len(p_ctn) >= 2: res["CTN"]["n"], res["CTN"]["p"] = clean_price_val(p_ctn[0]), clean_price_val(p_ctn[1])
        elif len(p_ctn) == 1: res["CTN"]["n"] = res["CTN"]["p"] = clean_price_val(p_ctn[0])

    # D. Promosi (Anchor: MEKANISME PROMO)
    anchor_promo = "MAU LEBIH UNTUNG? CEK MEKANISME PROMO BERIKUT"
    if anchor_promo in full_text_single:
        p_part = full_text_single.split(anchor_promo)[1]
        m_promo = re.search(r"[\|I1l]\s*(.*?)\s*[\|I1l]?\s*RAP", p_part, re.DOTALL)
        if m_promo: promo_desc = re.sub(r'^[^A-Z0-9]+', '', m_promo.group(1).strip())

    return res["PCS"], res["CTN"], prod_name, raw_ocr_output, pil_image, promo_desc

# ================= UI STREAMLIT =================
def norm(val): return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("📸 Price Check V13.2 - Live Scan Monitor")

c1, c2, c3 = st.columns(3)
with c1: m_code = st.text_input("📍 MASTER CODE").upper()
with c2: date_inp = st.text_input("📅 TANGGAL").upper()
with c3: week_inp = st.text_input("🗓️ WEEK")

files = st.file_uploader("📂 UPLOAD SCREENSHOTS", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code and date_inp and week_inp:
    if os.path.exists(FILE_PATH):
        db_ig = pd.read_excel(FILE_PATH, sheet_name=SHEET_MASTER_IG)
        db_targets = {s: pd.read_excel(FILE_PATH, sheet_name=s) for s in SHEETS_TARGET}
        final_list = []
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                with st.container(border=True):
                    img_pil = Image.open(f)
                    pcs, ctn, name, raw_txt, red_img, p_desc = process_ocr_strict(img_pil)
                    
                    # Match Product
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_n = str(row[COL_IG_NAME]).upper()
                        score = fuzz.partial_ratio(db_n, name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])
                    
                    st.subheader(f"📄 Scan Result: {f.name}")
                    
                    # --- TAMPILAN PREVIEW HASIL SCAN ---
                    col_view, col_raw = st.columns([1, 1])
                    
                    with col_view:
                        st.image(red_img, caption="Redacted Image Preview")
                        st.markdown(f"**Detected Name:** `{name}`")
                        st.markdown(f"**Prodcode:** `{match_code}` (Score: {best_score})")
                    
                    with col_raw:
                        st.caption("🔍 Full Raw OCR Text (Structured Lines):")
                        st.code(raw_txt, language="text") # Menampilkan semua hasil scan ocr
                    
                    # --- METRIK DATA ---
                    m_pcs, m_ctn, m_promo = st.columns([1, 1, 2])
                    m_pcs.metric("PCS (N / P)", f"{pcs['n']:,} / {pcs['p']:,}")
                    m_ctn.metric("CTN (N / P)", f"{ctn['n']:,} / {ctn['p']:,}")
                    m_promo.info(f"**Promo:** {p_desc}")

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            match_row = df_t[(df_t["PRODCODE"].astype(str).apply(norm) == match_code) & 
                                             (df_t["MASTER Code"].astype(str).apply(norm) == norm(m_code))]
                            if not match_row.empty:
                                final_list.append({
                                    "prodcode": match_code, "sheet": s_name, "index": match_row.index[0],
                                    "n_pcs": pcs['n'], "p_pcs": pcs['p'],
                                    "n_ctn": ctn['n'], "p_ctn": ctn['p'], "p_desc": p_desc
                                })
                                buf = io.BytesIO()
                                red_img.convert("RGB").save(buf, format="JPEG")
                                zf.writestr(f"{match_code}.jpg", buf.getvalue())
                                break
                gc.collect()

        if final_list:
            st.divider()
            if st.button("🚀 UPDATE DATABASE"):
                wb = load_workbook(FILE_PATH)
                for r in final_list:
                    ws = wb[r['sheet']]
                    headers = [str(c.value).strip() for c in ws[1]]
                    row_num = r['index'] + 2
                    mapping = {
                        "Normal Competitor Price (Pcs)": r['n_pcs'],
                        "Promo Competitor Price (Pcs)": r['p_pcs'],
                        "Normal Competitor Price (Ctn)": r['n_ctn'],
                        "Promo Competitor Price (Ctn)": r['p_ctn'],
                        "Promosi Competitor": r['p_desc']
                    }
                    for col_name, val in mapping.items():
                        if col_name in headers: ws.cell(row=row_num, column=headers.index(col_name) + 1).value = val
                wb.save(FILE_PATH)
                st.success("DATABASE UPDATED!")
                with open(FILE_PATH, "rb") as f: st.download_button("📥 DOWNLOAD REPORT", f, f"Report_{date_inp}.xlsx")
            st.download_button("🖼️ DOWNLOAD ZIP", zip_buffer.getvalue(), f"Photos_{m_code}.zip")

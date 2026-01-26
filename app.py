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

def clean_price_val(raw_str):
    """Membersihkan string harga menjadi integer murni."""
    if not raw_str: return 0
    clean = re.sub(r'[^\d]', '', str(raw_str))
    return int(clean) if clean else 0

def process_ocr_final(pil_image):
    # 1. Image Preprocessing
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. OCR Execution
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
    df_ocr = pd.DataFrame(d)
    df_ocr = df_ocr[df_ocr['text'].str.strip() != ""]
    df_ocr['text'] = df_ocr['text'].str.upper()

    # 3. Line Grouping
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

    # --- A. NAMA PRODUK & SENSOR ---
    anchor_nav = "SEMUA KATEGORI ~ CARI DI KLIK INDOGROSIR"
    for i, line in enumerate(lines_txt):
        if fuzz.partial_ratio(anchor_nav, line) > 75:
            y, h = lines_data[i]['top'] / scale, lines_data[i]['h'] / scale
            draw.rectangle([0, y - 5, pil_image.width, y + h + 5], fill="white")
            
            p_name_parts = []
            for j in range(i + 1, min(i + 3, len(lines_txt))):
                if any(k in lines_txt[j] for k in ["PILIH", "SATUAN", "POTONGAN"]): break
                p_name_parts.append(lines_txt[j])
            prod_name = " ".join(p_name_parts).strip()
            break

    # --- B. HARGA PCS ---
    pcs_pattern = r"PILIH\s*\d*\s*SATUAN\s*JUAL"
    if re.search(pcs_pattern, full_text_single):
        after_unit = re.split(pcs_pattern, full_text_single)[1]
        m = re.search(r"(PCS|RCG|BOX)(.*?)[\/|ISI|BS]", after_unit)
        if m:
            p = re.findall(r"RP\s*([\d\-\.,%]+)", m.group(2))
            if len(p) >= 2: res["PCS"]["n"], res["PCS"]["p"] = clean_price_val(p[0]), clean_price_val(p[1])
            elif len(p) == 1: res["PCS"]["n"] = res["PCS"]["p"] = clean_price_val(p[0])

    # --- C. HARGA CTN ---
    if "CTN" in full_text_single:
        after_ctn = full_text_single.split("CTN")[1]
        ctn_m = re.search(r"(.*?)[\/|ISI]", after_ctn)
        if ctn_m:
            p_ctn = re.findall(r"RP\s*([\d\-\.,%]+)", ctn_m.group(1))
            if len(p_ctn) >= 2: res["CTN"]["n"], res["CTN"]["p"] = clean_price_val(p_ctn[0]), clean_price_val(p_ctn[1])
            elif len(p_ctn) == 1: res["CTN"]["n"] = res["CTN"]["p"] = clean_price_val(p_ctn[0])

    # --- D. PROMOSI ---
    anchor_promo = "MAU LEBIH UNTUNG? CEK MEKANISME PROMO BERIKUT"
    for i, line in enumerate(lines_txt):
        if anchor_promo in line:
            promo_lines = []
            for j in range(i + 1, min(i + 3, len(lines_txt))):
                promo_lines.append(lines_txt[j])
            full_promo_txt = " ".join(promo_lines)
            promo_split = full_promo_txt.split("=")[0].strip()
            promo_clean = re.sub(r'\bRAP\b', '', promo_split)
            promo_clean = promo_clean.replace("|", "").strip()
            promo_desc = re.sub(r'^[^A-Z0-9]+', '', promo_clean)
            break

    return res["PCS"], res["CTN"], prod_name, raw_ocr_output, pil_image, promo_desc

# ================= UI STREAMLIT =================
def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("📸 Price Check")

col_a, col_b, col_c = st.columns(3)
with col_a: m_code = st.text_input("📍 MASTER CODE").upper()
with col_b: date_inp = st.text_input("📅 DAY (Contoh: 01JAN2026)").upper()
with col_c: week_inp = st.text_input("🗓️ WEEK (Contoh: 1)")

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
                    pcs, ctn, name, raw_txt, red_img, p_desc = process_ocr_final(img_pil)
                    
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_name = str(row[COL_IG_NAME]).upper()
                        score = fuzz.partial_ratio(db_name, name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])
                    
                    st.markdown(f"### 📄 {f.name}")
                    c1, c2 = st.columns([2, 1])
                    with c1: st.markdown(f"**OCR Name:** `{name}`")
                    with c2: 
                        if match_code: st.info(f"**Matched Code:** `{match_code}`")
                        else: st.warning("⚠️ Code Not Found")

                    m1, m2, m3 = st.columns([1, 1, 2])
                    m1.metric("UNIT (Normal/Promo)", f"{pcs['n']:,} / {pcs['p']:,}")
                    m2.metric("CTN (Normal/Promo)", f"{ctn['n']:,} / {ctn['p']:,}")
                    m3.success(f"**Promosi:** {p_desc}")

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
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🚀 UPDATE DATABASE", use_container_width=True):
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
                            if col_name in headers:
                                ws.cell(row=row_num, column=headers.index(col_name) + 1).value = val
                    wb.save(FILE_PATH)
                    st.success("✅ DATABASE UPDATED!")
                    
                    # NAMA FILE EXCEL SESUAI PERMINTAAN
                    excel_filename = f"Price Check W{week_inp}_{date_inp}.xlsx"
                    with open(FILE_PATH, "rb") as f:
                        st.download_button("📥 DOWNLOAD EXCEL", f, excel_filename, use_container_width=True)
            
            with col_btn2:
                # NAMA FILE ZIP SESUAI PERMINTAAN
                zip_filename = f"{m_code}.zip"
                st.download_button("🖼️ DOWNLOAD FOTO", zip_buffer.getvalue(), zip_filename, use_container_width=True)
    else:
        st.error(f"Database Excel tidak ditemukan di: {FILE_PATH}")

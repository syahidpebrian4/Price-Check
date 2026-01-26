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

# ================= CONFIG =================
FILE_PATH = "database/master_harga.xlsx"
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 

st.set_page_config(page_title="Price Check V13.5", layout="wide")

def clean_price_val(raw_str):
    if not raw_str: return 0
    clean = re.sub(r'[^\d]', '', str(raw_str))
    return int(clean) if clean else 0

def process_ocr_v13_5(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
    df_ocr = pd.DataFrame(d)
    df_ocr = df_ocr[df_ocr['text'].str.strip() != ""]
    df_ocr['text'] = df_ocr['text'].str.upper()

    # Line Grouping Logic
    df_ocr = df_ocr.sort_values(by=['top', 'left'])
    lines_data = []
    if not df_ocr.empty:
        current_top = df_ocr.iloc[0]['top']
        temp_words = []
        for _, row in df_ocr.iterrows():
            if row['top'] - current_top > 15:
                temp_words.sort(key=lambda x: x['left'])
                lines_data.append({"text": " ".join([w['text'] for w in temp_words]), "top": current_top, "h": max([w['height'] for w in temp_words])})
                temp_words = [{'text': row['text'], 'left': row['left'], 'height': row['height']}]
                current_top = row['top']
            else:
                temp_words.append({'text': row['text'], 'left': row['left'], 'height': row['height']})
        temp_words.sort(key=lambda x: x['left'])
        lines_data.append({"text": " ".join([w['text'] for w in temp_words]), "top": current_top, "h": 10})

    lines_txt = [l['text'] for l in lines_data]
    full_text_single = " ".join(lines_txt)
    
    prod_name, promo_desc = "N/A", "-"
    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}
    draw = ImageDraw.Draw(pil_image)

    # 1. LOGIKA NAMA PRODUK & SENSOR (INSTRUKSI BARU)
    anchor_header = "SEMUA KATEGORI ~ CARI DI KLIK INDOGROSIR"
    header_idx = -1
    
    for i, line in enumerate(lines_txt):
        if fuzz.partial_ratio(anchor_header, line) > 75:
            header_idx = i
            # Sensor baris header tersebut (Halo, Wayan, dsb)
            y, h = lines_data[i]['top'] / scale, lines_data[i]['h'] / scale
            draw.rectangle([0, y - 5, pil_image.width, y + h + 5], fill="white")
            break

    if header_idx != -1:
        # Cari nama produk di 3 baris setelah header
        potential_lines = []
        for j in range(header_idx + 1, min(header_idx + 4, len(lines_txt))):
            curr_line = lines_txt[j]
            # Berhenti jika bertemu kata kunci navigasi harga
            if any(k in curr_line for k in ["PILIH SATUAN", "POTONGAN", "HARGA TERMURAH"]):
                break
            potential_lines.append(curr_line)
        
        if potential_lines:
            prod_name = " ".join(potential_lines).strip()

    # 2. HARGA PCS (Anchor: PILIH SATUAN JUAL)
    if "PILIH SATUAN JUAL" in full_text_single:
        after_unit = full_text_single.split("PILIH SATUAN JUAL")[1]
        m = re.search(r"(PCS|RCG|BOX)(.*?)/", after_unit)
        if m:
            p = re.findall(r"RP\s*([\d\-\.,%]+)", m.group(2))
            if len(p) >= 2: res["PCS"]["n"], res["PCS"]["p"] = clean_price_val(p[0]), clean_price_val(p[1])
            elif len(p) == 1: res["PCS"]["n"] = res["PCS"]["p"] = clean_price_val(p[0])

    # 3. HARGA CTN (Anchor: CTN)
    if "CTN" in full_text_single:
        try:
            ctn_p = full_text_single.split("CTN")[1].split("/")[0]
            p_c = re.findall(r"RP\s*([\d\-\.,%]+)", ctn_p)
            if len(p_c) >= 2: res["CTN"]["n"], res["CTN"]["p"] = clean_price_val(p_c[0]), clean_price_val(p_c[1])
            elif len(p_c) == 1: res["CTN"]["n"] = res["CTN"]["p"] = clean_price_val(p_c[0])
        except: pass

    # 4. PROMOSI (Anchor: CEK MEKANISME PROMO BERIKUT)
    if "CEK MEKANISME PROMO BERIKUT" in full_text_single:
        p_part = full_text_single.split("CEK MEKANISME PROMO BERIKUT")[1]
        # Regex mencari teks di antara pipa pertama dan kata RAP
        m_p = re.search(r"[\|I1l]\s*(.*?)\s*[\|I1l]?\s*RAP", p_part, re.DOTALL)
        if m_p:
            promo_desc = m_p.group(1).strip()
            promo_desc = re.sub(r'^[^A-Z0-9]+', '', promo_desc)

    return res["PCS"], res["CTN"], prod_name, "\n".join(lines_txt), pil_image, promo_desc

# ================= UI STREAMLIT =================
def norm(val): return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("📸 Price Check V13.5 - Direct Anchor Logic")

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
                    pcs, ctn, name, raw_txt, red_img, p_desc = process_ocr_v13_5(img_pil)
                    
                    # Match Product
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_n = str(row[COL_IG_NAME]).upper()
                        score = fuzz.partial_ratio(db_n, name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])
                    
                    st.subheader(f"📄 Scan: {f.name}")
                    col_img, col_txt = st.columns([1, 1])
                    with col_img: st.image(red_img)
                    with col_txt:
                        st.markdown(f"**Nama Produk:** `{name}`")
                        st.markdown(f"**Prodcode:** `{match_code}`")
                        st.info(f"**Promosi:** {p_desc}")
                        st.caption("Raw OCR Results:")
                        st.code(raw_txt)

                    m1, m2 = st.columns(2)
                    m1.metric("PCS Normal / Promo", f"Rp {pcs['n']:,} / {pcs['p']:,}")
                    m2.metric("CTN Normal / Promo", f"Rp {ctn['n']:,} / {ctn['p']:,}")

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
            if st.button("🚀 UPDATE DATABASE EXCEL"):
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
                st.success("DATA UPDATED!")

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

st.set_page_config(page_title="Price Check V14.2 - Final", layout="wide")

def clean_price_val(raw_str):
    """Membersihkan string harga dan mengambil angka murni."""
    if not raw_str: return 0
    # Hanya ambil digit
    clean = re.sub(r'[^\d]', '', str(raw_str))
    return int(clean) if clean else 0

def process_ocr_v14_2(pil_image):
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

    # --- A. NAMA PRODUK (Anchor: SEMUA KATEGORI) ---
    anchor_nav = "SEMUA KATEGORI ~ CARI DI KLIK INDOGROSIR"
    for i, line in enumerate(lines_txt):
        if fuzz.partial_ratio(anchor_nav, line) > 75:
            y, h = lines_data[i]['top'] / scale, lines_data[i]['h'] / scale
            draw.rectangle([0, y - 5, pil_image.width, y + h + 5], fill="white")
            p_parts = []
            for j in range(i + 1, min(i + 3, len(lines_txt))):
                if any(k in lines_txt[j] for k in ["PILIH", "SATUAN", "POTONGAN"]): break
                p_parts.append(lines_txt[j])
            prod_name = " ".join(p_parts).strip()
            break

    # --- B. HARGA PCS (PILIH [X] SATUAN JUAL) ---
    pcs_pattern = r"PILIH\s*\d*\s*SATUAN\s*JUAL"
    if re.search(pcs_pattern, full_text_single):
        after_unit = re.split(pcs_pattern, full_text_single)[1]
        m = re.search(r"(PCS|RCG|BOX)(.*?)[\/|ISI|\)]", after_unit)
        if m:
            area_harga = m.group(2)
            # Bersihkan huruf P/B yang sering muncul di antara angka
            area_clean = re.sub(r'(?<=\d)[P|B|R](?=\d)', '0', area_area) 
            # Cari kelompok angka minimal 4 digit (format harga)
            prices = re.findall(r'(\d[\d\.\,]{3,})', area_harga)
            if len(prices) >= 2:
                res["PCS"]["n"], res["PCS"]["p"] = clean_price_val(prices[-2]), clean_price_val(prices[-1])
            elif len(prices) == 1:
                res["PCS"]["n"] = res["PCS"]["p"] = clean_price_val(prices[0])

    # --- C. HARGA CTN (Handling kasus 2P477.689) ---
    if "CTN" in full_text_single:
        after_ctn = full_text_single.split("CTN")[1]
        ctn_m = re.search(r"(.*?)[\/|ISI|\)]", after_ctn)
        if ctn_m:
            area_ctn = ctn_m.group(1)
            # Ubah 'P' di tengah angka menjadi '0' (Kasus: 2P477 -> 20477)
            area_ctn = re.sub(r'(?<=\d)P(?=\d)', '0', area_ctn)
            # Cari semua kelompok angka (Ribuan)
            prices_ctn = re.findall(r'(\d[\d\.\,]{3,})', area_ctn)
            
            # Ambil 2 angka TERAKHIR (Harga Normal & Promo biasanya paling kanan)
            if len(prices_ctn) >= 2:
                res["CTN"]["n"] = clean_price_val(prices_ctn[-2])
                res["CTN"]["p"] = clean_price_val(prices_ctn[-1])
            elif len(prices_ctn) == 1:
                res["CTN"]["n"] = res["CTN"]["p"] = clean_price_val(prices_ctn[0])

    # --- D. PROMOSI ---
    anchor_promo = "MAU LEBIH UNTUNG? CEK MEKANISME PROMO BERIKUT"
    for i, line in enumerate(lines_txt):
        if anchor_promo in line:
            p_lines = [lines_txt[j] for j in range(i + 1, min(i + 3, len(lines_txt)))]
            promo_final = " ".join(p_lines).split("=")[0].strip()
            promo_clean = re.sub(r'\bRAP\b', '', promo_final).replace("|", "").strip()
            promo_desc = re.sub(r'^[^A-Z0-9]+', '', promo_clean)
            break

    return res["PCS"], res["CTN"], prod_name, raw_ocr_output, pil_image, promo_desc

# ================= UI STREAMLIT =================
def norm(val): return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("📸 Price Check V14.2 - Advanced Price Recovery")

c1, c2, c3 = st.columns(3)
with c1: m_code = st.text_input("📍 MASTER CODE").upper()
with c2: date_inp = st.text_input("📅 TANGGAL SCAN").upper()
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
                    pcs, ctn, name, raw_txt, red_img, p_desc = process_ocr_v14_2(img_pil)
                    
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        score = fuzz.partial_ratio(str(row[COL_IG_NAME]).upper(), name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])
                    
                    st.subheader(f"📄 File: {f.name}")
                    col_v, col_r = st.columns([1, 1])
                    with col_v:
                        st.image(red_img, use_container_width=True)
                        st.write(f"**Nama:** {name} | **Code:** {match_code}")
                    with col_r:
                        st.caption("Raw OCR Monitor:")
                        st.code(raw_txt)

                    m1, m2, m3 = st.columns([1, 1, 2])
                    m1.metric("UNIT", f"{pcs['n']:,} / {pcs['p']:,}")
                    m2.metric("CTN", f"{ctn['n']:,} / {ctn['p']:,}")
                    m3.success(f"Promo: {p_desc}")

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
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
            if st.button("🚀 EXECUTE UPDATE EXCEL"):
                wb = load_workbook(FILE_PATH)
                for r in final_list:
                    ws = wb[r['sheet']]
                    headers = [str(c.value).strip() for c in ws[1]]
                    row_num = r['index'] + 2
                    mapping = {
                        "Normal Competitor Price (Pcs)": r['n_pcs'], "Promo Competitor Price (Pcs)": r['p_pcs'],
                        "Normal Competitor Price (Ctn)": r['n_ctn'], "Promo Competitor Price (Ctn)": r['p_ctn'],
                        "Promosi Competitor": r['p_desc']
                    }
                    for col_n, val in mapping.items():
                        if col_n in headers: ws.cell(row=row_num, column=headers.index(col_n)+1).value = val
                wb.save(FILE_PATH)
                st.success("EXCEL UPDATED!")

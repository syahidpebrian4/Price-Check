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

# Daftar kata yang harus disensor (Hanya area profil/atas)
TEXTS_TO_REDACT = ["AL YUYUN SUMARNI", "HALO", "MEMBER UMUM", "088902864826", "02121584272", "CS.KLIK@INDOGROSIR.CO.ID"]

st.set_page_config(page_title="PRICE CHECK V12.9", layout="wide")

def clean_price_val(raw_str):
    """Membersihkan segala karakter non-angka"""
    if not raw_str: return 0
    clean = re.sub(r'[^\d]', '', str(raw_str))
    return int(clean) if clean else 0

def process_ocr_v12_9(pil_image):
    # 1. Preprocessing
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. OCR Data Extraction
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
    df = pd.DataFrame(d)
    df = df[df['text'].str.strip() != ""]
    df['text'] = df['text'].str.upper()

    # 3. Baris demi Baris (Line Grouping)
    df = df.sort_values(by=['top', 'left'])
    lines = []
    if not df.empty:
        current_top = df.iloc[0]['top']
        temp_line = []
        for _, row in df.iterrows():
            if row['top'] - current_top > 15:
                temp_line.sort(key=lambda x: x['left'])
                lines.append(" ".join([w['text'] for w in temp_line]))
                temp_line = [{'text': row['text'], 'left': row['left']}]
                current_top = row['top']
            else:
                temp_line.append({'text': row['text'], 'left': row['left']})
        temp_line.sort(key=lambda x: x['left'])
        lines.append(" ".join([w['text'] for w in temp_line]))

    full_text_clean = "\n".join(lines)
    single_line = " ".join(lines)

    # 4. Nama Produk (Mencari baris dengan pola kode dalam kurung: (39007))
    prod_name = "N/A"
    for line in lines:
        if re.search(r"\(\d{4,7}\)", line):
            prod_name = line.strip()
            break

    # 5. Ekstraksi Harga (Per Baris agar tidak tertukar)
    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}
    for line in lines:
        # Deteksi Baris Eceran
        if any(x in line for x in ["PCS", "RCG", "BOX"]) and "/ ISI" in line:
            prices = re.findall(r"RP\s*([\d\-\.,%]+)", line)
            if len(prices) >= 2:
                res["PCS"]["n"], res["PCS"]["p"] = clean_price_val(prices[0]), clean_price_val(prices[1])
            elif len(prices) == 1:
                res["PCS"]["n"] = res["PCS"]["p"] = clean_price_val(prices[0])
        
        # Deteksi Baris Karton
        if "CTN" in line and "/ ISI" in line:
            # Mencari angka setelah 'RP' atau angka murni jika 'RP' tidak terbaca sempurna
            prices = re.findall(r"(?:RP\s*)?([\d\-\.,%]+)", line)
            # Filter hanya yang panjang angkanya masuk akal untuk harga karton (>3 digit)
            valid_prices = [p for p in prices if len(re.sub(r'[^\d]', '', p)) > 3]
            if len(valid_prices) >= 2:
                res["CTN"]["n"], res["CTN"]["p"] = clean_price_val(valid_prices[0]), clean_price_val(valid_prices[1])
            elif len(valid_prices) == 1:
                res["CTN"]["n"] = res["CTN"]["p"] = clean_price_val(valid_prices[0])

    # 6. Logika Promosi (Keyword -> | -> RAP)
    promo_txt = "-"
    keyword = "MAU LEBIH UNTUNG? CEK MEKANISME PROMO BERIKUT"
    if keyword in single_line:
        try:
            part = single_line.split(keyword)[1]
            # Mencari teks di antara pipa pertama dan kata RAP
            m = re.search(r"[\|I1l]\s*(.*?)\s*[\|I1l]?\s*RAP", part, re.DOTALL)
            if m:
                promo_txt = m.group(1).strip()
                promo_txt = re.sub(r'^[^A-Z0-9]+', '', promo_txt)
        except:
            pass

    # 7. Redact (Hanya sensor 25% area atas gambar)
    draw = ImageDraw.Draw(pil_image)
    limit_y = (img_resized.shape[0] * 0.25) # Batas sensor hanya di 25% atas
    for _, row in df.iterrows():
        if row['top'] < limit_y:
            for kw in TEXTS_TO_REDACT:
                if kw in row['text'] or fuzz.ratio(kw, row['text']) > 80:
                    x, y, w, h = row['left']/scale, row['top']/scale, row['width']/scale, row['height']/scale
                    draw.rectangle([x-2, y-2, x+w+2, y+h+2], fill="white")

    return res["PCS"], res["CTN"], prod_name, full_text_clean, pil_image, promo_txt

# ================= UI STREAMLIT =================
def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("PRICE CHECK - V12.9 STABLE")

c1, c2, c3 = st.columns(3)
with c1: m_code = st.text_input("📍 MASTER CODE").upper()
with c2: date_inp = st.text_input("📅 TANGGAL").upper()
with c3: week_inp = st.text_input("🗓️ WEEK")

files = st.file_uploader("📂 UPLOAD FOTO", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

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
                    pcs, ctn, name, raw_text, red_img, p_desc = process_ocr_v12_9(img_pil)
                    
                    # Fuzzy match untuk prodcode
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_n = str(row[COL_IG_NAME]).upper()
                        score = fuzz.partial_ratio(db_n, name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])
                    
                    st.subheader(f"🔍 {f.name}")
                    col_img, col_info = st.columns([1, 1.2])
                    with col_img: st.image(red_img)
                    with col_info:
                        st.markdown(f"### {name}")
                        st.write(f"Match Code: `{match_code}`")
                        st.info(f"**PROMOSI:** {p_desc}")
                        m1, m2 = st.columns(2)
                        m1.metric("UNIT Normal", f"Rp {pcs['n']:,}")
                        m1.metric("UNIT Promo", f"Rp {pcs['p']:,}")
                        m2.metric("CTN Normal", f"Rp {ctn['n']:,}")
                        m2.metric("CTN Promo", f"Rp {ctn['p']:,}")

                    with st.expander("📄 RAW TEXT"):
                        st.text(raw_text)

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
                        if col_name in headers:
                            ws.cell(row=row_num, column=headers.index(col_name) + 1).value = val
                wb.save(FILE_PATH)
                st.success("DATABASE BERHASIL DIPERBARUI!")
                with open(FILE_PATH, "rb") as f:
                    st.download_button("📥 DOWNLOAD REPORT", f, f"Report_{date_inp}.xlsx")
            st.download_button("🖼️ DOWNLOAD ZIP FOTO", zip_buffer.getvalue(), f"Photos_{m_code}.zip")

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

# Redaksi data sensitif
TEXTS_TO_REDACT = [
    "HALO, AL YUYUN SUMARNI", "AL YUYUN SUMARNI", "MEMBER UMUM KLIK", 
    "088902864826", "02121584272", "CS.KLIK@INDOGROSIR.CO.ID"
]

st.set_page_config(page_title="PRICE CHECK V12.8", layout="wide")

def clean_price_val(raw_str):
    """Membuang semua karakter kecuali angka agar kebal terhadap simbol sampah OCR"""
    if not raw_str: return 0
    clean = re.sub(r'[^\d]', '', raw_str)
    try:
        return int(clean)
    except:
        return 0

def process_ocr_v12_8(pil_image):
    # 1. Preprocessing
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.5 # Menaikkan skala sedikit untuk membantu pembacaan teks kecil
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. OCR Execution
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
    df = pd.DataFrame(d)
    df = df[df['text'].str.strip() != ""]
    df['text'] = df['text'].str.upper()

    # 3. Spatial Grouping (Top-Down & Left-Right Sorting)
    df = df.sort_values(by=['top', 'left'])
    lines = []
    if not df.empty:
        current_line_top = df.iloc[0]['top']
        temp_words = []
        for _, row in df.iterrows():
            if row['top'] - current_line_top > 20: # Toleransi baris
                temp_words.sort(key=lambda x: x['left'])
                lines.append(" ".join([w['text'] for w in temp_words]))
                temp_words = [{'text': row['text'], 'left': row['left']}]
                current_line_top = row['top']
            else:
                temp_words.append({'text': row['text'], 'left': row['left']})
        temp_words.sort(key=lambda x: x['left'])
        lines.append(" ".join([w['text'] for w in temp_words]))
    
    clean_full_text = "\n".join(lines)
    single_line_text = " ".join(lines)

    # 4. Deteksi Nama Produk Secara Dinamis
    # Mencari teks sebelum kata 'POTONGAN' atau setelah navigasi kategori
    prod_name = "N/A"
    # Pola: ambil baris yang mengandung (nomor kode) dan nama sebelum label 'POTONGAN'
    name_search = re.search(r"(.*?)\n\s*POTONGAN", clean_full_text, re.DOTALL)
    if name_search:
        # Mengambil baris terakhir dari hasil capture sebelum kata POTONGAN
        lines_before = name_search.group(1).strip().split('\n')
        prod_name = lines_before[-1].strip()
    
    # 5. Ekstraksi Harga (PCS & CTN)
    final_res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    
    # Pattern PCS/RCG/BOX
    unit_m = re.search(r"(PCS|RCG|BOX).*?RP\s*([\d\-\.,%]+).*?RP\s*([\d\-\.,%]+).*?/\s*ISI", single_line_text)
    if unit_m:
        final_res["PCS"]["normal"] = clean_price_val(unit_m.group(2))
        final_res["PCS"]["promo"] = clean_price_val(unit_m.group(3))

    # Pattern CTN (Ditingkatkan untuk menangkap angka di tengah simbol sampah seperti RPT%.000)
    ctn_m = re.search(r"CTN.*?RP?\s*([\d\-\.,%]+).*?RP\s*([\d\-\.,%]+).*?/\s*ISI", single_line_text)
    if ctn_m:
        final_res["CTN"]["normal"] = clean_price_val(ctn_m.group(1))
        final_res["CTN"]["promo"] = clean_price_val(ctn_m.group(2))

    # 6. Logika Promosi Khusus (Antara Keyword dan RAP)
    promo_desc = "-"
    keyword = "MAU LEBIH UNTUNG? CEK MEKANISME PROMO BERIKUT"
    if keyword in single_line_text:
        after_keyword = single_line_text.split(keyword)[-1]
        # Mengambil teks di antara tanda pipa pertama dan tanda pipa sebelum RAP
        promo_match = re.search(r"[\|I1l]\s*(.*?)\s*[\|I1l]\s*RAP", after_keyword, re.DOTALL)
        if promo_match:
            promo_desc = promo_match.group(1).strip()
            promo_desc = re.sub(r'^[^A-Z0-9]+', '', promo_desc) # Bersihkan sampah awal
        else:
            # Fallback jika pipa tidak lengkap
            promo_match_alt = re.search(r"[\|I1l]\s*(.*?RAP)", after_keyword, re.DOTALL)
            if promo_match_alt:
                promo_desc = promo_match_alt.group(1).replace("RAP", "").strip()

    # 7. Redaksi / Sensor
    draw = ImageDraw.Draw(pil_image)
    for _, row in df.iterrows():
        for kw in TEXTS_TO_REDACT:
            if fuzz.partial_ratio(kw, row['text']) > 80:
                x, y, w, h = row['left']/scale, row['top']/scale, row['width']/scale, row['height']/scale
                draw.rectangle([x-2, y-2, x+w+2, y+h+2], fill="white")
    
    return final_res["PCS"], final_res["CTN"], prod_name, clean_full_text, pil_image, promo_desc

# ================= UI STREAMLIT =================
def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("PRICE CHECK - V12.8")

c1, c2, c3 = st.columns(3)
with c1: m_code = st.text_input("📍 MASTER CODE").upper()
with c2: date_inp = st.text_input("📅 TANGGAL (Ex: 26 JAN)").upper()
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
                    pcs, ctn, name, raw_text, red_img, p_desc = process_ocr_v12_8(img_pil)
                    
                    # Fuzzy match untuk prodcode (karena nama produk sekarang sangat panjang)
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_n = str(row[COL_IG_NAME]).upper()
                        score = fuzz.partial_ratio(db_n, name)
                        if score > 75 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])
                    
                    st.subheader(f"📄 File: {f.name}")
                    col_img, col_info = st.columns([1, 1.2])
                    with col_img: st.image(red_img)
                    with col_info:
                        st.markdown(f"### {name}")
                        st.write(f"Match Prodcode: `{match_code}` (Score: {best_score})")
                        st.info(f"**PROMOSI:** {p_desc}")
                        m1, m2 = st.columns(2)
                        m1.metric("UNIT Normal", f"Rp {pcs['normal']:,}")
                        m1.metric("UNIT Promo", f"Rp {pcs['promo']:,}")
                        m2.metric("CTN Normal", f"Rp {ctn['normal']:,}")
                        m2.metric("CTN Promo", f"Rp {ctn['promo']:,}")

                    with st.expander("📄 LIHAT DATA MENTAH SCAN (RE-STRUCTURED)"):
                        st.text(raw_text)

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            match_row = df_t[(df_t["PRODCODE"].astype(str).apply(norm) == match_code) & 
                                             (df_t["MASTER Code"].astype(str).apply(norm) == norm(m_code))]
                            
                            if not match_row.empty:
                                final_list.append({
                                    "prodcode": match_code, "sheet": s_name, "index": match_row.index[0],
                                    "n_pcs": pcs['normal'], "p_pcs": pcs['promo'],
                                    "n_ctn": ctn['normal'], "p_ctn": ctn['promo'],
                                    "p_desc": p_desc
                                })
                                buf = io.BytesIO()
                                red_img.convert("RGB").save(buf, format="JPEG")
                                zf.writestr(f"{match_code}.jpg", buf.getvalue())
                                break
                gc.collect()

        if final_list:
            st.divider()
            if st.button("🚀 EKSEKUSI UPDATE KE EXCEL"):
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
                    st.download_button("📥 DOWNLOAD REPORT", f, f"Update_Report_{date_inp}.xlsx")
            st.download_button("🖼️ DOWNLOAD ZIP FOTO", zip_buffer.getvalue(), f"Photos_{m_code}.zip")

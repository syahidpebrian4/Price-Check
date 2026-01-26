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

st.set_page_config(page_title="Price Check V14.7 - Ultra Precision", layout="wide")

def clean_price_val(raw_str):
    """Menghapus semua karakter non-digit dan mengubah ke integer."""
    if not raw_str: return 0
    clean = re.sub(r'[^\d]', '', str(raw_str))
    return int(clean) if clean else 0

def process_ocr_v14_7(pil_image):
    # 1. Image Preprocessing (Skala 2x & Grayscale)
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. OCR Execution
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
    df_ocr = pd.DataFrame(d)
    df_ocr = df_ocr[df_ocr['text'].str.strip() != ""]
    df_ocr['text'] = df_ocr['text'].str.upper()

    # 3. Line Grouping (Menyusun kata menjadi baris kalimat)
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
    raw_ocr_output = "\n".join(lines_txt)
    
    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}
    prod_name, promo_desc = "N/A", "-"

    # --- HELPER: LOGIKA EKSTRAKSI ANGKA HARGA ---
    def get_clean_prices(text):
        # Abaikan bagian setelah pembatas (ISI, /, atau kurung) agar tidak ambil angka satuan
        main_text = re.split(r'[\/|ISI|\(]', text)[0]
        # Ubah karakter typo OCR (P=0, E=8, S=5, I=1)
        c = re.sub(r'[A-DF-HJK-LN-OQ-RT-Z]', ' ', main_text)
        c = c.replace('P', '0').replace('E', '8').replace('S', '5').replace('I', '1').replace('B', '8')
        # Ambil semua angka minimal 2 digit
        found = re.findall(r'(\d[\d\.\,]{1,})', c)
        return [clean_price_val(x) for x in found if clean_price_val(x) > 50]

    # --- SCANNER BARIS ---
    for i, line in enumerate(lines_txt):
        # A. Deteksi Nama Produk (Header Navigation)
        if "SEMUA KATEGORI" in line or "CARI DI KLIK" in line:
            p_parts = []
            for j in range(i + 1, i + 4):
                if j < len(lines_txt) and not any(k in lines_txt[j] for k in ["PILIH", "SATUAN", "HARGA"]):
                    p_parts.append(lines_txt[j])
            prod_name = " ".join(p_parts).strip()

        # B. Deteksi Baris PCS/BOX/RCG
        if any(unit in line for unit in ["PCS", "BOX", "RCG", "UNIT"]):
            prices = get_clean_prices(line)
            if len(prices) >= 2:
                res["PCS"]["n"], res["PCS"]["p"] = prices[-2], prices[-1]
            elif len(prices) == 1:
                res["PCS"]["n"] = res["PCS"]["p"] = prices[0]

        # C. Deteksi Baris CTN
        if "CTN" in line:
            prices = get_clean_prices(line)
            if len(prices) >= 2:
                res["CTN"]["n"], res["CTN"]["p"] = prices[-2], prices[-1]
            elif len(prices) == 1:
                res["CTN"]["n"] = res["CTN"]["p"] = prices[0]

    # --- D. EKSTRAKSI PROMO ---
    for i, line in enumerate(lines_txt):
        if any(x in line for x in ["MEKANISME", "UNTUNG?", "PROMO"]):
            p_lines = [lines_txt[j] for j in range(i + 1, min(i + 4, len(lines_txt)))]
            promo_txt = " ".join(p_lines).split("=")[0].strip()
            promo_desc = re.sub(r'^[^A-Z0-9]+', '', promo_txt).replace("|", "").strip()
            break

    return res["PCS"], res["CTN"], prod_name, raw_ocr_output, pil_image, promo_desc

# ================= UI STREAMLIT =================
st.title("📸 Price Check V14.7 - Ultra Precision")

c1, c2, c3 = st.columns(3)
with c1: m_code = st.text_input("📍 MASTER CODE").upper()
with c2: date_inp = st.text_input("📅 TANGGAL (DD-MM-YYYY)").upper()
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
                    pcs, ctn, name, raw_txt, red_img, p_desc = process_ocr_v14_7(img_pil)
                    
                    # Fuzzy Name Matching
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        score = fuzz.partial_ratio(str(row[COL_IG_NAME]).upper(), name)
                        if score > 60 and score > best_score:
                            best_score, match_code = score, str(row["PRODCODE"]).replace(".0","")
                    
                    st.subheader(f"📄 Scan: {f.name}")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(img_pil, use_container_width=True)
                    with col2:
                        st.write(f"**Nama Produk:** {name}")
                        st.write(f"**Prodcode Match:** {match_code} (Score: {best_score})")
                        
                        m1, m2 = st.columns(2)
                        m1.metric("PCS/UNIT (N/P)", f"Rp {pcs['n']:,} / {pcs['p']:,}")
                        m2.metric("CTN (N/P)", f"Rp {ctn['n']:,} / {ctn['p']:,}")
                        st.info(f"**Promo:** {p_desc}")
                        
                        with st.expander("Lihat Raw OCR"):
                            st.code(raw_txt)

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            p_c_n = str(match_code).strip().upper()
                            m_c_n = str(m_code).strip().upper()
                            
                            match_row = df_t[
                                (df_t["PRODCODE"].astype(str).str.replace(".0","").str.strip() == p_c_n) & 
                                (df_t["MASTER Code"].astype(str).str.strip().upper() == m_c_n)
                            ]
                            
                            if not match_row.empty:
                                final_list.append({
                                    "prodcode": p_c_n, "sheet": s_name, "index": match_row.index[0],
                                    "n_pcs": pcs['n'], "p_pcs": pcs['p'], "n_ctn": ctn['n'], "p_ctn": ctn['p'], "p_desc": p_desc
                                })
                                # Simpan foto ke ZIP
                                buf = io.BytesIO()
                                img_pil.save(buf, format="JPEG")
                                zf.writestr(f"{p_c_n}.jpg", buf.getvalue())
                                break
                gc.collect()

        if final_list:
            st.divider()
            if st.button("🚀 EKSEKUSI UPDATE DATABASE"):
                wb = load_workbook(FILE_PATH)
                for r in final_list:
                    ws = wb[r['sheet']]
                    headers = [str(c.value).strip() for c in ws[1]]
                    row_idx = r['index'] + 2
                    mapping = {
                        "Normal Competitor Price (Pcs)": r['n_pcs'], "Promo Competitor Price (Pcs)": r['p_pcs'],
                        "Normal Competitor Price (Ctn)": r['n_ctn'], "Promo Competitor Price (Ctn)": r['p_ctn'],
                        "Promosi Competitor": r['p_desc']
                    }
                    for col_n, val in mapping.items():
                        if col_n in headers:
                            ws.cell(row=row_idx, column=headers.index(col_n)+1).value = val
                wb.save(FILE_PATH)
                st.success("DATABASE EXCEL BERHASIL DIUPDATE!")
                
                with open(FILE_PATH, "rb") as f:
                    st.download_button("📥 DOWNLOAD REPORT", f, f"Report_{date_inp}.xlsx")
            st.download_button("🖼️ DOWNLOAD ZIP FOTO", zip_buffer.getvalue(), f"Photos_{m_code}.zip")
    else:
        st.error(f"File database tidak ditemukan di {FILE_PATH}")

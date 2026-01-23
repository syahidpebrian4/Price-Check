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
TARGET_IMAGE_SIZE_KB = 195 

TEXTS_TO_REDACT = [
    "HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "WAYAN GIYANTO", 
    "MEMBER UMUM KLIK", "DJUANMING", "NONOK JUNENGSIH", 
    "AGUNG KURNIAWAN", "ARIF RAMADHAN", "HILMI ATIQ"
]

st.set_page_config(page_title="Price Check V10.5 - Full Detail", layout="wide")

def clean_and_repair_price(raw_segment):
    translation_table = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'E': '8', 'G': '6', 'Z': '2', '+': '', 'A': '4'}
    text = re.sub(r'[\s.,\-]', '', str(raw_segment))
    for char, digit in translation_table.items():
        text = text.replace(char, digit)
    nums = re.findall(r'\d{3,7}', text)
    return int(nums[0]) if nums else None

def extract_prices_smart(text_content):
    text_content = re.sub(r'\(.*?\)|ISI\s*\d+', '', text_content)
    found_prices = []
    parts = text_content.split('RP')
    for part in parts[1:]:
        target_segment = part.split('/')[0].split('\n')[0]
        price = clean_and_repair_price(target_segment)
        if price and price > 500: found_prices.append(price)
    
    if not found_prices: return {"normal": 0, "promo": 0}
    if len(found_prices) >= 2:
        return {"normal": max(found_prices), "promo": min(found_prices)}
    else:
        return {"normal": found_prices[0], "promo": found_prices[0]}

def process_ocr_all_prices(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_np, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Get Data from Tesseract
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    
    data_list = []
    full_text_list = []
    for i in range(len(d['text'])):
        txt = d['text'][i].strip().upper()
        if txt:
            data_list.append({
                "text": txt,
                "top": d['top'][i],
                "left": d['left'][i],
                "width": d['width'][i],
                "height": d['height'][i]
            })
            full_text_list.append(txt)
    
    df_ocr = pd.DataFrame(data_list)
    raw_full_text = " ".join(full_text_list)
    final_results = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    product_name_on_image = "N/A"

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        idx_search = df_ocr[df_ocr['text'].str.contains("CARI|KLIK|SEMUA", na=False)].index
        if not idx_search.empty and (idx_search[-1] + 1) < len(df_ocr):
            product_name_on_image = " ".join(df_ocr.iloc[idx_search[-1] + 1 : idx_search[-1] + 6]['text'].tolist())

        if any(x in raw_full_text for x in ["PCS", "SATUAN", "RCG", "BOX"]):
            final_results["PCS"] = extract_prices_smart(raw_full_text)
        if any(x in raw_full_text for x in ["CTN", "KARTON", "DUS"]):
            final_results["CTN"] = extract_prices_smart(raw_full_text)

    # Redaksi
    draw = ImageDraw.Draw(pil_image)
    for i in range(len(d['text'])):
        word = d['text'][i].upper()
        if word:
            for keyword in TEXTS_TO_REDACT:
                if word in keyword.upper() and len(word) > 3:
                    x, y, w, h = d['left'][i]/1.5, d['top'][i]/1.5, d['width'][i]/1.5, d['height'][i]/1.5
                    draw.rectangle([x-2, y-2, x+w+2, y+h+2], fill="white")

    return final_results["PCS"], final_results["CTN"], product_name_on_image, raw_full_text, pil_image

def compress_to_target(pil_img, target_kb):
    if pil_img.mode in ("RGBA", "P"): pil_img = pil_img.convert("RGB")
    quality = 95
    buf = io.BytesIO()
    while quality > 10:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() / 1024 <= target_kb: break
        quality -= 5
    return buf.getvalue()

# ================= UI STREAMLIT =================
st.title("📸 Price Check (Full Scan Mode)")

col1, col2, col3 = st.columns(3)
with col1: m_code_input = st.text_input("📍 Master Code").strip().upper()
with col2: date_input = st.text_input("📅 Tanggal").strip().upper()
with col3: week_input = st.text_input("🗓️ Week").strip()

files = st.file_uploader("📂 Upload Foto", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code_input and date_input and week_input:
    if os.path.exists(FILE_PATH):
        xl = pd.ExcelFile(FILE_PATH)
        db_ig = xl.parse(SHEET_MASTER_IG)
        db_ig.columns = db_ig.columns.astype(str).str.strip()
        db_targets = {s: xl.parse(s) for s in SHEETS_TARGET if s in xl.sheet_names}

        final_list = []
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, f in enumerate(files):
                with st.expander(f"🔍 Detail: {f.name}", expanded=True):
                    img_pil = Image.open(f)
                    res_pcs, res_ctn, scanned_name, raw_text, redacted_pil = process_ocr_all_prices(img_pil)
                    
                    # Matching
                    found_prodcode, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_name = str(row[COL_IG_NAME]).upper()
                        score = fuzz.token_set_ratio(db_name, scanned_name)
                        if score > 75 and score > best_score:
                            best_score, found_prodcode = score, str(row["PRODCODE"])
                    
                    # Tampilan kolom kiri (Gambar) & kanan (Data)
                    ca, cb = st.columns([1, 2])
                    ca.image(redacted_pil, caption="Hasil Sensor")
                    
                    with cb:
                        st.write(f"**Nama Terdeteksi:** `{scanned_name}`")
                        st.write(f"**Match Code:** `{found_prodcode}` (Score: {best_score})")
                        st.markdown("**Hasil Scan Mentah:**")
                        st.caption(raw_text) # INI MENAMPILKAN SELURUH HASIL SCAN
                        
                        st.json({"PCS": res_pcs, "CTN": res_ctn})

                    if found_prodcode:
                        # Logic simpan ke list & zip (sama seperti sebelumnya)
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            match = df_t[(df_t["PRODCODE"].astype(str).str.contains(found_prodcode)) & 
                                         (df_t["MASTER Code"].astype(str).str.contains(m_code_input))]
                            if not match.empty:
                                final_list.append({
                                    "prodcode": found_prodcode, "sheet": s_name, "index": match.index[0],
                                    "n_pcs": res_pcs['normal'], "p_pcs": res_pcs['promo'],
                                    "n_ctn": res_ctn['normal'], "p_ctn": res_ctn['promo']
                                })
                                img_bytes = compress_to_target(redacted_pil, TARGET_IMAGE_SIZE_KB)
                                zip_file.writestr(f"{found_prodcode}.jpg", img_bytes)
                                break
                gc.collect()

        if final_list:
            st.write("### 📋 Ringkasan Update")
            st.dataframe(pd.DataFrame(final_list))
            
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🚀 UPDATE DATABASE EXCEL"):
                    wb = load_workbook(FILE_PATH)
                    for r in final_list:
                        ws = wb[r['sheet']]
                        headers = [str(c.value).strip() for c in ws[1]]
                        row_num = r['index'] + 2
                        mapping = {"Normal Competitor Price (Pcs)": r['n_pcs'], "Promo Competitor Price (Pcs)": r['p_pcs'], "Normal Competitor Price (Ctn)": r['n_ctn'], "Promo Competitor Price (Ctn)": r['p_ctn']}
                        for col_name, val in mapping.items():
                            if col_name in headers: ws.cell(row=row_num, column=headers.index(col_name) + 1).value = val
                    wb.save(FILE_PATH)
                    st.success("Database Updated!")
            
            with c2: st.download_button("📥 DOWNLOAD EXCEL", open(FILE_PATH, "rb"), f"Price Check W{week_input}_{date_input}.xlsx")
            with c3: st.download_button("🖼️ DOWNLOAD ZIP", zip_buffer.getvalue(), f"{m_code_input}_{date_input}.zip")

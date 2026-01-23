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

st.set_page_config(page_title="Price Check V10.5 - Tesseract", layout="wide")

# ==================================================
# CORE OCR ENGINE (TESSERACT STABLE)
# ==================================================
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
    # Pre-processing
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # Resize 1.5x untuk stabilitas Tesseract
    img_resized = cv2.resize(img_np, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Get Data from Tesseract
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    
    data = []
    full_text_raw = ""
    for i in range(len(d['text'])):
        txt = d['text'][i].strip().upper()
        if txt:
            data.append({
                "text": txt,
                "top": d['top'][i],
                "left": d['left'][i],
                "width": d['width'][i],
                "height": d['height'][i]
            })
            full_text_raw += txt + " "
    
    df_ocr = pd.DataFrame(data)
    final_results = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    product_name_on_image = ""

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        
        # Logika Nama Produk
        idx_search = df_ocr[df_ocr['text'].str.contains("CARI|KLIK|SEMUA", na=False)].index
        if not idx_search.empty and (idx_search[-1] + 1) < len(df_ocr):
            product_name_on_image = " ".join(df_ocr.iloc[idx_search[-1] + 1 : idx_search[-1] + 6]['text'].tolist())

        # Logika Harga PCS
        if any(x in full_text_raw for x in ["PCS", "SATUAN", "RCG", "BOX"]):
            final_results["PCS"] = extract_prices_smart(full_text_raw)
            
        # Logika Harga CTN
        if any(x in full_text_raw for x in ["CTN", "KARTON", "DUS"]):
            final_results["CTN"] = extract_prices_smart(full_text_raw)

    # --- Bagian Auto-Redact (Menggunakan Koordinat Tesseract) ---
    draw = ImageDraw.Draw(pil_image)
    for i in range(len(d['text'])):
        word = d['text'][i].upper()
        if word:
            for keyword in TEXTS_TO_REDACT:
                # Fuzzy simple check
                if word in keyword.upper() and len(word) > 3:
                    # Kembalikan koordinat ke skala asli (1.0 / 1.5)
                    x = d['left'][i] / 1.5
                    y = d['top'][i] / 1.5
                    w = d['width'][i] / 1.5
                    h = d['height'][i] / 1.5
                    draw.rectangle([x-2, y-2, x+w+2, y+h+2], fill="white")

    return final_results["PCS"], final_results["CTN"], product_name_on_image, pil_image

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
def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("📸 Price Check V10.5 (Tesseract Stable)")

col_input1, col_input2, col_input3 = st.columns(3)
with col_input1:
    m_code_input = st.text_input("📍 Master Code").strip().upper()
with col_input2:
    date_input = st.text_input("📅 Tanggal (23JAN2026)").strip().upper()
with col_input3:
    week_input = st.text_input("🗓️ Week").strip()

files = st.file_uploader("📂 Upload Foto", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code_input and date_input and week_input:
    if os.path.exists(FILE_PATH):
        xl = pd.ExcelFile(FILE_PATH)
        db_ig = xl.parse(SHEET_MASTER_IG)
        db_ig.columns = db_ig.columns.astype(str).str.strip()
        
        db_targets = {s: xl.parse(s) for s in SHEETS_TARGET if s in xl.sheet_names}
        for s in db_targets: db_targets[s].columns = db_targets[s].columns.astype(str).str.strip()

        final_list = []
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, f in enumerate(files):
                with st.status(f"Memproses {f.name}...", expanded=False) as status:
                    img_pil = Image.open(f)
                    res_pcs, res_ctn, scanned_name, redacted_pil = process_ocr_all_prices(img_pil)
                    
                    found_prodcode, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_name = str(row[COL_IG_NAME]).upper()
                        score = fuzz.token_set_ratio(db_name, scanned_name)
                        if score > 75 and score > best_score:
                            best_score, found_prodcode = score, norm(row["PRODCODE"])
                    
                    if found_prodcode:
                        target_sheet, target_idx = None, None
                        for s_name, df_t in db_targets.items():
                            match = df_t[(df_t["PRODCODE"].astype(str).apply(norm) == found_prodcode) & 
                                         (df_t["MASTER Code"].astype(str).apply(norm) == norm(m_code_input))]
                            if not match.empty:
                                target_sheet, target_idx = s_name, match.index[0]
                                break
                        
                        if target_sheet:
                            final_list.append({
                                "prodcode": found_prodcode, "sheet": target_sheet, "index": target_idx,
                                "n_pcs": res_pcs['normal'], "p_pcs": res_pcs['promo'],
                                "n_ctn": res_ctn['normal'], "p_ctn": res_ctn['promo']
                            })
                            img_bytes = compress_to_target(redacted_pil, TARGET_IMAGE_SIZE_KB)
                            zip_file.writestr(f"{found_prodcode}.jpg", img_bytes)
                            status.update(label=f"✅ Match: {found_prodcode}", state="complete")
                        else:
                            st.warning(f"⚠️ {found_prodcode} tidak ditemukan di DF/HBHC (MC: {m_code_input})")
                    else:
                        st.error(f"❌ '{scanned_name}' tak cocok di IG.")
                gc.collect()

        if final_list:
            st.write("### 📋 Ringkasan")
            st.table(pd.DataFrame(final_list)[["prodcode", "sheet", "n_pcs", "p_pcs", "n_ctn", "p_ctn"]])
            
            c1, c2, c3 = st.columns(3)
            with c1:
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
                            "Promo Competitor Price (Ctn)": r['p_ctn']
                        }
                        for col_name, val in mapping.items():
                            if col_name in headers:
                                ws.cell(row=row_num, column=headers.index(col_name) + 1).value = val
                    wb.save(FILE_PATH)
                    st.success("Database Updated!")
            
            with c2: st.download_button("📥 DOWNLOAD EXCEL", open(FILE_PATH, "rb"), f"Price Check W{week_input}_{date_input}.xlsx")
            with c3: st.download_button("🖼️ DOWNLOAD ZIP", zip_buffer.getvalue(), f"{m_code_input}_{date_input}.zip")

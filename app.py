import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
import os
from PIL import Image, ImageDraw
import io
import zipfile
from fuzzywuzzy import fuzz

# ================= CONFIG =================
FILE_PATH = "database/master_harga.xlsx"
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 
TARGET_IMAGE_SIZE_KB = 195 

TEXTS_TO_REDACT = ["HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "Halo WAYAN GIYANTO / WRG", "Halo MEMBER UMUM KLIK", "Halo DJUANMING / TK GOGO", "Halo NONOK JUNENGSIH", "Halo AGUNG KURNIAWAN", "Halo ARIF RAMADHAN", "Halo HILMI ATIQ / WR DINDA"]

st.set_page_config(page_title="Price Check", layout="wide")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False) 

reader = load_reader()

def process_ocr_all_prices(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_pil_for_redact = pil_image.copy()

    # OCR Proses
    img_resized = cv2.resize(img_np, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    processed = cv2.bilateralFilter(gray, 9, 75, 75)
    results_ocr = reader.readtext(processed, detail=1)

    data = []
    for (bbox, text, prob) in results_ocr:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        data.append({"text": text.upper(), "top": y_center})
    
    df_ocr = pd.DataFrame(data)
    final_results = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    product_name_on_image = ""

    def clean_and_repair_price(raw_segment):
        translation_table = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'E': '8', 'G': '6', 'Z': '2'}
        text = re.sub(r'[\s.,\-]', '', raw_segment)
        for char, digit in translation_table.items():
            text = text.replace(char, digit)
        nums = re.findall(r'\d{3,7}', text)
        return int(nums[0]) if nums else 0

    def extract_prices_smart(text_content):
        found_prices = []
        parts = text_content.split('RP')
        for part in parts[1:]:
            target_segment = part.split('/')[0]
            price = clean_and_repair_price(target_segment)
            if price > 0: found_prices.append(price)
        
        if not found_prices: return {"normal": 0, "promo": 0}
        return {"normal": found_prices[0], "promo": found_prices[1] if len(found_prices) > 1 else found_prices[0]}

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        # Cari Nama Produk
        idx_search = df_ocr[df_ocr['text'].str.contains("CARI DI KLIK|SEMUA KATEGORI", na=False)].index
        if not idx_search.empty and (idx_search[-1] + 1) < len(df_ocr):
            product_name_on_image = df_ocr.iloc[idx_search[-1] + 1]['text']

        # Cari Harga PCS
        idx_pilih = df_ocr[df_ocr['text'].str.contains("PILIH SATUAN", na=False)].index
        if not idx_pilih.empty:
            search_area = df_ocr.iloc[idx_pilih[0]:]
            idx_pcs = search_area[search_area['text'].str.contains("PCS|RCG|BOX|PCK", na=False)].index
            if not idx_pcs.empty:
                target_rows = df_ocr.iloc[idx_pcs[0] : idx_pcs[0] + 3]
                final_results["PCS"] = extract_prices_smart(" # ".join(target_rows['text'].tolist()))

        # Cari Harga CTN
        idx_ctn = df_ocr[df_ocr['text'].str.contains("CTN|KARTON|DUS", na=False)].index
        if not idx_ctn.empty:
            target_rows_ctn = df_ocr.iloc[idx_ctn[0] : idx_ctn[0] + 3]
            final_results["CTN"] = extract_prices_smart(" # ".join(target_rows_ctn['text'].tolist()))

    # --- Auto Redact ---
    draw = ImageDraw.Draw(original_pil_for_redact)
    results_redact = reader.readtext(cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY))
    for (bbox, text, prob) in results_redact:
        for keyword in TEXTS_TO_REDACT:
            if fuzz.partial_ratio(keyword.upper(), text.upper()) > 85:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                draw.rectangle([tuple(top_left), tuple(bottom_right)], fill="white")

    return final_results["PCS"], final_results["CTN"], product_name_on_image, original_pil_for_redact

def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

# ================= UI =================
st.title("📸 Price Check")

col1, col2, col3 = st.columns(3)
with col1: m_code = st.text_input("📍 Master Code").strip().upper()
with col2: tgl = st.text_input("📅 Tanggal (23JAN2026)").strip().upper()
with col3: week = st.text_input("🗓️ Week").strip()

files = st.file_uploader("📂 Upload Foto", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code and tgl and week:
    if os.path.exists(FILE_PATH):
        xl = pd.ExcelFile(FILE_PATH)
        db_ig = xl.parse(SHEET_MASTER_IG)
        
        final_list = []
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for f in files:
                img_pil = Image.open(f)
                res_pcs, res_ctn, scanned_name, redacted_img = process_ocr_all_prices(img_pil)
                
                # Matching
                found_code = None
                best_score = 0
                for _, row in db_ig.iterrows():
                    score = fuzz.token_set_ratio(str(row[COL_IG_NAME]).upper(), scanned_name)
                    if score > 80 and score > best_score:
                        best_score = score
                        found_code = norm(row["PRODCODE"])
                
                if found_code:
                    final_list.append({
                        "PRODCODE": found_code, "N_PCS": res_pcs['normal'], "P_PCS": res_pcs['promo'],
                        "N_CTN": res_ctn['normal'], "P_CTN": res_ctn['promo'],
                        "SCANNED": scanned_name, "IMAGE": redacted_img
                    })
                    
                    # Simpan ke ZIP
                    img_byte = io.BytesIO()
                    redacted_img.save(img_byte, format="JPEG")
                    zip_file.writestr(f"{found_code}.jpg", img_byte.getvalue())

        if final_list:
            st.success("Selesai!")
            
            # Download Buttons
            c_dl1, c_dl2 = st.columns(2)
            with c_dl1: st.download_button(f"📥 DOWNLOAD EXCEL", open(FILE_PATH, "rb"), f"Price Check W{week}_{tgl}.xlsx")
            with c_dl2: st.download_button(f"🖼️ DOWNLOAD ZIP", zip_buffer.getvalue(), f"{m_code}_{tgl}.zip")

            # --- BAGIAN PREVIEW (YANG ANDA TANYAKAN) ---
            st.write("---")
            st.write("### 🔍 Preview Hasil Scan")
            
            # Tabel Ringkasan
            df_final = pd.DataFrame(final_list).drop(columns=['IMAGE'])
            st.table(df_final)

            # Preview Gambar per Produk
            for item in final_list:
                with st.expander(f"🖼️ Detail: {item['PRODCODE']} - {item['SCANNED']}"):
                    col_p1, col_p2 = st.columns([1, 2])
                    with col_p1:
                        st.json({
                            "Harga PCS": {"Normal": item['N_PCS'], "Promo": item['P_PCS']},
                            "Harga CTN": {"Normal": item['N_CTN'], "Promo": item['P_CTN']}
                        })
                    with col_p2:
                        st.image(item['IMAGE'], caption=f"Hasil Redaksi: {item['PRODCODE']}")
    else:
        st.error("Database Excel tidak ditemukan.")

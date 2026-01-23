import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
from PIL import Image, ImageDraw
import io
import zipfile
from fuzzywuzzy import fuzz

# ================= CONFIG =================
URL_BASE = "https://docs.google.com/spreadsheets/d/1vz2tEQ7YbFuKSF172ihDFBh6YE3F4Ql1jSEOpptZN34"
SHEET_MASTER_IG = "IG"
COL_IG_NAME = "PRODNAME_IG"
TARGET_IMAGE_SIZE_KB = 195 

# Daftar Sensor (Redaksi)
TEXTS_TO_REDACT = [
    "HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "Halo WAYAN GIYANTO / WRG", 
    "Halo MEMBER UMUM KLIK", "Halo DJUANMING / TK GOGO", "Halo NONOK JUNENGSIH", 
    "Halo AGUNG KURNIAWAN", "Halo ARIF RAMADHAN", "Halo HILMI ATIQ / WR DINDA"
]

st.set_page_config(page_title="Price Check AI V10.5", layout="wide")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False) 

reader = load_reader()

# ==================================================
# CORE OCR ENGINE (LOGIKA ASLI V10.5 + AUTO-REDACT)
# ==================================================
def process_ocr_all_prices(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_pil_for_redact = pil_image.copy()

    # Logika Pre-processing V10.5
    img_resized_for_ocr = cv2.resize(img_np, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    gray_for_ocr = cv2.cvtColor(img_resized_for_ocr, cv2.COLOR_BGR2GRAY)
    processed_for_ocr = cv2.bilateralFilter(gray_for_ocr, 9, 75, 75)
    
    results_ocr_prices = reader.readtext(processed_for_ocr, detail=1)

    data = []
    for (bbox, text, prob) in results_ocr_prices:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        data.append({"text": text.upper(), "top": y_center, "bbox": bbox})
    
    df_ocr = pd.DataFrame(data)
    final_results = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    product_name_on_image = ""

    def clean_and_repair_price(raw_segment):
        translation_table = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'E': '8', 'G': '6', 'Z': '2', '+': '', 'A': '4'}
        text = re.sub(r'[\s.,\-]', '', str(raw_segment))
        for char, digit in translation_table.items():
            text = text.replace(char, digit)
        nums = re.findall(r'\d{3,7}', text)
        return int(nums[0]) if nums else 0

    def extract_prices_smart(text_content):
        text_content = re.sub(r'\(.*?\)|ISI\s*\d+', '', text_content)
        found_prices = []
        parts = text_content.split('RP')
        for part in parts[1:]:
            target_segment = part.split('/')[0]
            price = clean_and_repair_price(target_segment)
            if price and price > 0: found_prices.append(price)
        
        if not found_prices: return {"normal": 0, "promo": 0}
        if len(found_prices) >= 2:
            return {"normal": found_prices[0], "promo": found_prices[1]}
        else:
            return {"normal": found_prices[0], "promo": found_prices[0]}

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        
        # Deteksi Nama Produk
        idx_search = df_ocr[df_ocr['text'].str.contains("CARI DI KLIK|SEMUA KATEGORI", na=False)].index
        if not idx_search.empty and (idx_search[-1] + 1) < len(df_ocr):
            product_name_on_image = df_ocr.iloc[idx_search[-1] + 1]['text']

        # Deteksi Harga PCS
        idx_pilih = df_ocr[df_ocr['text'].str.contains("PILIH SATUAN", na=False)].index
        if not idx_pilih.empty:
            search_area = df_ocr.iloc[idx_pilih[0]:]
            idx_pcs = search_area[search_area['text'].str.contains("PCS|RCG|BOX|PCK", na=False)].index
            if not idx_pcs.empty:
                target_rows = df_ocr.iloc[idx_pcs[0] : idx_pcs[0] + 3]
                all_text = " # ".join(target_rows['text'].tolist())
                final_results["PCS"] = extract_prices_smart(all_text)

        # Deteksi Harga CTN
        idx_ctn = df_ocr[df_ocr['text'].str.contains("CTN|KARTON|DUS", na=False)].index
        if not idx_ctn.empty:
            target_rows_ctn = df_ocr.iloc[idx_ctn[0] : idx_ctn[0] + 3]
            all_text_ctn = " # ".join(target_rows_ctn['text'].tolist())
            final_results["CTN"] = extract_prices_smart(all_text_ctn)

    # --- Bagian Auto-Redact ---
    draw = ImageDraw.Draw(original_pil_for_redact)
    # Gunakan deteksi paragraph=False untuk akurasi posisi redaksi
    results_redact = reader.readtext(cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY), detail=1)

    for (bbox, text, prob) in results_redact:
        for keyword in TEXTS_TO_REDACT:
            if fuzz.partial_ratio(keyword.upper(), text.upper()) > 85:
                (tl, tr, br, bl) = bbox
                draw.rectangle([tl[0]-5, tl[1]-5, br[0]+5, br[1]+5], fill="white")
                break

    return final_results["PCS"], final_results["CTN"], product_name_on_image, original_pil_for_redact

def compress_to_target(pil_img, target_kb):
    if pil_img.mode in ("RGBA", "P"): pil_img = pil_img.convert("RGB")
    quality = 95
    buf = io.BytesIO()
    while quality > 10:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() / 1024 <= target_kb: return buf.getvalue()
        quality -= 5
    return buf.getvalue()

def get_data_fixed(sheet_name):
    try:
        csv_url = f"{URL_BASE}/export?format=csv&sheet={sheet_name}"
        df = pd.read_csv(csv_url, usecols=[0, 1], on_bad_lines='skip', engine='python')
        df.columns = ["PRODCODE", "PRODNAME_IG"]
        df = df.dropna(subset=["PRODCODE", "PRODNAME_IG"]).reset_index(drop=True)
        df["PRODCODE"] = df["PRODCODE"].astype(str).str.strip()
        df["PRODNAME_IG"] = df["PRODNAME_IG"].astype(str).apply(lambda x: x.strip().upper())
        return df
    except Exception as e:
        st.error(f"Gagal memuat database: {e}")
        return None

# ================= UI STREAMLIT =================
st.title("📸 Price Check AI V10.5")

c1, c2, c3 = st.columns(3)
with c1: m_code_in = st.text_input("📍 Master Code").strip().upper()
with c2: tgl_in = st.text_input("📅 Tanggal (Ex: 23JAN2026)").strip().upper()
with c3: week_in = st.text_input("🗓️ Week").strip()

files = st.file_uploader("📂 Upload Foto", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code_in:
    db_ig = get_data_fixed(SHEET_MASTER_IG)
    
    if db_ig is not None:
        st.success(f"✅ Database IG Terhubung ({len(db_ig)} Produk)")
        
        final_list = []
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for f in files:
                img_pil = Image.open(f)
                res_pcs, res_ctn, s_name, red_img = process_ocr_all_prices(img_pil)
                
                # Matching Logic
                best_code, max_score = None, 0
                for _, row in db_ig.iterrows():
                    score = fuzz.token_set_ratio(row["PRODNAME_IG"], s_name)
                    if score > 80 and score > max_score:
                        max_score, best_code = score, row["PRODCODE"]
                
                if best_code:
                    with st.expander(f"✅ Match: {best_code}"):
                        st.write(f"Nama Scan: **{s_name}**")
                        st.json({"PCS": res_pcs, "CTN": res_ctn})
                        st.image(red_img, width=400)
                        
                        # Simpan ke ZIP (dengan Kompresi target 195KB)
                        img_bytes = compress_to_target(red_img, TARGET_IMAGE_SIZE_KB)
                        zip_file.writestr(f"{best_code}.jpg", img_bytes)
                        
                        final_list.append({
                            "PRODCODE": best_code, 
                            "N_PCS": res_pcs['normal'], "P_PCS": res_pcs['promo'],
                            "N_CTN": res_ctn['normal'], "P_CTN": res_ctn['promo']
                        })
                else:
                    st.warning(f"⚠️ Tidak Dikenali: {s_name}")

        if final_list:
            st.write("### 📋 Ringkasan Hasil Scan")
            st.dataframe(pd.DataFrame(final_list))
            
            # Tombol Download
            zip_fn = f"{m_code_in}_{tgl_in}.zip"
            st.download_button("📥 Download ZIP (JPG 195KB)", zip_buffer.getvalue(), zip_fn)

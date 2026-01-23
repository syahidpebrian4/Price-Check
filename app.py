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
from streamlit_gsheets import GSheetsConnection

# ================= CONFIG =================
# FILE_PATH tidak lagi digunakan untuk baca, tapi bisa tetap ada jika ingin simpan cache
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

# ==================================================
# CORE OCR ENGINE
# ==================================================
def process_ocr_all_prices(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_pil_for_redact = pil_image.copy()

    # Resize untuk OCR
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
        text = re.sub(r'[\s.,\-]', '', raw_segment)
        for char, digit in translation_table.items():
            text = text.replace(char, digit)
        nums = re.findall(r'\d{3,7}', text)
        return int(nums[0]) if nums else None

    def extract_prices_smart(text_content):
        text_content = re.sub(r'\(.*?\)|ISI\s*\d+', '', text_content)
        found_prices = []
        parts = text_content.split('RP')
        for part in parts[1:]:
            target_segment = part.split('/')[0]
            price = clean_and_repair_price(target_segment)
            if price: found_prices.append(price)
        
        if not found_prices: return {"normal": 0, "promo": 0}
        if len(found_prices) >= 2:
            return {"normal": found_prices[0], "promo": found_prices[1]}
        else:
            return {"normal": found_prices[0], "promo": found_prices[0]}

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        idx_search = df_ocr[df_ocr['text'].str.contains("CARI DI KLIK|SEMUA KATEGORI", na=False)].index
        if not idx_search.empty and (idx_search[-1] + 1) < len(df_ocr):
            product_name_on_image = df_ocr.iloc[idx_search[-1] + 1]['text']

        idx_pilih = df_ocr[df_ocr['text'].str.contains("PILIH SATUAN", na=False)].index
        if not idx_pilih.empty:
            search_area = df_ocr.iloc[idx_pilih[0]:]
            idx_pcs = search_area[search_area['text'].str.contains("PCS|RCG|BOX|PCK", na=False)].index
            if not idx_pcs.empty:
                target_rows = df_ocr.iloc[idx_pcs[0] : idx_pcs[0] + 3]
                all_text = " # ".join(target_rows['text'].tolist())
                final_results["PCS"] = extract_prices_smart(all_text)

        idx_ctn = df_ocr[df_ocr['text'].str.contains("CTN|KARTON|DUS", na=False)].index
        if not idx_ctn.empty:
            target_rows_ctn = df_ocr.iloc[idx_ctn[0] : idx_ctn[0] + 3]
            all_text_ctn = " # ".join(target_rows_ctn['text'].tolist())
            final_results["CTN"] = extract_prices_smart(all_text_ctn)

    # --- Bagian Auto-Redact ---
    draw = ImageDraw.Draw(original_pil_for_redact)
    temp_img_for_redact = cv2.resize(img_np, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
    results_redact = reader.readtext(cv2.cvtColor(temp_img_for_redact, cv2.COLOR_BGR2GRAY), detail=1, paragraph=False)

    for (bbox, text, prob) in results_redact:
        for keyword in TEXTS_TO_REDACT:
            if fuzz.partial_ratio(keyword.upper(), text.upper()) > 85:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                x_min, y_min = int(min(top_left[0], bottom_left[0])), int(min(top_left[1], top_right[1]))
                x_max, y_max = int(max(top_right[0], bottom_right[0])), int(max(bottom_left[1], bottom_right[1]))
                draw.rectangle([(x_min-5, y_min-5), (x_max+5, y_max+5)], fill="white")
                break

    return final_results["PCS"], final_results["CTN"], processed_for_ocr, product_name_on_image, original_pil_for_redact

def compress_to_target(pil_img, target_kb):
    if pil_img.mode in ("RGBA", "P"): pil_img = pil_img.convert("RGB")
    quality = 95
    while quality > 10:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() / 1024 <= target_kb: return buf.getvalue()
        quality -= 5
    return buf.getvalue()

# ================= UI STREAMLIT =================
def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("📸 Price Check")

col_input1, col_input2, col_input3 = st.columns(3)
with col_input1:
    m_code_input = st.text_input("📍 Masukkan Master Code").strip().upper()
with col_input2:
    date_input = st.text_input("📅 Masukkan Tanggal (Contoh: 23JAN2026)").strip().upper()
with col_input3:
    week_input = st.text_input("🗓️ Masukkan Week (Contoh: 4)").strip()

files = st.file_uploader("📂 Upload Foto Product", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code_input and date_input and week_input:
    # --- KONEKSI GOOGLE SHEETS ---
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        db_ig = conn.read(worksheet=SHEET_MASTER_IG, ttl="1m")
        db_ig.columns = db_ig.columns.astype(str).str.strip()
        
        db_targets = {}
        for s in SHEETS_TARGET:
            df_t = conn.read(worksheet=s, ttl="1m")
            df_t.columns = df_t.columns.astype(str).str.strip()
            db_targets[s] = df_t
            
    except Exception as e:
        st.error(f"Gagal koneksi ke Google Sheets: {e}")
        st.stop()

    if COL_IG_NAME in db_ig.columns:
        final_list = []
        zip_buffer = io.BytesIO()
        progress_bar = st.progress(0)
        status_text = st.empty()

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for idx, f in enumerate(files):
                status_text.text(f"⏳ Memproses {f.name} ({idx+1}/{len(files)})...")
                img_pil = Image.open(f)
                res_pcs, res_ctn, _, scanned_name, redacted_pil = process_ocr_all_prices(img_pil)
                
                # Matching
                found_prodcode, best_score = None, 0
                for _, row in db_ig.iterrows():
                    db_name = str(row[COL_IG_NAME]).upper()
                    score = fuzz.token_set_ratio(db_name, scanned_name)
                    if score > 80 and score > best_score:
                        best_score = score
                        found_prodcode = norm(row["PRODCODE"])
                
                if found_prodcode:
                    target_sheet, target_idx = None, None
                    for s_name, df_t in db_targets.items():
                        if "PRODCODE" in df_t.columns and "MASTER Code" in df_t.columns:
                            match = df_t[
                                (df_t["PRODCODE"].astype(str).apply(norm) == found_prodcode) & 
                                (df_t["MASTER Code"].astype(str).apply(norm) == norm(m_code_input))
                            ]
                            if not match.empty:
                                target_sheet, target_idx = s_name, match.index[0]
                                break
                    
                    if target_sheet:
                        final_list.append({
                            "prodcode": found_prodcode, "sheet": target_sheet, "index": target_idx,
                            "n_pcs": res_pcs['normal'], "p_pcs": res_pcs['promo'],
                            "n_ctn": res_ctn['normal'], "p_ctn": res_ctn['promo']
                        })
                        
                        # Compress & Zip
                        w, h = redacted_pil.size
                        img_cropped = redacted_pil.crop((0, 75, w, h)) if h > 75 else redacted_pil
                        zip_file.writestr(f"{found_prodcode}.jpg", compress_to_target(img_cropped, TARGET_IMAGE_SIZE_KB))
                        
                        with st.expander(f"✅ Match: {found_prodcode}"):
                            st.json({"PCS": res_pcs, "CTN": res_ctn})
                            st.image(redacted_pil, width=400)
                    else:
                        st.warning(f"⚠️ {found_prodcode} tidak ditemukan di sheet target (DF/HBHC).")
                else:
                    st.error(f"❌ '{scanned_name}' tidak cocok di Master IG.")
                
                progress_bar.progress((idx + 1) / len(files))

        status_text.success("Pemrosesan Selesai!")

        if final_list:
            st.write("### 📋 Ringkasan Hasil")
            res_df = pd.DataFrame(final_list)
            st.table(res_df[["prodcode", "sheet", "n_pcs", "p_pcs", "n_ctn", "p_ctn"]])
            
            # Catatan: Fitur 'UPDATE DATABASE' via GSheetsConnection memerlukan akses write khusus.
            # Untuk saat ini, user bisa download file hasil untuk referensi.
            
            excel_filename = f"Price_Check_W{week_input}_{date_input}.xlsx"
            zip_filename = f"{m_code_input}_{date_input}.zip"

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                # Simpan Ringkasan ke Excel sementara untuk didownload
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    res_df.to_excel(writer, index=False, sheet_name='Summary')
                st.download_button("📥 DOWNLOAD SUMMARY EXCEL", output.getvalue(), excel_filename)
            with col_dl2:
                st.download_button("🖼️ DOWNLOAD ZIP FOTO", zip_buffer.getvalue(), zip_filename)

    else:
        st.error(f"Kolom '{COL_IG_NAME}' tidak ditemukan di Spreadsheet.")

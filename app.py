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

# ================= CONFIG (LINK DATABASE) =================
# Link Google Sheets Anda
URL_DB = "https://docs.google.com/spreadsheets/d/1vz2tEQ7YbFuKSF172ihDFBh6YE3F4Ql1jSEOpptZN34/edit?usp=sharing"

SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 
TARGET_IMAGE_SIZE_KB = 195 

# Daftar teks untuk sensor otomatis (Redaksi)
TEXTS_TO_REDACT = [
    "HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "Halo WAYAN GIYANTO / WRG", 
    "Halo MEMBER UMUM KLIK", "Halo DJUANMING / TK GOGO", "Halo NONOK JUNENGSIH", 
    "Halo AGUNG KURNIAWAN", "Halo ARIF RAMADHAN", "Halo HILMI ATIQ / WR DINDA"
]

st.set_page_config(page_title="Price Check AI", layout="wide")

@st.cache_resource
def load_reader():
    # Menggunakan CPU karena Streamlit Free Tier tidak mendukung GPU
    return easyocr.Reader(['en'], gpu=False) 

reader = load_reader()

# ================= CORE OCR ENGINE =================
def process_ocr_all_prices(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_pil = pil_image.copy()

    # Pre-processing Image
    img_resized = cv2.resize(img_np, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    processed = cv2.bilateralFilter(gray, 9, 75, 75)
    
    results = reader.readtext(processed, detail=1)
    data = []
    for (bbox, text, prob) in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        data.append({"text": text.upper(), "top": y_center, "bbox": bbox})
    
    df_ocr = pd.DataFrame(data)
    final_res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    scanned_name = ""

    def clean_repair_price(raw):
        trans = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'E': '8', 'G': '6', 'Z': '2', '+': '', 'A': '4'}
        text = re.sub(r'[\s.,\-]', '', str(raw))
        for char, digit in trans.items():
            text = text.replace(char, digit)
        nums = re.findall(r'\d{3,7}', text)
        return int(nums[0]) if nums else 0

    def extract_prices_smart(text_content):
        text_content = re.sub(r'\(.*?\)|ISI\s*\d+', '', text_content)
        parts = text_content.split('RP')
        found = [clean_repair_price(p.split('/')[0]) for p in parts[1:] if clean_repair_price(p.split('/')[0]) > 0]
        if not found: return {"normal": 0, "promo": 0}
        return {"normal": found[0], "promo": found[1] if len(found) >= 2 else found[0]}

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        
        # Deteksi Nama Produk
        idx_search = df_ocr[df_ocr['text'].str.contains("CARI DI KLIK|SEMUA KATEGORI", na=False)].index
        if not idx_search.empty and (idx_search[-1] + 1) < len(df_ocr):
            scanned_name = df_ocr.iloc[idx_search[-1] + 1]['text']

        # Deteksi Harga PCS
        idx_pilih = df_ocr[df_ocr['text'].str.contains("PILIH SATUAN", na=False)].index
        if not idx_pilih.empty:
            area = df_ocr.iloc[idx_pilih[0]:]
            idx_pcs = area[area['text'].str.contains("PCS|RCG|BOX|PCK", na=False)].index
            if not idx_pcs.empty:
                rows = df_ocr.iloc[idx_pcs[0] : idx_pcs[0] + 3]
                final_res["PCS"] = extract_prices_smart(" # ".join(rows['text'].tolist()))

        # Deteksi Harga CTN
        idx_ctn = df_ocr[df_ocr['text'].str.contains("CTN|KARTON|DUS", na=False)].index
        if not idx_ctn.empty:
            rows_ctn = df_ocr.iloc[idx_ctn[0] : idx_ctn[0] + 3]
            final_res["CTN"] = extract_prices_smart(" # ".join(rows_ctn['text'].tolist()))

    # --- Sensor Data Pribadi (Redaksi) ---
    draw = ImageDraw.Draw(original_pil)
    results_redact = reader.readtext(cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY), detail=1)
    for (bbox, text, prob) in results_redact:
        for kw in TEXTS_TO_REDACT:
            if fuzz.partial_ratio(kw.upper(), text.upper()) > 85:
                (tl, tr, br, bl) = bbox
                draw.rectangle([min(tl[0], bl[0])-5, min(tl[1], tr[1])-5, max(tr[0], br[0])+5, max(bl[1], br[1])+5], fill="white")
                break

    return final_res["PCS"], final_res["CTN"], scanned_name, original_pil

def compress_to_target(pil_img, target_kb):
    if pil_img.mode in ("RGBA", "P"): pil_img = pil_img.convert("RGB")
    quality = 95
    while quality > 10:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() / 1024 <= target_kb: return buf.getvalue()
        quality -= 5
    return buf.getvalue()

def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

# ================= UI STREAMLIT =================
st.title("📸 Price Check AI")

col1, col2, col3 = st.columns(3)
with col1: m_code_input = st.text_input("📍 Master Code").strip().upper()
with col2: date_input = st.text_input("📅 Tanggal (Ex: 23JAN2026)").strip().upper()
with col3: week_input = st.text_input("🗓️ Week").strip()

files = st.file_uploader("📂 Upload Foto", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code_input and date_input and week_input:
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # Load data dari Spreadsheet
        db_ig = conn.read(spreadsheet=URL_DB, worksheet=SHEET_MASTER_IG, ttl=0)
        db_ig.columns = db_ig.columns.astype(str).str.strip()
        
        db_targets = {}
        for s in SHEETS_TARGET:
            df_t = conn.read(spreadsheet=URL_DB, worksheet=s, ttl=0)
            df_t.columns = df_t.columns.astype(str).str.strip()
            db_targets[s] = df_t
            
        st.sidebar.success("✅ Database Terkoneksi")
            
    except Exception as e:
        st.error(f"Gagal memuat Spreadsheet. Pastikan link benar & tab 'IG' ada. Error: {e}")
        st.stop()

    if COL_IG_NAME in db_ig.columns:
        final_list = []
        zip_buffer = io.BytesIO()
        prog_bar = st.progress(0)
        status = st.empty()

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for idx, f in enumerate(files):
                status.text(f"⏳ Memproses {f.name}...")
                img_pil = Image.open(f)
                res_pcs, res_ctn, s_name, red_pil = process_ocr_all_prices(img_pil)
                
                # Matching Produk Fuzzy
                found_code, best_score = None, 0
                for _, row in db_ig.iterrows():
                    score = fuzz.token_set_ratio(str(row[COL_IG_NAME]).upper(), s_name)
                    if score > 80 and score > best_score:
                        best_score, found_code = score, norm(row["PRODCODE"])
                
                if found_code:
                    t_sheet, t_idx = None, None
                    for sn, df_t in db_targets.items():
                        match = df_t[(df_t["PRODCODE"].apply(norm) == found_code) & (df_t["MASTER Code"].apply(norm) == norm(m_code_input))]
                        if not match.empty:
                            t_sheet, t_idx = sn, match.index[0]
                            break
                    
                    if t_sheet:
                        final_list.append({
                            "prodcode": found_code, "sheet": t_sheet, "index": t_idx,
                            "n_pcs": res_pcs['normal'], "p_pcs": res_pcs['promo'],
                            "n_ctn": res_ctn['normal'], "p_ctn": res_ctn['promo']
                        })
                        
                        # Compress Gambar & ZIP
                        w, h = red_pil.size
                        crop = red_pil.crop((0, 75, w, h)) if h > 75 else red_pil
                        zip_file.writestr(f"{found_code}.jpg", compress_to_target(crop, TARGET_IMAGE_SIZE_KB))
                        
                        with st.expander(f"✅ Produk Cocok: {found_code}"):
                            st.write(f"Scanned: {s_name}")
                            st.json({"PCS": res_pcs, "CTN": res_ctn})
                            st.image(red_pil, width=300)
                prog_bar.progress((idx + 1) / len(files))

        status.success("Proses Selesai!")

        if final_list:
            st.write("### 📋 Ringkasan Data Scan")
            res_df = pd.DataFrame(final_list)
            st.table(res_df[["prodcode", "sheet", "n_pcs", "p_pcs", "n_ctn", "p_ctn"]])
            
            # --- TOMBOL UPDATE PERMANEN ---
            if st.button("🚀 SIMPAN PERMANEN KE GOOGLE SHEETS"):
                with st.spinner("Mengupdate data ke Google Sheets..."):
                    for s_name in SHEETS_TARGET:
                        df_up = db_targets[s_name]
                        for r in final_list:
                            if r['sheet'] == s_name:
                                i = r['index']
                                df_up.at[i, "Normal Competitor Price (Pcs)"] = r['n_pcs']
                                df_up.at[i, "Promo Competitor Price (Pcs)"] = r['p_pcs']
                                df_up.at[i, "Normal Competitor Price (Ctn)"] = r['n_ctn']
                                df_up.at[i, "Promo Competitor Price (Ctn)"] = r['p_ctn']
                        
                        # Menulis kembali ke Spreadsheet
                        conn.update(spreadsheet=URL_DB, worksheet=s_name, data=df_up)
                    st.success("✅ Database Google Sheets Berhasil Diupdate!")

            # Download Gambar ZIP
            zip_filename = f"{m_code_input}_{date_input}.zip"
            st.download_button("🖼️ DOWNLOAD ZIP JPG", zip_buffer.getvalue(), zip_filename)
    else:
        st.error(f"Kolom '{COL_IG_NAME}' tidak ditemukan di tab 'IG'.")

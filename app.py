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
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 
TARGET_IMAGE_SIZE_KB = 195 

# Daftar teks untuk sensor (Redact)
TEXTS_TO_REDACT = [
    "HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "Halo WAYAN GIYANTO / WRG", 
    "Halo MEMBER UMUM KLIK", "Halo DJUANMING / TK GOGO", "Halo NONOK JUNENGSIH", 
    "Halo AGUNG KURNIAWAN", "Halo ARIF RAMADHAN", "Halo HILMI ATIQ / WR DINDA"
]

st.set_page_config(page_title="Price Check", layout="wide")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False) 

reader = load_reader()

# ================= CORE OCR ENGINE =================
def process_ocr_all_prices(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_pil = pil_image.copy()

    # Pre-processing untuk akurasi harga
    img_resized = cv2.resize(img_np, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    processed = cv2.bilateralFilter(gray, 9, 75, 75)
    
    results = reader.readtext(processed, detail=1)

    data = []
    for (bbox, text, prob) in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        data.append({"text": text.upper(), "top": y_center})
    
    df_ocr = pd.DataFrame(data)
    final_res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    scanned_name = ""

    def clean_price(raw):
        trans = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'E': '8', 'G': '6', 'Z': '2', '+': '', 'A': '4'}
        text = re.sub(r'[\s.,\-]', '', raw)
        for char, digit in trans.items():
            text = text.replace(char, digit)
        nums = re.findall(r'\d{3,7}', text)
        return int(nums[0]) if nums else 0

    def extract_smart(text_content):
        text_content = re.sub(r'\(.*?\)|ISI\s*\d+', '', text_content)
        parts = text_content.split('RP')
        found = [clean_price(p.split('/')[0]) for p in parts[1:] if clean_price(p.split('/')[0]) > 0]
        if not found: return {"normal": 0, "promo": 0}
        return {"normal": found[0], "promo": found[1] if len(found) >= 2 else found[0]}

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        # Cari Nama Produk
        idx_search = df_ocr[df_ocr['text'].str.contains("CARI DI KLIK|SEMUA KATEGORI", na=False)].index
        if not idx_search.empty and (idx_search[-1] + 1) < len(df_ocr):
            scanned_name = df_ocr.iloc[idx_search[-1] + 1]['text']

        # Cari Harga PCS
        idx_pilih = df_ocr[df_ocr['text'].str.contains("PILIH SATUAN", na=False)].index
        if not idx_pilih.empty:
            area = df_ocr.iloc[idx_pilih[0]:]
            idx_pcs = area[area['text'].str.contains("PCS|RCG|BOX|PCK", na=False)].index
            if not idx_pcs.empty:
                rows = df_ocr.iloc[idx_pcs[0] : idx_pcs[0] + 3]
                final_res["PCS"] = extract_smart(" # ".join(rows['text'].tolist()))

        # Cari Harga CTN
        idx_ctn = df_ocr[df_ocr['text'].str.contains("CTN|KARTON|DUS", na=False)].index
        if not idx_ctn.empty:
            rows_ctn = df_ocr.iloc[idx_ctn[0] : idx_ctn[0] + 3]
            final_res["CTN"] = extract_smart(" # ".join(rows_ctn['text'].tolist()))

    # --- Auto-Redact ---
    draw = ImageDraw.Draw(original_pil)
    res_redact = reader.readtext(cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY), detail=1)
    for (bbox, text, prob) in res_redact:
        for kw in TEXTS_TO_REDACT:
            if fuzz.partial_ratio(kw.upper(), text.upper()) > 85:
                (tl, tr, br, bl) = bbox
                draw.rectangle([min(tl[0], bl[0])-5, min(tl[1], tr[1])-5, max(tr[0], br[0])+5, max(bl[1], br[1])+5], fill="white")
    
    return final_res["PCS"], final_res["CTN"], scanned_name, original_pil

def compress_img(pil_img, target_kb):
    if pil_img.mode in ("RGBA", "P"): pil_img = pil_img.convert("RGB")
    q = 95
    while q > 10:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=q, optimize=True)
        if buf.tell() / 1024 <= target_kb: return buf.getvalue()
        q -= 5
    return buf.getvalue()

# ================= UI STREAMLIT =================
def norm(val): return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("📸 Price Check AI")

c1, c2, c3 = st.columns(3)
with c1: m_code = st.text_input("📍 Master Code").strip().upper()
with c2: date_val = st.text_input("📅 Tanggal (23JAN2026)").strip().upper()
with c3: week_val = st.text_input("🗓️ Week").strip()

files = st.file_uploader("📂 Upload Foto Product", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code and date_val and week_val:
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        db_ig = conn.read(worksheet=SHEET_MASTER_IG, ttl="0")
        db_ig.columns = db_ig.columns.astype(str).str.strip()
        
        db_targets = {s: conn.read(worksheet=s, ttl="0") for s in SHEETS_TARGET}
        for s in SHEETS_TARGET: db_targets[s].columns = db_targets[s].columns.astype(str).str.strip()
    except Exception as e:
        st.error(f"Gagal koneksi Spreadsheet: {e}")
        st.stop()

    final_list = []
    zip_buffer = io.BytesIO()
    
    if COL_IG_NAME in db_ig.columns:
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_f:
            prog = st.progress(0)
            for idx, f in enumerate(files):
                img_pil = Image.open(f)
                res_pcs, res_ctn, s_name, red_pil = process_ocr_all_prices(img_pil)
                
                # Matching
                found_code, best_score = None, 0
                for _, row in db_ig.iterrows():
                    score = fuzz.token_set_ratio(str(row[COL_IG_NAME]).upper(), s_name)
                    if score > 80 and score > best_score:
                        best_score, found_code = score, norm(row["PRODCODE"])
                
                if found_code:
                    t_sheet, t_idx = None, None
                    for sn, df_t in db_targets.items():
                        m = df_t[(df_t["PRODCODE"].apply(norm) == found_code) & (df_t["MASTER Code"].apply(norm) == norm(m_code))]
                        if not m.empty:
                            t_sheet, t_idx = sn, m.index[0]
                            break
                    
                    if t_sheet:
                        final_list.append({
                            "prodcode": found_code, "sheet": t_sheet, "index": t_idx,
                            "n_pcs": res_pcs['normal'], "p_pcs": res_pcs['promo'],
                            "n_ctn": res_ctn['normal'], "p_ctn": res_ctn['promo']
                        })
                        zip_f.writestr(f"{found_code}.jpg", compress_img(red_pil, TARGET_IMAGE_SIZE_KB))
                        with st.expander(f"✅ {found_code} Match"):
                            st.image(red_pil, width=300)
                prog.progress((idx + 1) / len(files))

        if final_list:
            st.write("### 📋 Ringkasan")
            res_df = pd.DataFrame(final_list)
            st.table(res_df[["prodcode", "sheet", "n_pcs", "p_pcs", "n_ctn", "p_ctn"]])
            
            if st.button("🚀 UPDATE HARGA KE SPREADSHEET"):
                with st.spinner("Sedang mengupdate data permanen..."):
                    for s_name in SHEETS_TARGET:
                        df_up = db_targets[s_name]
                        for r in final_list:
                            if r['sheet'] == s_name:
                                i = r['index']
                                df_up.at[i, "Normal Competitor Price (Pcs)"] = r['n_pcs']
                                df_up.at[i, "Promo Competitor Price (Pcs)"] = r['p_pcs']
                                df_up.at[i, "Normal Competitor Price (Ctn)"] = r['n_ctn']
                                df_up.at[i, "Promo Competitor Price (Ctn)"] = r['p_ctn']
                        conn.update(worksheet=s_name, data=df_up)
                    st.success("✅ Database Terupdate!")

            st.download_button("📥 DOWNLOAD ZIP", zip_buffer.getvalue(), f"{m_code}_{date_val}.zip")

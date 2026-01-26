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

# Daftar Master sebagai "Jangkar" agar deteksi tetap akurat meski teks berantakan
PRODUCT_MASTER_LIST = [
    "KAPAL API KOPI BUBUK SPECIAL RCG", "DELMONTE KETCHUP SAUS TOMAT", "DELMONTE CHILLI SAUCE EXTRA HOT",
    "MAESTRO MAYONNAISE", "PRINGLES POTATO CRISPS ORIGINAL", "ALE-ALE JUICE ORANGE",
    "MOMOGI SNACK STICK JAGUNG BAKAR", "INDOFOOD BUMBU RACIK AYAM GORENG", "ROSE TEPUNG BERAS",
    "MINUTE MAID JUICE PULPY ORANGE", "TEH GELAS MINUMAN TEH ALAMI", "LUWAK WHITE KOFFIE ORIGINAL",
    "FLORIDINA JUICE PULP ORANGE", "ENERGEN CEREAL INSTANT VANILLA", "ROMA BISCUIT KELAPA",
    "AJINOMOTO PENYEDAP RASA MASAKO", "REGAL MARIE", "CHOKI CHOKI CHOCO CASHEW",
    "NABATI RICHEESE WAFER KRIM KEJU", "FRISIAN FLAG KENTAL MANIS", "INDOCAFE COFFEE MIX 3 IN 1",
    "POP ICE DRINK POWDER", "ABC KOPI INSTANT", "ULTRA SUSU CAIR UHT", "YOU C1000 HEALTH DRINK",
    "POCARI SWEAT MINUMAN ISOTONIK", "BLUE BAND MARGARINE SERBAGUNA", "INDOMIE MIE GORENG PLUS SPECIAL",
    "INDOMIE MIE INSTANT SOTO MIE", "POP MIE MI INSTAN LAPEER", "SEDAAP MIE MIE INSTANT",
    "3 AYAM MIE TELOR", "BANGO KECAP MANIS", "ULTRA TEH KOTAK JASMINE", "GOLDA COFFEE DRINK",
    "GOOD DAY KOPI INSTANT", "MILKU SUSU UHT", "BENG-BENG WAFER CHOCOLATE", "SUN KARA SANTAN KELAPA",
    "ROMA SANDWICH BETTER", "CHIKI SNACK BALLS AYAM", "ROYCO BUMBU PELEZAT", "TANGO WAFER CHOCOLATE",
    "LOTTE CHOCO PIE", "MR.P KACANG MADU", "SOSRO TEH BOTOL", "ABC KECAP MANIS", "SEGITIGA BIRU TEPUNG TERIGU",
    "KOMPAS TEPUNG TERIGU", "AJINOMOTO TEPUNG BUMBU SAJIKU", "TARO SNACK", "PANTHER MINUMAN SUPPLEMEN",
    "MAX TEA INSTANT DRINK", "YUPI CANDY GUMMY", "MILO HEALTY DRINK", "AQUA AIR MINERAL",
    "MEIJI BISCUIT HELLO PANDA", "SIDO MUNCUL JAMU TOLAK ANGIN", "MAMY POKO PANTS",
    "REXONA DEO LOTION", "PANTENE SHAMPOO", "PEPSODENT PASTA GIGI", "DOWNY SOFTENER",
    "SUNSILK SHAMPOO", "SUNLIGHT CAIRAN CUCI PIRING", "EKONOMI SABUN CREAM", "SO KLIN DETERGENT CAIR",
    "BAYGON INSEKTISIDA SPRAY", "LIFEBUOY SABUN MANDI", "MOLTO PELEMBUT", "MAMA LEMON",
    "RINSO DETERGEN", "NUVO SABUN KESEHATAN", "SO SOFT DETERJEN CAIR", "CAP LANG MINYAK KAYU PUTIH",
    "ZINC SHAMPOO", "POSH MEN DEO LOTION"
]

TEXTS_TO_REDACT = [
    "HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "WAYAN GIYANTO", 
    "MEMBER UMUM KLIK", "DJUANMING", "NONOK JUNENGSIH", 
    "AGUNG KURNIAWAN", "ARIF RAMADHAN", "HILMI ATIQ"
]

st.set_page_config(page_title="Price Check V11.5", layout="wide")

def clean_price_robust(text_segment):
    """Menghapus noise dan mengambil angka terakhir (harga promo)"""
    clean_txt = re.sub(r'[^\d]', '', str(text_segment))
    nums = re.findall(r'\d{3,7}', clean_txt)
    return int(nums[-1]) if nums else 0

def process_ocr_indogrosir(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config)
    
    df_ocr = pd.DataFrame(d)
    df_ocr['text'] = df_ocr['text'].fillna('').str.upper()
    
    # Gabungkan baris untuk scan teks mentah
    df_ocr['line_id'] = df_ocr['block_num'].astype(str) + "_" + df_ocr['line_num'].astype(str)
    lines_df = df_ocr.groupby('line_id').agg({'text': lambda x: " ".join(x), 'top': 'min', 'height': 'max'}).reset_index()
    full_text_raw = " ".join(df_ocr['text'].tolist())
    
    final_res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    prod_name = "N/A"

    # --- 1. DETEKSI NAMA PRODUK DENGAN MASTER LIST ---
    best_match_score = 0
    for master_name in PRODUCT_MASTER_LIST:
        score = fuzz.partial_ratio(master_name.upper(), full_text_raw)
        if score > 80 and score > best_match_score:
            best_match_score = score
            prod_name = master_name

    # --- 2. DETEKSI HARGA PCS & CTN ---
    # Regex ini mencari angka setelah 'PCS - RP' atau 'CTN - RP' meski teks berantakan
    pcs_match = re.search(r"PCS\s*-\s*RP\s*([\d\.\-\s,]+?)(?=\s|ISI|/|PCS|CTN|$)", full_text_raw)
    if pcs_match:
        val = clean_price_robust(pcs_match.group(1))
        final_res["PCS"] = {"normal": val, "promo": val}

    ctn_match = re.search(r"CTN\s*-\s*RP\s*([\d\.\-\s,]+?)(?=\s|ISI|/|PCS|CTN|$)", full_text_raw)
    if ctn_match:
        val = clean_price_robust(ctn_match.group(1))
        final_res["CTN"] = {"normal": val, "promo": val}

    # --- 3. LOGIKA REDAKSI (SENSOR) ---
    draw = ImageDraw.Draw(pil_image)
    for _, row in lines_df.iterrows():
        line_txt = str(row['text']).upper()
        for kw in TEXTS_TO_REDACT:
            if fuzz.partial_ratio(kw, line_txt) > 75:
                y = row['top'] / scale
                h = row['height'] / scale
                draw.rectangle([0, y - 8, pil_image.width, y + h + 8], fill="white")
                break 

    return final_res["PCS"], final_res["CTN"], prod_name, full_text_raw, pil_image

# ================= UI STREAMLIT =================
def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("📸 Price Check V11.5 - Master Keywords")

c1, c2, c3 = st.columns(3)
with c1: m_code = st.text_input("📍 Master Code").upper()
with c2: date_inp = st.text_input("📅 Tanggal (26JAN2026)").upper()
with c3: week_inp = st.text_input("🗓️ Week")

files = st.file_uploader("📂 Upload Foto", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code and date_inp and week_inp:
    if os.path.exists(FILE_PATH):
        db_ig = pd.read_excel(FILE_PATH, sheet_name=SHEET_MASTER_IG)
        db_ig.columns = db_ig.columns.astype(str).str.strip()
        db_targets = {s: pd.read_excel(FILE_PATH, sheet_name=s) for s in SHEETS_TARGET}

        final_list = []
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                with st.expander(f"🔍 Scan: {f.name}", expanded=True):
                    img_pil = Image.open(f)
                    pcs, ctn, name, raw, red_img = process_ocr_indogrosir(img_pil)
                    
                    # Fuzzy Match ke Database IG
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_n = str(row[COL_IG_NAME]).upper()
                        score = fuzz.token_set_ratio(db_n, name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])
                    
                    col_img, col_info = st.columns([1, 1.2])
                    with col_img: st.image(red_img)
                    with col_info:
                        st.write(f"**Produk:** `{name}`")
                        st.write(f"**Match:** `{match_code}` (Score: {best_score})")
                        st.json({"PCS": pcs, "CTN": ctn})
                        st.text_area("Debug Raw Text", value=raw, height=100)

                    if match_code:
                        for s_name, df_t in db_targets.items():
                            df_t.columns = df_t.columns.astype(str).str.strip()
                            match_row = df_t[(df_t["PRODCODE"].astype(str).apply(norm) == match_code) & 
                                             (df_t["MASTER Code"].astype(str).apply(norm) == norm(m_code))]
                            
                            if not match_row.empty:
                                final_list.append({
                                    "prodcode": match_code, "sheet": s_name, "index": match_row.index[0],
                                    "n_pcs": pcs['normal'], "p_pcs": pcs['promo'],
                                    "n_ctn": ctn['normal'], "p_ctn": ctn['promo']
                                })
                                if red_img.mode != "RGB": red_img = red_img.convert("RGB")
                                buf = io.BytesIO()
                                red_img.save(buf, format="JPEG", quality=85)
                                zf.writestr(f"{match_code}.jpg", buf.getvalue())
                                break
                gc.collect()

        if final_list:
            st.divider()
            if st.button("🚀 EKSEKUSI UPDATE DATABASE EXCEL"):
                wb = load_workbook(FILE_PATH)
                for r in final_list:
                    ws = wb[r['sheet']]
                    headers = [str(c.value).strip() for c in ws[1]]
                    row_num = r['index'] + 2
                    mapping = {
                        "Normal Competitor Price (Pcs)": r['n_pcs'], "Promo Competitor Price (Pcs)": r['p_pcs'],
                        "Normal Competitor Price (Ctn)": r['n_ctn'], "Promo Competitor Price (Ctn)": r['p_ctn']
                    }
                    for col_name, val in mapping.items():
                        if col_name in headers: ws.cell(row=row_num, column=headers.index(col_name) + 1).value = val
                wb.save(FILE_PATH)
                st.success("✅ Database diperbarui!")
                
                with open(FILE_PATH, "rb") as f:
                    st.download_button("📥 DOWNLOAD EXCEL", f, f"Report_{date_inp}.xlsx")
            
            st.download_button("🖼️ DOWNLOAD ZIP FOTO", zip_buffer.getvalue(), f"{m_code}_{date_inp}.zip")

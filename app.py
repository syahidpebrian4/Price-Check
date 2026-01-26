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

st.set_page_config(page_title="PRICE CHECK", layout="wide")

def clean_price_val(raw_str):
    if not raw_str: return 0
    clean = re.sub(r'[.,\-]', '', raw_str)
    try:
        return int(clean)
    except:
        return 0

def process_ocr_spatial_v2(pil_image):
    # 1. Image Preprocessing
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. Get OCR Data
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
    df = pd.DataFrame(d)
    df = df[df['text'].str.strip() != ""]
    df['text'] = df['text'].str.upper()

    # 3. Double Sorting (Top then Left)
    df = df.sort_values(by=['top', 'left'])
    lines = []
    if not df.empty:
        current_line_top = df.iloc[0]['top']
        temp_words = []
        for _, row in df.iterrows():
            if row['top'] - current_line_top > 15:
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

    # 4. Extraction
    final_res = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    prod_name = "N/A"
    promo_text = ""

    # -- Prod Name --
    best_match_score = 0
    for m_name in PRODUCT_MASTER_LIST:
        score = fuzz.partial_ratio(m_name.upper(), single_line_text)
        if score > 80 and score > best_match_score:
            best_match_score = score
            prod_name = m_name

    # -- Unit Price (PCS/RCG/BOX) --
    # Regex ini lebih kuat: Mencari Unit -> Angka Harga 1 -> Angka Harga 2 -> Sebelum / ISI
    unit_match = re.search(r"(PCS|RCG|BOX).*?RP\s*([\d\-\.,]+).*?RP\s*([\d\-\.,]+).*?/\s*ISI", single_line_text)
    if unit_match:
        final_res["PCS"]["normal"] = clean_price_val(unit_match.group(2))
        final_res["PCS"]["promo"] = clean_price_val(unit_match.group(3))
    else:
        # Fallback Single Price
        unit_match_s = re.search(r"(PCS|RCG|BOX).*?RP\s*([\d\-\.,]+).*?/\s*ISI", single_line_text)
        if unit_match_s:
            val = clean_price_val(unit_match_s.group(2))
            final_res["PCS"] = {"normal": val, "promo": val}

    # -- Carton Price (CTN) --
    ctn_match = re.search(r"CTN.*?RP\s*([\d\-\.,]+).*?RP\s*([\d\-\.,]+).*?/\s*ISI", single_line_text)
    if ctn_match:
        final_res["CTN"]["normal"] = clean_price_val(ctn_match.group(1))
        final_res["CTN"]["promo"] = clean_price_val(ctn_match.group(2))

    # -- Promo Logic (Between | and RAP) --
    if "MAU LEBIH UNTUNG" in single_line_text:
        # Mencari pipa | yang diikuti teks dan diakhiri RAP
        promo_match = re.search(r"[\|I1]\s*(.*?RAP)", single_line_text, re.DOTALL)
        if promo_match:
            promo_text = promo_match.group(1).strip()
            # Bersihkan jika ada simbol sampah di awal teks yang terambil
            promo_text = re.sub(r'^[^A-Z0-9]+', '', promo_text)

    # 5. Redaction
    draw = ImageDraw.Draw(pil_image)
    for _, row in df.iterrows():
        for kw in TEXTS_TO_REDACT:
            if fuzz.partial_ratio(kw, row['text']) > 80:
                x, y, w, h = row['left']/scale, row['top']/scale, row['width']/scale, row['height']/scale
                draw.rectangle([x-2, y-2, x+w+2, y+h+2], fill="white")
    
    return final_res["PCS"], final_res["CTN"], prod_name, clean_full_text, pil_image, promo_text

# ================= UI STREAMLIT =================
def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

st.title("PRICE CHECK")

c1, c2, c3 = st.columns(3)
with c1: m_code = st.text_input("📍 MASTER CODE").upper()
with c2: date_inp = st.text_input("📅 TANGGAL (Contoh: 26 JAN)").upper()
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
                    pcs, ctn, name, raw_text, red_img, p_desc = process_ocr_spatial_v2(img_pil)
                    
                    # Prodcode Mapping
                    match_code, best_score = None, 0
                    for _, row in db_ig.iterrows():
                        db_n = str(row[COL_IG_NAME]).upper()
                        score = fuzz.token_set_ratio(db_n, name)
                        if score > 70 and score > best_score:
                            best_score, match_code = score, norm(row["PRODCODE"])
                    
                    st.subheader(f"📄 Scan: {f.name}")
                    col_img, col_info = st.columns([1, 1.2])
                    with col_img: st.image(red_img)
                    with col_info:
                        st.markdown(f"### {name}")
                        st.write(f"Match Code: `{match_code}`")
                        st.info(f"**PROMOSI:** {p_desc if p_desc else '-'}")
                        m1, m2 = st.columns(2)
                        m1.metric("UNIT Normal", f"Rp {pcs['normal']:,}")
                        m1.metric("UNIT Promo", f"Rp {pcs['promo']:,}")
                        m2.metric("CTN Normal", f"Rp {ctn['normal']:,}")
                        m2.metric("CTN Promo", f"Rp {ctn['promo']:,}")

                    with st.expander("📄 LIHAT HASIL SCAN TERSTRUKTUR"):
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
                        "Promo Competitor Price (Ctn)": r['p_ctn'],
                        "Promosi Competitor": r['p_desc']
                    }
                    for col_name, val in mapping.items():
                        if col_name in headers:
                            ws.cell(row=row_num, column=headers.index(col_name) + 1).value = val
                wb.save(FILE_PATH)
                st.success("DATABASE BERHASIL DIUPDATE!")
                with open(FILE_PATH, "rb") as f:
                    st.download_button("📥 DOWNLOAD REPORT", f, f"Report_{date_inp}.xlsx")
            st.download_button("🖼️ DOWNLOAD ZIP FOTO", zip_buffer.getvalue(), f"{m_code}_{date_inp}.zip")

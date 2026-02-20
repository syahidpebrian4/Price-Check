import streamlit as st
import pytesseract
import cv2
import numpy as np
import pandas as pd
import re
import os
import io
import zipfile
import time
import gc
import base64
from PIL import Image, ImageDraw
from fuzzywuzzy import fuzz
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from openpyxl import load_workbook

# ================= CONFIG & PATHS =================
DB_PATH = "database/master_harga.xlsx"
SHEET_MASTER_IG = "IG"
COL_IG_NAME = "PRODNAME_IG"

st.set_page_config(page_title="Price Check", layout="wide", initial_sidebar_state="expanded")

# --- HELPER: LOGO BASE64 ---
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

# --- CSS CUSTOM HEADER ---
logo_b64 = get_base64_image("lotte_logo.png")
st.markdown(f"""
    <style>
        .custom-header {{
            position: fixed;
            top: 0; left: 0; width: 100%; height: 90px;
            background-color: white;
            display: flex; align-items: center;
            padding: 0 30px; border-bottom: 3px solid #eeeeee;
            z-index: 999999;
        }}
        .header-logo {{ height: 55px; margin-right: 25px; }}
        .header-title {{
            font-size: 32px; font-weight: 900;
            font-family: 'Arial Black', sans-serif; color: black; margin: 0;
        }}
        .main .block-container {{ padding-top: 130px !important; }}
        [data-testid="stSidebar"] {{ background-color: #FF0000 !important; margin-top: 90px !important; }}
        [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {{ color: white !important; }}
        header {{ visibility: hidden; }}
    </style>
    <div class="custom-header">
        <img src="data:image/png;base64,{logo_b64 if logo_b64 else ''}" class="header-logo">
        <h1 class="header-title">PRICE CHECK</h1>
    </div>
""", unsafe_allow_html=True)

# ================= FUNGSI LOGIKA 1: OCR =================
def process_ocr_minimal(pil_image, master_product_names=None):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
    df_ocr = pd.DataFrame(d)
    df_ocr = df_ocr[df_ocr['text'].str.strip() != ""]
    df_ocr['text'] = df_ocr['text'].str.upper()

    df_ocr = df_ocr.sort_values(by=['top', 'left'])
    lines_data = []
    if not df_ocr.empty:
        current_top = df_ocr.iloc[0]['top']
        temp_words = []
        for _, row in df_ocr.iterrows():
            if row['top'] - current_top > 15:
                temp_words.sort(key=lambda x: x['left'])
                lines_data.append({
                    "text": " ".join([w['text'] for w in temp_words]),
                    "top": current_top,
                    "h": max([w['height'] for w in temp_words])
                })
                temp_words = [{'text': row['text'], 'left': row['left'], 'height': row['height']}]
                current_top = row['top']
            else:
                temp_words.append({'text': row['text'], 'left': row['left'], 'height': row['height']})
        lines_data.append({"text": " ".join([w['text'] for w in temp_words]), "top": current_top, "h": 10})

    lines_txt = [l['text'] for l in lines_data]
    full_text_single = " # ".join(lines_txt)
    draw = ImageDraw.Draw(pil_image)

    anchor_nav = "SEMUA KATEGORI"
    for i, line in enumerate(lines_txt):
        if fuzz.partial_ratio(anchor_nav, line) > 65:
            y_coord = lines_data[i]['top'] / scale
            if y_coord < (pil_image.height * 0.3):
                h_box = min(lines_data[i]['h'] / scale, 40)
                draw.rectangle([0, y_coord - 5, pil_image.width, y_coord + h_box + 5], fill="white")
                break

    prod_name = "N/A"
    if master_product_names:
        best_match, highest_score = "N/A", 0
        for ref_name in master_product_names:
            score = fuzz.partial_ratio(str(ref_name).upper(), full_text_single)
            if score > 80 and score > highest_score:
                highest_score, best_match = score, str(ref_name).upper()
        prod_name = best_match

    return prod_name, pil_image

# ================= FUNGSI LOGIKA 2: SCRAPER =================
def clean_price(teks):
    if not teks or teks == "0": return 0
    return int(re.sub(r'[^\d]', '', str(teks)))

def extract_product_name(html):
    match = re.search(r'<meta\s+property="og:title"\s+content="(.*?)"', html)
    return match.group(1).split('|')[0].strip() if match else "Nama Tidak Ditemukan"

def extract_promo_text(html):
    promo_pattern = r'class="promo-list[^>]*>.*?<ul[^>]*>(.*?)</ul>'
    match = re.search(promo_pattern, html, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
        clean_text = re.sub(r'<[^>]+>', '', content).strip()
        return re.sub(r'\s+', ' ', clean_text)
    return "-"

def extract_price_by_unit(unit_list, html):
    for unit in unit_list:
        promo_pattern = fr'{unit}\s*-\s*<span.*?line-through.*?>Rp([\d\.,]+)</span>\s*<span.*?red.*?>Rp([\d\.,]+)</span>'
        promo_match = re.search(promo_pattern, html, re.DOTALL | re.IGNORECASE)
        if promo_match: return unit, promo_match.group(1), promo_match.group(2), "Promo"
        
        reguler_pattern = fr'{unit}\s*-\s*Rp\s*([\d\.,]+)'
        reguler_match = re.search(reguler_pattern, html, re.IGNORECASE)
        if reguler_match:
            price = reguler_match.group(1)
            return unit, price, price, "Reguler"
    return "N/A", "0", "0", "N/A"

def update_excel_database(results_df, master_code):
    if not os.path.exists(DB_PATH): return False
    try:
        df_ig = pd.read_excel(DB_PATH, sheet_name="IG", dtype={'PRODCODE': str})
        wb = load_workbook(DB_PATH)
        ws_df = wb["DF"]
        header_row = 3
        headers = {cell.value: cell.column for cell in ws_df[header_row] if cell.value}
        
        success_count = 0
        for _, row in results_df.iterrows():
            match_ig = df_ig[df_ig['PRODNAME_IG'] == row['Nama Produk']]
            if not match_ig.empty:
                prod_code_target = str(match_ig.iloc[0]['PRODCODE']).strip()
                for r in range(5, ws_df.max_row + 1):
                    cell_mc = ws_df.cell(row=r, column=headers["MASTER CODE"]).value
                    cell_pc = ws_df.cell(row=r, column=headers["PRODCODE"]).value
                    if str(cell_mc).strip() == str(master_code).strip() and str(cell_pc).strip() == prod_code_target:
                        ws_df.cell(row=r, column=headers["Promosi Competitor"]).value = row['Mekanisme Promo']
                        ws_df.cell(row=r, column=headers["Normal Competitor Price (Pcs)"]).value = row['Satuan Normal']
                        ws_df.cell(row=r, column=headers["Promo Competitor Price (Pcs)"]).value = row['Satuan Promo']
                        ws_df.cell(row=r, column=headers["Normal Competitor Price (Ctn)"]).value = row['CTN Normal']
                        ws_df.cell(row=r, column=headers["Promo Competitor Price (Ctn)"]).value = row['CTN Promo']
                        success_count += 1
                        break
        wb.save(DB_PATH)
        return success_count
    except Exception as e:
        st.error(f"Error Update Excel: {e}")
        return 0

# ================= MAIN APP NAVIGATION =================
with st.sidebar:
    st.title("MENU UTAMA")
    menu = st.radio("Pilih Fitur:", ["📸 Image", "🏷️ Price"])
    st.divider()

if menu == "📸 Image":
    st.subheader("📸 Image")
    with st.sidebar:
        m_code = st.text_input("📍 MASTER CODE:", placeholder="6001").upper()
        date_inp = st.text_input("🗓️ DATE:", placeholder="01JAN2026").upper()

    files = st.file_uploader("UPLOAD GAMBAR", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if files and m_code and date_inp:
        if os.path.exists(DB_PATH):
            db_ig = pd.read_excel(DB_PATH, sheet_name=SHEET_MASTER_IG)
            list_nama_master = db_ig[COL_IG_NAME].dropna().unique().tolist()
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    with st.container(border=True):
                        img_pil = Image.open(f)
                        name, red_img = process_ocr_minimal(img_pil, list_nama_master)
                        match_code = None
                        match_row = db_ig[db_ig[COL_IG_NAME].str.upper() == name]
                        if not match_row.empty:
                            match_code = str(match_row.iloc[0]["PRODCODE"]).replace(".0","")

                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.image(red_img, caption=f"Identified: {name}", width=350)
                        with c2:
                            if match_code:
                                st.success(f"Code: {match_code}")
                                buf = io.BytesIO()
                                red_img.convert("RGB").save(buf, format="JPEG")
                                zf.writestr(f"{match_code}_{date_inp}.jpg", buf.getvalue())
                            else:
                                st.warning("Code Not Found")
            
            if zip_buffer.getvalue():
                st.download_button("🖼️ DOWNLOAD ALL (ZIP)", zip_buffer.getvalue(), f"{m_code}_{date_inp}.zip", use_container_width=True)
        else:
            st.error("Database Excel tidak ditemukan!")

elif menu == "🏷️ Price":
    st.subheader("🏷️ Price")
    with st.sidebar:
        mc_sync = st.text_input("📍 MASTER CODE:", placeholder="06001")
        date_inp_p = st.text_input("🗓️ DATE_P:", placeholder="01JAN2026").upper()
        week_inp = st.text_input("📅 WEEK:", placeholder="1").upper()

    urls_area = st.text_area("Paste URLs (satu per baris):", height=200)

    if st.button("🚀 Jalankan Scraper"):
        if not urls_area or not mc_sync:
            st.error("Lengkapi URL dan Master Code!")
        else:
            # --- KONFIGURASI JENDELA CHROME TERBUKA (NON-HEADLESS) ---
            chrome_options = Options()
            # Bagian ini saya hapus --headless agar jendela muncul
            chrome_options.add_argument("--start-maximized") # Langsung layar penuh
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            try:
                # Driver manager otomatis mendownload driver yang cocok dengan Chrome Anda
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)
                
                all_results = []
                list_urls = [u.strip() for u in urls_area.split('\n') if u.strip()]
                prog = st.progress(0)
                
                for i, url in enumerate(list_urls):
                    driver.get(url)
                    time.sleep(5) # Memberi waktu Anda melihat jendela yang terbuka
                    html = driver.page_source
                    nama = extract_product_name(html)
                    promo_txt = extract_promo_text(html)
                    _, ctn_n, ctn_p, _ = extract_price_by_unit(["CTN"], html)
                    _, sat_n, sat_p, _ = extract_price_by_unit(["PCS", "PCK", "RCG", "BOX"], html)
                    
                    all_results.append({
                        "Nama Produk": nama, "Mekanisme Promo": promo_txt,
                        "Satuan Normal": clean_price(sat_n), "Satuan Promo": clean_price(sat_p),
                        "CTN Normal": clean_price(ctn_n), "CTN Promo": clean_price(ctn_p)
                    })
                    prog.progress((i + 1) / len(list_urls))
                
                df_scrape = pd.DataFrame(all_results)
                st.table(df_scrape)
                
                count = update_excel_database(df_scrape, mc_sync)
                if count > 0:
                    st.success(f"🔥 Berhasil update {count} baris ke Excel!")
                    with open(DB_PATH, "rb") as f:
                        st.download_button("📥 DOWNLOAD HASIL EXCEL", f, f"PRICE_CHECK_W{week_inp}_{date_inp_p}.xlsx", use_container_width=True)
                else:
                    st.warning("Data tidak ditemukan di database.")
                
                # Jendela akan menutup otomatis setelah selesai. 
                # Hapus baris di bawah jika ingin jendela tetap terbuka.
                driver.quit() 
                
            except Exception as e:
                st.error(f"Error saat membuka Jendela Chrome: {e}")

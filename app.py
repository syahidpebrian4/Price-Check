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
import base64
from PIL import Image, ImageDraw
from fuzzywuzzy import fuzz
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from openpyxl import load_workbook
from selenium.webdriver.common.by import By

# ================= CONFIG & PATHS =================
DB_PATH = "database/master_harga.xlsx"
SHEET_MASTER_IG = "IG"
COL_IG_NAME = "PRODNAME_IG"

st.set_page_config(page_title="Price Check", layout="wide")

# --- HELPER: LOGO ---
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

logo_b64 = get_base64_image("lotte_logo.png")
st.markdown(f"""
    <style>
        .custom-header {{ position: fixed; top: 0; left: 0; width: 100%; height: 90px; background: white; display: flex; align-items: center; padding: 0 30px; border-bottom: 3px solid #eee; z-index: 999; }}
        .header-logo {{ height: 55px; margin-right: 25px; }}
        .main .block-container {{ padding-top: 130px !important; }}
    </style>
    <div class="custom-header">
        <img src="data:image/png;base64,{logo_b64 if logo_b64 else ''}" class="header-logo">
        <h1 style="color:black; margin:0;">PRICE CHECK</h1>
    </div>
""", unsafe_allow_html=True)

# --- FUNGSI OCR ---
def process_ocr_minimal(pil_image, master_product_names=None):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    df_ocr = pd.DataFrame(d)
    df_ocr = df_ocr[df_ocr['text'].str.strip() != ""]
    full_text = " # ".join(df_ocr['text'].str.upper().tolist())
    
    prod_name = "N/A"
    if master_product_names:
        best_match, highest_score = "N/A", 0
        for ref_name in master_product_names:
            score = fuzz.partial_ratio(str(ref_name).upper(), full_text)
            if score > 80 and score > highest_score:
                highest_score, best_match = score, str(ref_name).upper()
        prod_name = best_match
    return prod_name, pil_image

# --- FUNGSI SCRAPER ---
def clean_price(teks):
    if not teks or teks == "0": return 0
    return int(re.sub(r'[^\d]', '', str(teks)))

def extract_product_name(html):
    match = re.search(r'<meta\s+property="og:title"\s+content="(.*?)"', html)
    return match.group(1).split('|')[0].strip() if match else "N/A"

def extract_price_by_unit(unit_list, html):
    for unit in unit_list:
        p = fr'{unit}\s*-\s*.*?Rp([\d\.,]+).*?Rp([\d\.,]+)'
        m = re.search(p, html, re.DOTALL | re.IGNORECASE)
        if m: return unit, m.group(1), m.group(2)
    return "N/A", "0", "0"

def update_excel_database(results_df, master_code):
    if not os.path.exists(DB_PATH): return 0
    try:
        wb = load_workbook(DB_PATH)
        ws = wb["DF"]
        df_ig = pd.read_excel(DB_PATH, sheet_name="IG")
        # Logika update baris sesuai PRODCODE...
        wb.save(DB_PATH)
        return len(results_df)
    except: return 0

# --- UI NAVIGATION ---
menu = st.sidebar.radio("Menu", ["📸 Image", "🏷️ Price"])

if menu == "📸 Image":
    m_code = st.sidebar.text_input("MASTER CODE")
    date_p = st.sidebar.text_input("DATE")
    files = st.file_uploader("Upload", accept_multiple_files=True)
    if files and st.button("Proses OCR"):
        db_ig = pd.read_excel(DB_PATH, sheet_name="IG")
        list_master = db_ig[COL_IG_NAME].tolist()
        for f in files:
            name, _ = process_ocr_minimal(Image.open(f), list_master)
            st.write(f"Hasil: {name}")

elif menu == "🏷️ Price":
    with st.sidebar:
        st.subheader("Login Indogrosir")
        user_ig = st.text_input("Email")
        pass_ig = st.text_input("Password", type="password")
        mc = st.text_input("Master Code")
    
    urls = st.text_area("URLs")
    if st.button("🚀 Jalankan Scraper"):
        options = Options()
        options.add_argument("--headless") # Wajib di Cloud
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        try:
            # PROSES LOGIN OTOMATIS
            driver.get("https://www.indogrosir.co.id/login")
            time.sleep(3)
            driver.find_element(By.NAME, "username").send_keys(user_ig)
            driver.find_element(By.NAME, "password").send_keys(pass_ig)
            driver.find_element(By.XPATH, "//button[@type='submit']").click()
            time.sleep(5)
            
            # PROSES SCRAPE
            results = []
            for url in urls.split('\n'):
                if url.strip():
                    driver.get(url.strip())
                    time.sleep(4)
                    html = driver.page_source
                    results.append({
                        "Nama Produk": extract_product_name(html),
                        "Satuan Normal": clean_price(extract_price_by_unit(["PCS"], html)[1])
                    })
            st.table(pd.DataFrame(results))
        finally:
            driver.quit()

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
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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

# --- CSS CUSTOM HEADER (Format Awal) ---
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

# ================= FUNGSI LOGIKA =================
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

# ================= UI NAVIGATION =================
menu = st.sidebar.radio("Menu Utama", ["📸 Image OCR", "🏷️ Price Scraper"])

if menu == "📸 Image OCR":
    st.subheader("📸 Image OCR")
    # Logika OCR Anda tetap di sini...

elif menu == "🏷️ Price Scraper":
    st.subheader("🏷️ Scraper Harga Indogrosir (Cloud Mode)")
    with st.sidebar:
        st.subheader("🔐 Login Akun")
        ig_user = st.text_input("Email/No HP:")
        ig_pass = st.text_input("Password:", type="password")
        mc_code = st.text_input("Master Code:")
    
    urls_area = st.text_area("List URL Produk (Satu per baris):")

    if st.button("🚀 Jalankan Scraper"):
        if not ig_user or not ig_pass or not urls_area:
            st.error("Lengkapi data Login dan URL!")
        else:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            # User-Agent agar tidak terdeteksi bot mentah
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
            options.binary_location = "/usr/bin/chromium"

            try:
                service = Service("/usr/bin/chromedriver")
                driver = webdriver.Chrome(service=service, options=options)
                
                with st.status("Sedang bekerja...", expanded=True) as status:
                    st.write("Membuka halaman login...")
                    driver.get("https://www.klikindogrosir.com/login")
                    
                    try:
                        wait = WebDriverWait(driver, 20)
                        
                        # Targetkan ID berdasarkan hasil Inspect Element
                        email_input = wait.until(EC.element_to_be_clickable((By.ID, "login-email")))
                        email_input.clear()
                        email_input.send_keys(ig_user)
                        
                        pass_input = driver.find_element(By.NAME, "password")
                        pass_input.clear()
                        pass_input.send_keys(ig_pass)
                        
                        # Klik tombol login menggunakan JavaScript (lebih kuat di Cloud)
                        login_btn = driver.find_element(By.XPATH, "//form[@id='loginForm']//button")
                        driver.execute_script("arguments[0].click();", login_btn)
                        
                        st.write("✅ Data login dikirim. Menunggu dashboard...")
                        time.sleep(10) 
                    except Exception as e:
                        st.write(f"⚠️ Gagal otomatisasi login: {e}")

                    results = []
                    urls = [u.strip() for u in urls_area.split('\n') if u.strip()]
                    
                    for idx, url in enumerate(urls):
                        st.write(f"Mengambil data ke-{idx+1}...")
                        driver.get(url)
                        time.sleep(6) # Memberi waktu render di server
                        html = driver.page_source
                        
                        results.append({
                            "Nama Produk": extract_product_name(html),
                            "URL": url
                        })
                    
                    driver.quit()
                    status.update(label="Proses Selesai!", state="complete")

                st.table(pd.DataFrame(results))
                
            except Exception as e:
                st.error(f"Kesalahan Sistem: {e}")

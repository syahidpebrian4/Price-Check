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

# ================= INITIAL CHECK =================
# Cek apakah folder database ada agar tidak langsung Crash
if not os.path.exists("database"):
    os.makedirs("database")

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
        <h1 style="color:black; margin:0; font-family: sans-serif;">PRICE CHECK</h1>
    </div>
""", unsafe_allow_html=True)

# --- FUNGSI LOGIKA ---
def clean_price(teks):
    if not teks or teks == "0": return 0
    return int(re.sub(r'[^\d]', '', str(teks)))

def extract_product_name(html):
    match = re.search(r'<meta\s+property="og:title"\s+content="(.*?)"', html)
    return match.group(1).split('|')[0].strip() if match else "N/A"

# --- UI ---
menu = st.sidebar.radio("Menu", ["📸 Image", "🏷️ Price"])

if menu == "📸 Image":
    st.subheader("📸 Image OCR Process")
    if not os.path.exists(DB_PATH):
        st.error(f"File {DB_PATH} tidak ditemukan di GitHub!")
    else:
        m_code = st.sidebar.text_input("MASTER CODE")
        files = st.file_uploader("Upload Gambar", accept_multiple_files=True)
        if files and st.button("Jalankan OCR"):
            st.info("Sedang memproses...")

elif menu == "🏷️ Price":
    st.subheader("🏷️ Price Scraper (Cloud Mode)")
    with st.sidebar:
        user_ig = st.text_input("User Indogrosir")
        pass_ig = st.text_input("Pass Indogrosir", type="password")
        mc_sync = st.text_input("Master Code")
    
    urls_area = st.text_area("Paste URLs (per baris)")
    
    if st.button("🚀 Jalankan Scraper"):
        if not user_ig or not pass_ig:
            st.error("Masukkan User & Pass Indogrosir!")
        else:
            # KONFIGURASI DRIVER CLOUD
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            
            try:
                with st.spinner("Menyiapkan Chrome di Server..."):
                    service = Service(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=options)
                
                # LOGIN PROCESS
                st.info("Mencoba login...")
                driver.get("https://www.indogrosir.co.id/login")
                time.sleep(3)
                
                try:
                    driver.find_element(By.NAME, "username").send_keys(user_ig)
                    driver.find_element(By.NAME, "password").send_keys(pass_ig)
                    driver.find_element(By.XPATH, "//button[@type='submit']").click()
                    time.sleep(5)
                    st.success("Sesi Login Aktif")
                except:
                    st.warning("Tombol login tidak ditemukan, mencoba lanjut scraping...")

                # SCRAPING PROCESS
                results = []
                list_urls = [u.strip() for u in urls_area.split('\n') if u.strip()]
                prog = st.progress(0)
                
                for i, url in enumerate(list_urls):
                    driver.get(url)
                    time.sleep(4)
                    html = driver.page_source
                    results.append({
                        "Nama Produk": extract_product_name(html),
                        "URL": url
                    })
                    prog.progress((i + 1) / len(list_urls))
                
                st.table(pd.DataFrame(results))
                driver.quit()
                
            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {str(e)}")

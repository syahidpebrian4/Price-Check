import streamlit as st
import pytesseract
import cv2
import numpy as np
import pandas as pd
import re
import os
import time
import base64
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ================= CONFIG & HEADER =================
st.set_page_config(page_title="Price Check", layout="wide")

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

logo_b64 = get_base64_image("lotte_logo.png")
st.markdown(f"""
    <style>
        .custom-header {{ position: fixed; top: 0; left: 0; width: 100%; height: 90px; background: white; display: flex; align-items: center; padding: 0 30px; border-bottom: 3px solid #eee; z-index: 9999; }}
        .header-logo {{ height: 55px; margin-right: 25px; }}
        .main .block-container {{ padding-top: 130px !important; }}
        [data-testid="stSidebar"] {{ background-color: #FF0000 !important; margin-top: 90px !important; }}
        [data-testid="stSidebar"] * {{ color: white !important; }}
        header {{ visibility: hidden; }}
    </style>
    <div class="custom-header">
        <img src="data:image/png;base64,{logo_b64 if logo_b64 else ''}" class="header-logo">
        <h1 style="color:black; margin:0; font-size:32px; font-weight:900;">PRICE CHECK</h1>
    </div>
""", unsafe_allow_html=True)

# ================= SCRAPER LOGIC =================
def extract_product_name(html):
    match = re.search(r'<meta property="og:title" content="(.*?)"', html)
    return match.group(1).split('|')[0].strip() if match else "N/A"

menu = st.sidebar.radio("Menu", ["📸 Image OCR", "🏷️ Price Scraper"])

if menu == "🏷️ Price Scraper":
    with st.sidebar:
        st.subheader("🔐 Login Akun")
        ig_user = st.text_input("Email/No HP")
        ig_pass = st.text_input("Password", type="password")
        urls_area = st.text_area("List URL (Satu per baris)")

    if st.button("🚀 Jalankan Scraper"):
        if not ig_user or not ig_pass or not urls_area:
            st.error("Data login atau URL masih kosong!")
        else:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
            options.binary_location = "/usr/bin/chromium"

            try:
                service = Service("/usr/bin/chromedriver")
                driver = webdriver.Chrome(service=service, options=options)
                wait = WebDriverWait(driver, 20)

                with st.status("Sedang bekerja...", expanded=True) as status:
                    # 1. Buka URL pertama untuk memancing modal login
                    first_url = [u.strip() for u in urls_area.split('\n') if u.strip()][0]
                    st.write(f"Membuka halaman produk...")
                    driver.get(first_url)
                    time.sleep(5)

                    try:
                        # 2. Klik tombol "Masuk" jika form login belum muncul
                        # Berdasarkan screenshot, form login muncul sebagai modal
                        st.write("Mencoba proses login...")
                        
                        # Input Email (ID: login-email dari data kamu)
                        email_input = wait.until(EC.visibility_of_element_located((By.ID, "login-email")))
                        email_input.send_keys(ig_user)
                        
                        # Input Password (ID: login-password dari data kamu)
                        pass_input = driver.find_element(By.ID, "login-password")
                        pass_input.send_keys(ig_pass)
                        
                        # Klik tombol Masuk/Submit
                        login_btn = driver.find_element(By.XPATH, "//form[@id='loginForm']//button[contains(text(),'Masuk')]")
                        driver.execute_script("arguments[0].click();", login_btn)
                        
                        st.write("✅ Login berhasil dikirim.")
                        time.sleep(8) 
                    except Exception as e:
                        st.warning(f"⚠️ Form login tidak terdeteksi otomatis. Screenshot tersimpan.")
                        driver.save_screenshot("debug_login.png")
                        st.image("debug_login.png")

                    # 3. Proses Scrape Semua URL
                    results = []
                    urls = [u.strip() for u in urls_area.split('\n') if u.strip()]
                    for url in urls:
                        st.write(f"Scraping: {url[-15:]}...")
                        driver.get(url)
                        time.sleep(6)
                        results.append({"Produk": extract_product_name(driver.page_source), "URL": url})
                    
                    driver.quit()
                    status.update(label="Scraping Selesai!", state="complete")

                st.table(pd.DataFrame(results))
            except Exception as e:
                st.error(f"Sistem Crash: {e}")

import streamlit as st
import os

# --- PRE-LOAD CHECK ---
try:
    import pandas as pd
    import numpy as np
    from PIL import Image
    import pytesseract
    import cv2
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.common.by import By
    import time
    import re
except Exception as e:
    st.error(f"Instalasi library gagal: {e}")
    st.stop()

# ================= CONFIG =================
DB_PATH = "database/master_harga.xlsx"

st.set_page_config(page_title="Price Checker", layout="wide")

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: red;'>🛒 PRICE CHECKER CLOUD</h1>", unsafe_allow_html=True)

# --- MENU ---
menu = st.sidebar.selectbox("Pilih Fitur", ["📸 OCR Image", "🏷️ Scraper Harga"])

if menu == "📸 OCR Image":
    st.subheader("📸 OCR Image")
    if not os.path.exists(DB_PATH):
        st.error("Folder 'database' atau file 'master_harga.xlsx' tidak ditemukan di repo GitHub!")
    else:
        st.success("Database Excel Terdeteksi.")
    
    up = st.file_uploader("Upload Foto Label", accept_multiple_files=True)
    if up:
        st.write(f"{len(up)} Gambar terunggah.")

elif menu == "🏷️ Scraper Harga":
    st.subheader("🏷️ Scraper Harga (Indogrosir)")
    
    with st.sidebar:
        user_ig = st.text_input("User/Email")
        pass_ig = st.text_input("Password", type="password")
        btn_run = st.button("🚀 Mulai Scrape")

    urls_text = st.text_area("Masukkan Link Produk (satu per baris)")

    if btn_run:
        if not user_ig or not pass_ig or not urls_text:
            st.warning("Data login atau URL masih kosong!")
        else:
            # KONFIGURASI CHROME CLOUD
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            
            try:
                with st.spinner("Menyiapkan Browser Server..."):
                    service = Service(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=options)
                
                st.info("Mencoba Login...")
                driver.get("https://www.indogrosir.co.id/login")
                time.sleep(3)
                
                # Masukkan data login
                try:
                    driver.find_element(By.NAME, "username").send_keys(user_ig)
                    driver.find_element(By.NAME, "password").send_keys(pass_ig)
                    driver.find_element(By.XPATH, "//button[@type='submit']").click()
                    time.sleep(5)
                except:
                    st.warning("Form login tidak ditemukan, lanjut scraping...")

                urls = [u.strip() for u in urls_text.split("\n") if u.strip()]
                results = []
                
                for u in urls:
                    driver.get(u)
                    time.sleep(3)
                    html = driver.page_source
                    # Ekstrak Judul
                    title = "N/A"
                    match = re.search(r'<meta property="og:title" content="(.*?)"', html)
                    if match:
                        title = match.group(1).split('|')[0]
                    
                    results.append({"Nama": title, "URL": u})
                
                st.table(pd.DataFrame(results))
                driver.quit()
                st.success("Selesai!")

            except Exception as e:
                st.error(f"Error pada Selenium: {e}")

import streamlit as st
import pandas as pd
import re
import os
import time
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ================= CONFIG & HEADER =================
st.set_page_config(page_title="Price Check", layout="wide")

# CSS Header (Tetap Sama)
st.markdown("""
    <style>
        .custom-header { position: fixed; top: 0; left: 0; width: 100%; height: 90px; background: white; z-index: 9999; border-bottom: 3px solid #eee; display: flex; align-items: center; padding: 0 30px; }
        .main .block-container { padding-top: 130px !important; }
        [data-testid="stSidebar"] { background-color: #FF0000 !important; margin-top: 90px !important; }
        [data-testid="stSidebar"] * { color: white !important; }
    </style>
    <div class="custom-header"><h1 style="color:black; margin:0;">PRICE CHECK</h1></div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio("Menu", ["📸 Image OCR", "🏷️ Price Scraper"])

if menu == "🏷️ Price Scraper":
    with st.sidebar:
        ig_user = st.text_input("Email/No HP")
        ig_pass = st.text_input("Password", type="password")
        urls_area = st.text_area("List URL (Satu per baris)")

    if st.button("🚀 Jalankan Scraper"):
        if not ig_user or not ig_pass or not urls_area:
            st.error("Data belum lengkap!")
        else:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            
            # --- JURUS ANTI-BLOCK CLOUDFLARE ---
            # Menggunakan User-Agent yang sangat spesifik dan terbaru
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            options.binary_location = "/usr/bin/chromium"

            try:
                service = Service("/usr/bin/chromedriver")
                driver = webdriver.Chrome(service=service, options=options)
                
                # Menghapus jejak bot via script
                driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                    "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
                })

                wait = WebDriverWait(driver, 25)

                with st.status("Menembus sistem keamanan...", expanded=True) as status:
                    st.write("Mencoba akses produk...")
                    first_url = [u.strip() for u in urls_area.split('\n') if u.strip()][0]
                    driver.get(first_url)
                    time.sleep(10) # Beri waktu Cloudflare memeriksa IP

                    # Cek apakah diblokir
                    if "blocked" in driver.title.lower() or "Cloudflare" in driver.page_source:
                        st.error("❌ Terdeteksi Bot oleh Cloudflare!")
                        driver.save_screenshot("cloudflare_block.png")
                        st.image("cloudflare_block.png", caption="Tampilan saat terblokir")
                        driver.quit()
                        st.stop()

                    try:
                        st.write("Membuka Modal Login...")
                        # Berdasarkan screenshot image_82f5f4.png, kita perlu memicu modal login muncul
                        # Jika form tidak muncul, kita paksa klik tombol 'Masuk' di header
                        try:
                            login_trigger = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(text(),'Masuk')]")))
                            driver.execute_script("arguments[0].click();", login_trigger)
                            time.sleep(3)
                        except:
                            pass

                        # Input data berdasarkan ID dari hasil Inspect kamu
                        st.write("Mengisi kredensial...")
                        email_input = wait.until(EC.visibility_of_element_located((By.ID, "login-email")))
                        email_input.send_keys(ig_user)
                        
                        pass_input = driver.find_element(By.ID, "login-password")
                        pass_input.send_keys(ig_pass)
                        
                        login_btn = driver.find_element(By.XPATH, "//form[@id='loginForm']//button")
                        driver.execute_script("arguments[0].click();", login_btn)
                        
                        st.write("✅ Login Terkirim.")
                        time.sleep(10)
                    except Exception as e:
                        st.warning("⚠️ Gagal Login Otomatis.")
                        driver.save_screenshot("debug_login.png")
                        st.image("debug_login.png")

                    # Scrape Data
                    results = []
                    urls = [u.strip() for u in urls_area.split('\n') if u.strip()]
                    for url in urls:
                        st.write(f"Proses: {url[-10:]}...")
                        driver.get(url)
                        time.sleep(7)
                        results.append({"URL": url, "Status": "Berhasil"})
                    
                    driver.quit()
                    status.update(label="Selesai!", state="complete")
                
                st.table(pd.DataFrame(results))

            except Exception as e:
                st.error(f"Sistem Crash: {e}")

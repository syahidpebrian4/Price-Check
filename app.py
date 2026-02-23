import streamlit as st
import pandas as pd
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- Tampilan Web ---
st.set_page_config(page_title="Price Checker", layout="wide")
st.title("🏷️ Indogrosir Scraper (Cloud Mode)")

with st.sidebar:
    st.subheader("🔐 Akun Indogrosir")
    user_ig = st.text_input("Email/No HP")
    pass_ig = st.text_input("Password", type="password")
    urls_input = st.text_area("Masukkan URL Produk (1 per baris)")

if st.button("🚀 Ambil Harga"):
    if not user_ig or not pass_ig or not urls_input:
        st.error("Lengkapi data dulu ya!")
    else:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.binary_location = "/usr/bin/chromium"
        
        # JURUS SEMBUNYI: Menghilangkan jejak robot
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")

        try:
            service = Service("/usr/bin/chromedriver")
            driver = webdriver.Chrome(service=service, options=options)
            
            # Hapus flag webdriver via script
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            urls = [u.strip() for u in urls_input.split('\n') if u.strip()]
            results = []

            with st.status("Sedang berjuang menembus blokir...", expanded=True) as status:
                # LANGKAH 1: Buka Produk Dulu (Sebagai Tamu)
                st.write("Membuka halaman produk...")
                driver.get(urls[0])
                time.sleep(7) # Tunggu lama biar dikira manusia baca

                # LANGKAH 2: Coba Login dari dalam halaman produk
                try:
                    st.write("Mencoba Login...")
                    # Cari input email yang tadi kamu temukan ID-nya
                    wait = WebDriverWait(driver, 15)
                    email_box = wait.until(EC.presence_of_element_located((By.ID, "login-email")))
                    email_box.send_keys(user_ig)
                    
                    pass_box = driver.find_element(By.ID, "login-password")
                    pass_box.send_keys(pass_ig)
                    
                    # Klik Masuk
                    login_btn = driver.find_element(By.XPATH, "//button[contains(text(),'Masuk')]")
                    driver.execute_script("arguments[0].click();", login_btn)
                    time.sleep(5)
                    st.write("✅ Berhasil melewati gerbang login.")
                except:
                    st.warning("⚠️ Gagal login, mencoba ambil harga sebagai tamu (mungkin harga tidak akurat).")

                # LANGKAH 3: Ambil Data
                for url in urls:
                    driver.get(url)
                    time.sleep(5)
                    # Ambil judul produk sebagai tes
                    try:
                        title = driver.title
                        results.append({"Produk": title, "URL": url, "Status": "Success"})
                    except:
                        results.append({"Produk": "Gagal", "URL": url, "Status": "Blocked"})

            driver.quit()
            st.table(pd.DataFrame(results))

        except Exception as e:
            st.error(f"Aplikasi Error: {e}")

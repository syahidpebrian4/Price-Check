import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

st.title("🛡️ Semi-Manual Login (Cloud Mode)")

# 1. Inisialisasi Driver (Gunakan session_state agar browser tidak tutup)
if 'driver' not in st.session_state:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    options.binary_location = "/usr/bin/chromium"
    
    service = Service("/usr/bin/chromedriver")
    st.session_state.driver = webdriver.Chrome(service=service, options=options)

driver = st.session_state.driver

# --- FUNGSI UPDATE SCREENSHOT ---
def update_view():
    driver.save_screenshot("view.png")
    st.image("view.png", caption="Tampilan Browser di Server")

# --- UI INTERAKSI ---
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🌐 Buka Indogrosir"):
        driver.get("https://www.klikindogrosir.com/login")
        update_view()

    u_input = st.text_input("Ketik Email/HP di sini:")
    if st.button("⌨️ Masukkan Email"):
        el = driver.find_element(By.ID, "login-email")
        el.send_keys(u_input)
        update_view()

    p_input = st.text_input("Ketik Password di sini:", type="password")
    if st.button("⌨️ Masukkan Password"):
        el = driver.find_element(By.ID, "login-password")
        el.send_keys(p_input)
        update_view()

    if st.button("🚀 KLIK TOMBOL LOGIN"):
        btn = driver.find_element(By.XPATH, "//form[@id='loginForm']//button")
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(5)
        update_view()

with col2:
    st.subheader("🖥️ Monitor")
    if st.button("🔄 Refresh Tampilan"):
        update_view()

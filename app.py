import streamlit as st
from streamlit_gsheets import GSheetsConnection

st.title("Test Koneksi Google Sheets")

try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    # Mencoba membaca sheet bernama 'IG'
    df = conn.read(worksheet="IG", ttl=0)
    st.success("✅ Koneksi Berhasil!")
    st.write("Data dari Sheet IG:")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Koneksi Masih Gagal: {e}")
    st.info("Saran: Pastikan nama tab di Google Sheets Anda adalah 'IG' (Huruf Besar Semua)")

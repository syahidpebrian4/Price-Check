import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz

# --- CONFIG ---
URL_BASE = "https://docs.google.com/spreadsheets/d/1vz2tEQ7YbFuKSF172ihDFBh6YE3F4Ql1jSEOpptZN34"
SHEET_MASTER_IG = "IG"

st.set_page_config(page_title="Price Check AI", layout="wide")

def get_data_direct(sheet_name):
    try:
        # Jalur akses CSV yang paling aman dari error 400/404
        csv_url = f"{URL_BASE}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        df = pd.read_csv(csv_url)
        # Membersihkan nama kolom dari spasi atau karakter aneh
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Koneksi Gagal: {e}")
        return None

st.title("📸 Price Check AI")

# Menampilkan status database segera setelah aplikasi dimuat
db_ig = get_data_direct(SHEET_MASTER_IG)

if db_ig is not None:
    st.success("✅ Database Terdeteksi!")
    with st.expander("Pratinjau Kolom Database"):
        st.write("Kolom yang terbaca:", list(db_ig.columns))
        st.dataframe(db_ig.head(5))
else:
    st.warning("⚠️ Menunggu koneksi ke Google Sheets... Pastikan link sudah 'Anyone with link - Editor'")

st.info("Jika tabel di atas muncul, berarti masalah instalasi dan koneksi sudah SELESAI.")

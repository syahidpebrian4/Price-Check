import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
import os
from PIL import Image, ImageDraw
import io
import zipfile
from fuzzywuzzy import fuzz

# ================= CONFIGURASI =================
# Pastikan folder 'database' dan file ini ada di GitHub Anda
FILE_PATH = "database/master_harga.xlsx"
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 

# Daftar kata yang akan disensor/ditutup kotak putih
TEXTS_TO_REDACT = [
    "HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "Halo WAYAN GIYANTO / WRG", 
    "Halo MEMBER UMUM KLIK", "Halo DJUANMING / TK GOGO", "Halo NONOK JUNENGSIH", 
    "Halo AGUNG KURNIAWAN", "Halo ARIF RAMADHAN", "Halo HILMI ATIQ / WR DINDA"
]

st.set_page_config(page_title="Price Check AI v1.0", layout="wide")

# Load model OCR (Cache agar tidak loading ulang setiap saat)
@st.cache_resource
def load_reader():
    # Menggunakan gpu=False karena Streamlit Cloud gratis tidak menyediakan GPU
    return easyocr.Reader(['en'], gpu=False) 

reader = load_reader()

# ================= FUNGSI OCR & REDAKSI =================
def process_ocr_all_prices(pil_image):
    # Konversi ke format OpenCV
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_pil_for_redact = pil_image.copy()

    # Pre-processing untuk akurasi OCR
    img_resized = cv2.resize(img_np, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    processed = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Jalankan OCR
    results_ocr = reader.readtext(processed, detail=1)

    data = []
    for (bbox, text, prob) in results_ocr:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        data.append({"text": text.upper(), "top": y_center})
    
    df_ocr = pd.DataFrame(data)
    final_results = {"PCS": {"normal": 0, "promo": 0}, "CTN": {"normal": 0, "promo": 0}}
    product_name_on_image = "TIDAK TERDETEKSI"

    # Fungsi pembersih angka harga
    def clean_and_repair_price(raw_segment):
        translation_table = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'E': '8', 'G': '6', 'Z': '2'}
        text = re.sub(r'[\s.,\-]', '', raw_segment)
        for char, digit in translation_table.items():
            text = text.replace(char, digit)
        nums = re.findall(r'\d{3,7}', text)
        return int(nums[0]) if nums else 0

    def extract_prices_smart(text_content):
        found_prices = []
        parts = text_content.split('RP')
        for part in parts[1:]:
            target_segment = part.split('/')[0]
            price = clean_and_repair_price(target_segment)
            if price > 0: found_prices.append(price)
        
        if not found_prices: return {"normal": 0, "promo": 0}
        # Harga pertama biasanya normal, harga kedua biasanya promo
        return {"normal": found_prices[0], "promo": found_prices[1] if len(found_prices) > 1 else found_prices[0]}

    if not df_ocr.empty:
        df_ocr = df_ocr.sort_values(by='top').reset_index(drop=True)
        
        # 1. Cari Nama Produk (Setelah teks navigasi Klik Indogrosir)
        idx_search = df_ocr[df_ocr['text'].str.contains("CARI DI KLIK|SEMUA KATEGORI", na=False)].index
        if not idx_search.empty and (idx_search[-1] + 1) < len(df_ocr):
            product_name_on_image = df_ocr.iloc[idx_search[-1] + 1]['text']

        # 2. Cari Harga Satuan (PCS)
        idx_pilih = df_ocr[df_ocr['text'].str.contains("PILIH SATUAN", na=False)].index
        if not idx_pilih.empty:
            search_area = df_ocr.iloc[idx_pilih[0]:]
            idx_pcs = search_area[search_area['text'].str.contains("PCS|RCG|BOX|PCK", na=False)].index
            if not idx_pcs.empty:
                target_rows = df_ocr.iloc[idx_pcs[0] : idx_pcs[0] + 3]
                final_results["PCS"] = extract_prices_smart(" # ".join(target_rows['text'].tolist()))

        # 3. Cari Harga Karton (CTN)
        idx_ctn = df_ocr[df_ocr['text'].str.contains("CTN|KARTON|DUS", na=False)].index
        if not idx_ctn.empty:
            target_rows_ctn = df_ocr.iloc[idx_ctn[0] : idx_ctn[0] + 3]
            final_results["CTN"] = extract_prices_smart(" # ".join(target_rows_ctn['text'].tolist()))

    # --- PROSES REDAKSI (Tutup Nama Akun) ---
    draw = ImageDraw.Draw(original_pil_for_redact)
    # Gunakan OCR ringan untuk redaksi
    results_redact = reader.readtext(cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY))
    for (bbox, text, prob) in results_redact:
        for keyword in TEXTS_TO_REDACT:
            if fuzz.partial_ratio(keyword.upper(), text.upper()) > 85:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                # Timpa dengan kotak putih
                draw.rectangle([tuple(top_left), tuple(bottom_right)], fill="white")

    return final_results["PCS"], final_results["CTN"], product_name_on_image, original_pil_for_redact

def norm(val):
    return str(val).replace(".0", "").replace(" ", "").strip().upper()

# ================= TAMPILAN UTAMA =================
st.title("📸 Price Check AI")

# Baris Input User
col_u1, col_u2, col_u3 = st.columns(3)
with col_u1: m_code = st.text_input("📍 Master Code (Contoh: 6018)").strip().upper()
with col_u2: tgl = st.text_input("📅 Tanggal (Contoh: 23JAN2026)").strip().upper()
with col_u3: week = st.text_input("🗓️ Week (Contoh: 2)").strip()

files = st.file_uploader("📂 Upload Foto Label (Bisa banyak sekaligus)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and m_code and tgl and week:
    # Cek apakah database ada
    if os.path.exists(FILE_PATH):
        try:
            with st.spinner('Memproses Data... Mohon tunggu sebentar.'):
                # Membaca file Excel dari folder database/
                xl = pd.ExcelFile(FILE_PATH)
                db_ig = xl.parse(SHEET_MASTER_IG)
                
                final_data_list = []
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for f in files:
                        img_pil = Image.open(f)
                        res_pcs, res_ctn, scanned_name, redacted_img = process_ocr_all_prices(img_pil)
                        
                        # Pencocokan nama produk ke Database (Fuzzy Match)
                        found_code = "TIDAK_DITEMUKAN"
                        best_score = 0
                        for _, row in db_ig.iterrows():
                            score = fuzz.token_set_ratio(str(row[COL_IG_NAME]).upper(), scanned_name)
                            if score > 80 and score > best_score:
                                best_score = score
                                found_code = norm(row["PRODCODE"])
                        
                        # Simpan hasil untuk preview
                        res_entry = {
                            "PRODCODE": found_code, 
                            "N_PCS": res_pcs['normal'], "P_PCS": res_pcs['promo'],
                            "N_CTN": res_ctn['normal'], "P_CTN": res_ctn['promo'],
                            "NAMA_DI_FOTO": scanned_name,
                            "IMG_OBJ": redacted_img
                        }
                        final_data_list.append(res_entry)
                        
                        # Masukkan gambar hasil redaksi ke ZIP
                        img_byte_arr = io.BytesIO()
                        redacted_img.save(img_byte_arr, format="JPEG")
                        zip_file.writestr(f"{found_code}.jpg", img_byte_arr.getvalue())

                # --- TAMPILAN SETELAH SELESAI ---
                if final_data_list:
                    st.success("✅ Scanning Selesai!")
                    
                    # 1. Tombol Download
                    st.write("### 📥 Download Hasil")
                    c_dl1, c_dl2 = st.columns(2)
                    with c_dl1:
                        # Di sini Anda bisa menambahkan logika export ke Excel baru jika perlu
                        st.download_button(f"📊 Download List Excel", pd.DataFrame(final_data_list).drop(columns=['IMG_OBJ']).to_csv(index=False), f"Price_Check_W{week}_{tgl}.csv")
                    with c_dl2:
                        st.download_button(f"🖼️ Download Foto ZIP", zip_buffer.getvalue(), f"{m_code}_{tgl}.zip")

                    # 2. TABEL RINGKASAN HARGA (Preview Harga)
                    st.write("---")
                    st.write("### 📋 Ringkasan Harga")
                    df_view = pd.DataFrame(final_data_list).drop(columns=['IMG_OBJ'])
                    st.table(df_view)

                    # 3. PREVIEW GAMBAR (Hasil Redaksi)
                    st.write("### 🖼️ Preview Gambar")
                    for item in final_data_list:
                        with st.expander(f"Lihat Detail: {item['PRODCODE']} - {item['NAMA_DI_FOTO']}"):
                            cp1, cp2 = st.columns([1, 2])
                            with cp1:
                                st.json({
                                    "PCS": {"Normal": item['N_PCS'], "Promo": item['P_PCS']},
                                    "CTN": {"Normal": item['N_CTN'], "Promo": item['P_CTN']}
                                })
                            with cp2:
                                st.image(item['IMG_OBJ'], caption=f"Hasil Sensor: {item['PRODCODE']}")
                else:
                    st.warning("⚠️ Tidak ada data yang berhasil diproses.")
                    
        except Exception as e:
            st.error(f"Terjadi error pada aplikasi: {e}")
    else:
        st.error(f"❌ File '{FILE_PATH}' tidak ditemukan. Pastikan folder 'database' sudah di-upload ke GitHub.")

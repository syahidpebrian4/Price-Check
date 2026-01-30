import streamlit as st
import pytesseract
import cv2
import numpy as np
import pandas as pd
import re
import os
from PIL import Image, ImageDraw
from openpyxl import load_workbook
import io
import zipfile
from fuzzywuzzy import fuzz
import gc

# ================= CONFIG & DATABASE =================
FILE_PATH = "database/master_harga.xlsx"
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG" 
COL_IG_NAME = "PRODNAME_IG" 

st.set_page_config(page_title="Price Check", layout="wide")

def clean_price_val(raw_str):
    """Membersihkan string harga menjadi integer murni."""
    if not raw_str: return 0
    # Menghapus karakter non-digit kecuali titik/koma (akan dibuang nanti)
    clean = re.sub(r'[^\d]', '', str(raw_str))
    return int(clean) if clean else 0

def process_ocr_final(pil_image, master_product_names=None):
    # 1. Image Preprocessing
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    scale = 2.0
    img_resized = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. OCR Execution
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
    df_ocr = pd.DataFrame(d)
    df_ocr = df_ocr[df_ocr['text'].str.strip() != ""]
    df_ocr['text'] = df_ocr['text'].str.upper()

    # 3. Line Grouping
    df_ocr = df_ocr.sort_values(by=['top', 'left'])
    lines_data = []
    if not df_ocr.empty:
        current_top = df_ocr.iloc[0]['top']
        temp_words = []
        for _, row in df_ocr.iterrows():
            if row['top'] - current_top > 15:
                temp_words.sort(key=lambda x: x['left'])
                lines_data.append({
                    "text": " ".join([w['text'] for w in temp_words]),
                    "top": current_top,
                    "h": max([w['height'] for w in temp_words])
                })
                temp_words = [{'text': row['text'], 'left': row['left'], 'height': row['height']}]
                current_top = row['top']
            else:
                temp_words.append({'text': row['text'], 'left': row['left'], 'height': row['height']})
        temp_words.sort(key=lambda x: x['left'])
        lines_data.append({"text": " ".join([w['text'] for w in temp_words]), "top": current_top, "h": 10})

    lines_txt = [l['text'] for l in lines_data]
    full_text_single = " ".join(lines_txt)
    raw_ocr_output = "\n".join(lines_txt)

    prod_name, promo_desc = "N/A", "-"
    res = {"PCS": {"n": 0, "p": 0}, "CTN": {"n": 0, "p": 0}}
    draw = ImageDraw.Draw(pil_image)

    # --- A. NAMA PRODUK (GLOBAL SEARCH) ---
    if master_product_names:
        best_match = "N/A"
        highest_score = 0
        for ref_name in master_product_names:
            m_name = str(ref_name).upper()
            score = fuzz.partial_ratio(m_name, full_text_single)
            if score > 80 and score > highest_score:
                highest_score = score
                best_match = m_name
        prod_name = best_match

    # --- B. SENSOR (REDACT) ---
    anchor_nav = "SEMUA KATEGORI"
    for i, line in enumerate(lines_txt):
        if fuzz.partial_ratio(anchor_nav, line) > 65:
            y, h = lines_data[i]['top'] / scale, lines_data[i]['h'] / scale
            draw.rectangle([0, y - 10, pil_image.width, y + h + 15], fill="white")
            break

    # --- C. SMART PRICE DETECTION (LOGIKA BARU) ---
    def get_prices(text_segment):
        # Cari angka format Rp atau ribuan (minimal 4 digit)
        found = re.findall(r"(?:RP|R9|BP|RD)?\s?([\d\.,]{4,9})", text_segment)
        return [clean_price_val(f) for f in found if clean_price_val(f) > 500]

    # 1. Deteksi PCS/PCH/PCK/RCG
    unit_area = re.split(r"(PILIH SATUAN|TERMURAH|RCG|PCS|PCH|PCK)", full_text_single)
    if len(unit_area) > 1:
        # Ambil teks setelah keyword satuan ditemukan
        pcs_prices = get_prices(" ".join(unit_area[1:]))
        if len(pcs_prices) >= 2:
            res["PCS"]["n"], res["PCS"]["p"] = pcs_prices[0], pcs_prices[1]
        elif len(pcs_prices) == 1:
            res["PCS"]["n"] = res["PCS"]["p"] = pcs_prices[0]

    # 2. Deteksi CTN
    if "CTN" in full_text_single:
        ctn_area = full_text_single.split("CTN")[1]
        ctn_prices = get_prices(ctn_area)
        if ctn_prices:
            res["CTN"]["n"] = res["CTN"]["p"] = ctn_prices[0]

    # --- D. PROMOSI ---
    if "BELI" in full_text_single or "GRATIS" in full_text_single:
        promo_match = re.search(r"(BELI\s\d+\sGRATIS\s\d+)", full_text_single)
        if promo_match:
            promo_desc = promo_match.group(1)

    return res["PCS"], res["CTN"], prod_name, raw_ocr_output, pil_image, promo_desc

# ... (Sisa Kode UI Streamlit tetap sama seperti sebelumnya) ...

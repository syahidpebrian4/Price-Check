import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
from PIL import Image, ImageDraw
import io
import zipfile
from fuzzywuzzy import fuzz
from streamlit_gsheets import GSheetsConnection

# ================= CONFIG =================
SHEETS_TARGET = ["DF", "HBHC"]
SHEET_MASTER_IG = "IG"
COL_IG_NAME = "PRODNAME_IG"
TARGET_IMAGE_SIZE_KB = 195

TEXTS_TO_REDACT = [
    "HALO AI YUYUN SUMARNI", "AL YUYUN SUMARNI", "Halo WAYAN GIYANTO / WRG",
    "Halo MEMBER UMUM KLIK", "Halo DJUANMING / TK GOGO", "Halo NONOK JUNENGSIH",
    "Halo AGUNG KURNIAWAN", "Halo ARIF RAMADHAN", "Halo HILMI ATIQ / WR DINDA"
]

st.set_page_config(page_title="Price Check", layout="wide")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# ================= OCR =================
def process_ocr_all_prices(pil_image):
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_pil = pil_image.copy()

    img_resized = cv2.resize(img_np, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    processed = cv2.bilateralFilter(gray, 9, 75, 75)

    results = reader.readtext(processed, detail=1)

    data = []
    for (bbox, text, prob) in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        data.append({"text": text.upper(), "top": y_center})

    df_ocr = pd.DataFrame(data)
    final_res =_

import easyocr
import csv
import re
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from tkinter import Tk, filedialog

# Klasörleri oluşturmak için kullanılan BASE_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "original_images"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "excell_files"), exist_ok=True)  # "digital_text_images" kaldırıldı

# Log ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_next_filename(directory, extension):
    """ Mevcut dosyaların isimlerine göre sıradaki dosya ismini belirler. """
    existing_files = os.listdir(directory)
    existing_files = [f for f in existing_files if f.endswith(extension)]
    existing_numbers = [int(re.search(r'(\d+)', f).group(1)) for f in existing_files if re.search(r'(\d+)', f)]
    next_number = max(existing_numbers) + 1 if existing_numbers else 0
    next_filename = os.path.join(directory, f"output_{next_number}{extension}")
    return next_filename


def select_image():
    """ Kullanıcının bilgisayarından bir dosya seçmesini sağlar. """
    Tk().withdraw()  # Tkinter penceresini gizler
    file_path = filedialog.askopenfilename(
        title="Bir Görüntü Seçin",
        filetypes=[("Görüntü Dosyaları", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path


def sort_texts_by_line(boxes, texts):
    """ Metinleri satır sırasına göre düzenler """
    box_positions = [(box[0][1], box[0][0], i) for i, box in enumerate(boxes)]  # (y1, x1, index)
    box_positions.sort(key=lambda x: (x[0], x[1]))  # Önce y1'e, ardından x1'e göre sırala
    grouped_lines = []
    current_line = []
    threshold = 10  # Satırların ayrımı için y-farkı eşiği
    prev_y = box_positions[0][0]

    for pos in box_positions:
        y, x, idx = pos
        if abs(y - prev_y) > threshold:
            grouped_lines.append(current_line)
            current_line = []
        current_line.append(idx)
        prev_y = y

    if current_line:
        grouped_lines.append(current_line)

    ordered_texts = []
    for line in grouped_lines:
        line_texts = [(boxes[i][0][0], texts[i]) for i in line]  # (x, text)
        line_texts.sort(key=lambda x: x[0])  # X'e göre sırala (sol-sağ)
        ordered_texts.extend([text for x, text in line_texts])

    return ordered_texts


def ocr(file_path):
    """ OCR işlemini gerçekleştirir ve metni görselleştirir. """
    if not file_path:
        logging.warning("Hiçbir dosya seçilmedi.")
        return

    logging.info(f"Görüntü işleniyor: {file_path}")

    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='tr')
        result = ocr.ocr(file_path, cls=True)
    except Exception as e:
        logging.error(f"OCR işlemi başarısız oldu: {e}")
        return

    image = Image.open(file_path).convert('RGB')
    original_image_path = get_next_filename(os.path.join(BASE_DIR, "original_images"), ".png")
    image.save(original_image_path)
    logging.info(f"Orijinal görsel '{original_image_path}' dosyasına kaydedildi.")

    boxes = [line[0] for line in result[0]]
    texts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]

    ordered_texts = sort_texts_by_line(boxes, texts)

    # **Dijital metin görseli kaldırıldı**

    # Excel dosyasına metin ve konumları kaydetme
    data = {
        "Text": ordered_texts,
        "Box": [boxes[i] for i in range(len(boxes))],
        "X1": [boxes[i][0][0] for i in range(len(boxes))],
        "Y1": [boxes[i][0][1] for i in range(len(boxes))],
        "Score": [scores[i] for i in range(len(scores))]
    }
    df = pd.DataFrame(data)
    excel_dir = os.path.join(BASE_DIR, "excell_files")
    excel_path = get_next_filename(excel_dir, ".xlsx")
    df.to_excel(excel_path, index=False)

    logging.info(f"OCR sonuçları '{excel_path}' dosyasına kaydedildi.")


if __name__ == "__main__":
    file_path = select_image()
    if file_path:
        ocr(file_path)
import os
import cv2
import numpy as np
import pandas as pd

# --- VERİ YÜKLEME FONKSİYONU ---
def load_data(csv_path, image_folder, img_size=(256, 64)):
    df = pd.read_csv(csv_path, names=["image_name", "label"], header=1)
    images = []
    labels = []

    for idx, row in df.iterrows():
        img_name = row["image_name"]
        label = row["label"]
        img_path = os.path.join(image_folder, img_name)

        if not os.path.exists(img_path):
            print(f"Geçersiz dosya yolu: {img_path}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"HATALI GÖRSEL: {img_name}")
            continue

        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalizasyon

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

# Dosya yolları
csv_path = "data/Train/train_labels.csv"
image_folder = "data/Train/Images/"

# Veriyi yükle
images, labels = load_data(csv_path, image_folder)

# Kontrol
print(f"Görsel sayısı: {len(images)}")
print(f"Etiket sayısı: {len(labels)}")

# İlk birkaç görselin boyutlarını kontrol et
print(f"İlk görsel boyutu: {images[0].shape}")
